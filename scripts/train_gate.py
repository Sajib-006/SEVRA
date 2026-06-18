from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import average_precision_score, roc_auc_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

from sevra.gates import attempt_from_row, format_gate_input


def format_input(row: dict) -> str:
    return format_gate_input(attempt_from_row(row))


def load_active_verify_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.open() if line.strip()]
    rows = [row for row in rows if str(row["action"]) == "active_verify"]
    if not rows:
        raise ValueError("No active_verify rows found.")
    if any(bool(row.get("intervention_response_empty", False)) for row in rows):
        raise ValueError("Dataset contains empty active_verify responses.")
    return rows


def split_by_example(rows: list[dict], dev_fraction: float, seed: int) -> tuple[list[dict], list[dict], dict]:
    example_ids = sorted({str(row["example_id"]) for row in rows})
    rng = random.Random(seed)
    rng.shuffle(example_ids)
    dev_count = max(1, round(len(example_ids) * dev_fraction))
    dev_ids = set(example_ids[:dev_count])
    train_rows = [row for row in rows if str(row["example_id"]) not in dev_ids]
    dev_rows = [row for row in rows if str(row["example_id"]) in dev_ids]
    return train_rows, dev_rows, {
        "train_example_ids": sorted(set(example_ids) - dev_ids),
        "dev_example_ids": sorted(dev_ids),
    }


class BinaryWeightedTrainer(Trainer):
    def __init__(self, *args, pos_weight: torch.Tensor, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weight = pos_weight

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = F.binary_cross_entropy_with_logits(
            outputs.logits.reshape(-1),
            labels.float().reshape(-1),
            pos_weight=self.pos_weight.to(outputs.logits.device),
        )
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_prediction):
    logits, labels = eval_prediction
    probabilities = 1.0 / (1.0 + np.exp(-np.asarray(logits).reshape(-1)))
    target = np.asarray(labels).reshape(-1)
    metrics = {"accuracy": float(np.mean((probabilities >= 0.5) == target))}
    if len(np.unique(target)) > 1:
        metrics["auroc"] = float(roc_auc_score(target, probabilities))
        metrics["auprc"] = float(average_precision_score(target, probabilities))
    return metrics


def policy_curve(rows: list[dict], probabilities: np.ndarray) -> list[dict]:
    curve = []
    thresholds = sorted(
        {
            0.0,
            1.0,
            *np.linspace(0.0, 1.0, 101).tolist(),
            *np.quantile(probabilities, np.linspace(0.0, 1.0, 101)).tolist(),
        }
    )
    for threshold in thresholds:
        choose_verify = probabilities >= threshold
        final_correct = np.asarray(
            [
                bool(row["final_correct"]) if choose else bool(row["base_correct"])
                for row, choose in zip(rows, choose_verify)
            ]
        )
        flips = np.asarray(
            [
                bool(row["base_correct"]) and choose and not bool(row["final_correct"])
                for row, choose in zip(rows, choose_verify)
            ]
        )
        action_tokens = np.asarray(
            [
                int(row.get("action_actual_tokens", 0)) if choose else 0
                for row, choose in zip(rows, choose_verify)
            ]
        )
        curve.append(
            {
                "threshold": float(threshold),
                "accuracy": float(final_correct.mean()),
                "intervention_rate": float(choose_verify.mean()),
                "harmful_flip_rate": float(flips.mean()),
                "avg_action_tokens": float(action_tokens.mean()),
            }
        )
    return curve


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a selective active-verification gate with QLoRA.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--dev-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-4bit", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_active_verify_rows(Path(args.input))
    train_rows, dev_rows, split = split_by_example(rows, args.dev_fraction, args.seed)
    (output_dir / "split.json").write_text(json.dumps(split, indent=2))

    positives = sum(bool(row["helpful_fix"]) for row in train_rows)
    negatives = len(train_rows) - positives
    if positives == 0:
        raise ValueError("Training split contains no helpful active-verification fixes.")
    pos_weight = torch.tensor([negatives / positives], dtype=torch.float32)
    print(
        f"Active-verify examples: train={len(train_rows)}, dev={len(dev_rows)}, "
        f"train helpful fixes={positives}, pos_weight={pos_weight.item():.3f}"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def encode(row):
        encoded = tokenizer(format_input(row), truncation=True, max_length=args.max_length)
        encoded["labels"] = [float(row["helpful_fix"])]
        return encoded

    train_dataset = Dataset.from_list(train_rows).map(encode, remove_columns=list(train_rows[0].keys()))
    dev_dataset = Dataset.from_list(dev_rows).map(encode, remove_columns=list(dev_rows[0].keys()))

    quantization_config = None
    if not args.no_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=1,
        problem_type="multi_label_classification",
        quantization_config=quantization_config,
        torch_dtype=torch.float16,
        device_map="auto" if quantization_config else None,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    if quantization_config:
        model = prepare_model_for_kbit_training(model)

    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            modules_to_save=["score"],
        ),
    )
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        fp16=True,
        bf16=False,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="eval_auprc",
        greater_is_better=True,
        report_to="none",
        seed=args.seed,
        gradient_checkpointing=True,
        remove_unused_columns=True,
    )

    trainer = BinaryWeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        pos_weight=pos_weight,
    )
    trainer.train()
    metrics = trainer.evaluate()
    prediction = trainer.predict(dev_dataset)
    probabilities = 1.0 / (1.0 + np.exp(-np.asarray(prediction.predictions).reshape(-1)))
    curve = policy_curve(dev_rows, probabilities)
    best_policy = max(curve, key=lambda point: (point["accuracy"], -point["avg_action_tokens"]))

    base_accuracy = float(np.mean([bool(row["base_correct"]) for row in dev_rows]))
    verify_accuracy = float(np.mean([bool(row["final_correct"]) for row in dev_rows]))
    oracle_accuracy = float(
        np.mean([bool(row["base_correct"]) or bool(row["final_correct"]) for row in dev_rows])
    )
    summary = {
        **metrics,
        "base_accuracy": base_accuracy,
        "always_verify_accuracy": verify_accuracy,
        "always_verify_avg_action_tokens": float(
            np.mean([int(row.get("action_actual_tokens", 0)) for row in dev_rows])
        ),
        "selective_oracle_accuracy": oracle_accuracy,
        "best_dev_policy": best_policy,
    }
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    (output_dir / "policy_curve.json").write_text(json.dumps(curve, indent=2))
    (output_dir / "final_metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
