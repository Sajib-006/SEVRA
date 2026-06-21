from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from peft import PeftConfig, PeftModel
from sklearn.metrics import average_precision_score, roc_auc_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig

from sevra.gates import attempt_from_row, format_gate_input


def format_input(row: dict) -> str:
    return format_gate_input(attempt_from_row(row))


def load_rows(path: Path) -> tuple[list[dict], dict[str, list[dict]]]:
    rows = [json.loads(line) for line in path.open() if line.strip()]
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["example_id"])].append(row)
    incomplete = [
        example_id
        for example_id, example_rows in grouped.items()
        if not any(str(row["action"]) == "accept" for row in example_rows)
        or not any(str(row["action"]) == "active_verify" for row in example_rows)
    ]
    if incomplete:
        raise ValueError(f"Missing accept or active_verify rows for {len(incomplete)} examples.")
    return rows, grouped


def predict_probabilities(
    model_ref: str,
    examples: list[dict],
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    adapter_config = PeftConfig.from_pretrained(model_ref)
    tokenizer = AutoTokenizer.from_pretrained(model_ref, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    base_model = AutoModelForSequenceClassification.from_pretrained(
        adapter_config.base_model_name_or_path,
        num_labels=1,
        problem_type="multi_label_classification",
        quantization_config=quantization_config,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    base_model.config.pad_token_id = tokenizer.pad_token_id
    model = PeftModel.from_pretrained(base_model, model_ref)
    model.eval()

    probabilities = []
    with torch.inference_mode():
        for start in range(0, len(examples), batch_size):
            batch = examples[start : start + batch_size]
            encoded = tokenizer(
                [format_input(row) for row in batch],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(model.device) for key, value in encoded.items()}
            logits = model(**encoded).logits.reshape(-1)
            probabilities.extend(torch.sigmoid(logits).float().cpu().tolist())
    return np.asarray(probabilities)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a frozen selective-verification gate.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--predictions-output",
        help="Optional JSONL path for per-example probabilities and policy outcomes.",
    )
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=1536)
    args = parser.parse_args()

    _, grouped = load_rows(Path(args.input))
    example_ids = sorted(grouped)
    representative_rows = [
        next(row for row in grouped[example_id] if str(row["action"]) == "active_verify")
        for example_id in example_ids
    ]
    probabilities = predict_probabilities(
        args.model_dir,
        representative_rows,
        args.batch_size,
        args.max_length,
    )

    threshold = args.threshold
    if threshold is None:
        local_metrics = Path(args.model_dir) / "final_metrics.json"
        metrics_path = (
            local_metrics
            if local_metrics.exists()
            else Path(hf_hub_download(repo_id=args.model_dir, filename="final_metrics.json"))
        )
        threshold = float(json.loads(metrics_path.read_text())["best_dev_policy"]["threshold"])

    base_correct = []
    verify_correct = []
    policy_correct = []
    policy_flips = []
    policy_tokens = []
    sampled_oracle = []
    helpful_targets = []
    repeated_probabilities = []
    prediction_rows = []
    chosen_examples = 0
    for example_id, probability in zip(example_ids, probabilities):
        example_rows = grouped[example_id]
        accept = next(row for row in example_rows if str(row["action"]) == "accept")
        verify_rows = [row for row in example_rows if str(row["action"]) == "active_verify"]
        choose_verify = bool(probability >= threshold)
        chosen_examples += int(choose_verify)
        base_correct.append(bool(accept["base_correct"]))
        verify_outcomes = []
        verify_tokens = []
        verify_flips = []
        helpful_fixes = []
        for row in verify_rows:
            final_correct = bool(row["final_correct"])
            verify_outcomes.append(final_correct)
            verify_tokens.append(int(row.get("action_actual_tokens", 0)))
            verify_flips.append(bool(accept["base_correct"]) and not final_correct)
            helpful_fixes.append(bool(row["helpful_fix"]))
            verify_correct.append(final_correct)
            policy_correct.append(final_correct if choose_verify else bool(accept["base_correct"]))
            policy_flips.append(
                choose_verify and bool(accept["base_correct"]) and not final_correct
            )
            policy_tokens.append(int(row.get("action_actual_tokens", 0)) if choose_verify else 0)
            sampled_oracle.append(bool(accept["base_correct"]) or final_correct)
            helpful_targets.append(bool(row["helpful_fix"]))
            repeated_probabilities.append(float(probability))
        prediction_rows.append(
            {
                "example_id": accept["example_id"],
                "probability": float(probability),
                "threshold": float(threshold),
                "choose_verify": choose_verify,
                "base_correct": bool(accept["base_correct"]),
                "verify_accuracy": float(np.mean(verify_outcomes)),
                "verify_avg_action_tokens": float(np.mean(verify_tokens)),
                "verify_harmful_flip_rate": float(np.mean(verify_flips)),
                "helpful_fix_rate": float(np.mean(helpful_fixes)),
                "features": accept.get("features", {}),
                "base_finalizer_used": bool(accept.get("base_finalizer_used", False)),
                "base_done_reason": accept.get("base_usage", {}).get("done_reason"),
                "base_actual_tokens": int(accept.get("base_actual_tokens", 0)),
            }
        )

    helpful_targets_array = np.asarray(helpful_targets)
    repeated_probabilities_array = np.asarray(repeated_probabilities)
    summary = {
        "examples": len(example_ids),
        "active_verify_rollouts": len(verify_correct),
        "threshold": threshold,
        "base_accuracy": float(np.mean(base_correct)),
        "always_verify_accuracy": float(np.mean(verify_correct)),
        "always_verify_avg_action_tokens": float(
            np.mean(
                [
                    int(row.get("action_actual_tokens", 0))
                    for rows in grouped.values()
                    for row in rows
                    if str(row["action"]) == "active_verify"
                ]
            )
        ),
        "policy_accuracy": float(np.mean(policy_correct)),
        "policy_intervention_rate": chosen_examples / len(example_ids),
        "policy_avg_action_tokens": float(np.mean(policy_tokens)),
        "policy_harmful_flip_rate": float(np.mean(policy_flips)),
        "sampled_selective_oracle_accuracy": float(np.mean(sampled_oracle)),
    }
    if len(np.unique(helpful_targets_array)) > 1:
        summary["helpful_fix_auroc"] = float(
            roc_auc_score(helpful_targets_array, repeated_probabilities_array)
        )
        summary["helpful_fix_auprc"] = float(
            average_precision_score(helpful_targets_array, repeated_probabilities_array)
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2))
    predictions_path = (
        Path(args.predictions_output)
        if args.predictions_output
        else output_path.with_suffix(".predictions.jsonl")
    )
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    with predictions_path.open("w") as handle:
        for row in prediction_rows:
            handle.write(json.dumps(row) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
