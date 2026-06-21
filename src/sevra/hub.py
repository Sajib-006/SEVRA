from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .gates import format_gate_input
from .schema import Attempt


@dataclass
class HuggingFaceGate:
    """A lazily loaded PEFT recoverability gate hosted on Hugging Face."""

    tokenizer: Any
    model: Any
    threshold: float
    max_length: int = 1536

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str,
        *,
        threshold: float | None = None,
        device_map: str | None = "auto",
        load_in_4bit: bool = False,
        max_length: int = 1536,
    ) -> HuggingFaceGate:
        """Load a public/local SEVRA adapter and its frozen operating threshold."""

        try:
            import torch
            from huggingface_hub import hf_hub_download
            from peft import PeftConfig, PeftModel
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
                BitsAndBytesConfig,
            )
        except ImportError as exc:
            raise ImportError(
                'HuggingFaceGate requires the training extra: pip install "sevra[train]"'
            ) from exc

        adapter = PeftConfig.from_pretrained(repo_id)
        tokenizer = AutoTokenizer.from_pretrained(repo_id, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        quantization_config = None
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )

        base = AutoModelForSequenceClassification.from_pretrained(
            adapter.base_model_name_or_path,
            num_labels=1,
            problem_type="multi_label_classification",
            quantization_config=quantization_config,
            torch_dtype="auto",
            device_map=device_map,
        )
        base.config.pad_token_id = tokenizer.pad_token_id
        model = PeftModel.from_pretrained(base, repo_id).eval()

        if threshold is None:
            local_metrics = Path(repo_id) / "final_metrics.json"
            metrics_path = (
                local_metrics
                if local_metrics.exists()
                else Path(hf_hub_download(repo_id=repo_id, filename="final_metrics.json"))
            )
            metrics = json.loads(metrics_path.read_text())
            threshold = float(metrics["best_dev_policy"]["threshold"])

        return cls(
            tokenizer=tokenizer,
            model=model,
            threshold=float(threshold),
            max_length=max_length,
        )

    def score(self, attempt: Attempt) -> float:
        """Return the predicted probability that active verification is a helpful fix."""

        import torch

        encoded = self.tokenizer(
            format_gate_input(attempt),
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        device = next(self.model.parameters()).device
        encoded = {name: value.to(device) for name, value in encoded.items()}
        with torch.inference_mode():
            logit = self.model(**encoded).logits.reshape(-1)[0]
        return float(torch.sigmoid(logit).cpu())
