from __future__ import annotations

import argparse
import json
from pathlib import Path

from .metrics import calibrate_threshold, evaluate_policy


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a SEVRA policy on paired outcomes.")
    parser.add_argument("--input", required=True, help="JSONL with paired outcome fields.")
    parser.add_argument("--threshold", type=float, help="Frozen gate threshold.")
    parser.add_argument("--calibrate", action="store_true", help="Calibrate on this file.")
    args = parser.parse_args()
    if (args.threshold is None) == (not args.calibrate):
        parser.error("provide exactly one of --threshold or --calibrate")

    rows = [json.loads(line) for line in Path(args.input).open() if line.strip()]
    fields = {
        "base_correct": [bool(row["base_correct"]) for row in rows],
        "verified_correct": [bool(row["verified_correct"]) for row in rows],
        "gate_scores": [float(row["gate_score"]) for row in rows],
        "verification_tokens": [int(row.get("verification_tokens", 0)) for row in rows],
    }
    if args.calibrate:
        threshold, metrics = calibrate_threshold(**fields)
    else:
        threshold = float(args.threshold)
        metrics = evaluate_policy(threshold=threshold, **fields)
    print(json.dumps({"threshold": threshold, **metrics.to_dict()}, indent=2))


if __name__ == "__main__":
    main()
