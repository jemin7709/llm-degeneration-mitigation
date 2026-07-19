#!/usr/bin/env python3
"""Normalize the three reviewed degeneration-mitigation preference datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


VARIANTS = {
    "naive": {"train": 11_400, "validation": 600},
    "math_solution": {"train": 6_356, "validation": 334},
    "data_filtering": {"train": 3_443, "validation": 181},
}
REQUIRED_FIELDS = ("prompt", "chosen", "rejected")


def validate_messages(name: str, messages: object, expected_role: str) -> None:
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"{name} must be a non-empty message list")
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError(f"{name} contains a non-object message")
        if message.get("role") != expected_role or not isinstance(
            message.get("content"), str
        ):
            raise ValueError(f"invalid {name} message")


def normalize_row(variant: str, row: dict) -> dict:
    missing = [field for field in REQUIRED_FIELDS if field not in row]
    if missing:
        raise ValueError(f"missing fields: {missing}")

    validate_messages("prompt", row["prompt"], "user")
    validate_messages("chosen", row["chosen"], "assistant")
    validate_messages("rejected", row["rejected"], "assistant")

    content = {field: row[field] for field in REQUIRED_FIELDS}
    canonical = json.dumps(
        [variant, content], ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "id": f"{variant}-{hashlib.sha256(canonical).hexdigest()[:16]}",
        "variant": variant,
        **content,
    }


def normalize_file(source: Path, destination: Path, variant: str) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    count = 0
    with source.open(encoding="utf-8") as src, destination.open(
        "w", encoding="utf-8"
    ) as dst:
        for line_number, line in enumerate(src, start=1):
            try:
                normalized = normalize_row(variant, json.loads(line))
            except (json.JSONDecodeError, ValueError) as error:
                raise ValueError(f"{source}:{line_number}: {error}") from error
            if normalized["id"] in seen:
                raise ValueError(f"{source}:{line_number}: duplicate content id")
            seen.add(normalized["id"])
            dst.write(json.dumps(normalized, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_root", type=Path)
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()

    counts: dict[str, dict[str, int]] = {}
    for variant, splits in VARIANTS.items():
        counts[variant] = {}
        for split, expected in splits.items():
            count = normalize_file(
                args.source_root / variant / f"{split}.jsonl",
                args.output_root / variant / f"{split}.jsonl",
                variant,
            )
            if count != expected:
                raise ValueError(
                    f"{variant}/{split}: expected {expected} rows, found {count}"
                )
            counts[variant][split] = count

    metadata = {
        "base_model": "Qwen/Qwen3-0.6B",
        "source_dataset": "MATH train",
        "attack_method": "LoopLLM multi-prompt GCG",
        "attack_seed": 23,
        "format": "DPO chat messages",
        "transformation": (
            "Kept full prompt/chosen/rejected messages, removed all other fields, "
            "and added a content-derived id and variant."
        ),
        "counts": counts,
    }
    (args.output_root / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(counts, sort_keys=True))


if __name__ == "__main__":
    main()
