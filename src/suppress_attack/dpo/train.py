from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


Json = dict[str, Any]
Rows = list[Json]
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a LoRA DPO adapter from preference JSONL."
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name-or-path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-eval-samples", type=int)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--run-name", default="qwen3-0.6b-dpo")
    parser.add_argument("--wandb-project", default="llm-degeneration-mitigation")
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    return parser.parse_args()


def read_jsonl(path: Path) -> Rows:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise SystemExit(f"No DPO rows found in {path}")
    return rows


def limited(rows: Rows, limit: int | None) -> Rows:
    return rows[:limit] if limit is not None else rows


def load_preference_rows(args: argparse.Namespace) -> tuple[Rows, Rows]:
    train_rows = limited(
        read_jsonl(args.dataset_dir / "train.jsonl"), args.max_train_samples
    )
    eval_rows = limited(
        read_jsonl(args.dataset_dir / "validation.jsonl"), args.max_eval_samples
    )
    return train_rows, eval_rows


def read_summary(dataset_dir: Path) -> Json | None:
    path = dataset_dir / "summary.json"
    if not path.exists():
        return None
    return {"path": str(path), "content": json.loads(path.read_text(encoding="utf-8"))}


def row_counts(train_rows: Rows, eval_rows: Rows) -> Json:
    return {"train_rows": len(train_rows), "validation_rows": len(eval_rows)}


def metadata(args: argparse.Namespace, train_rows: Rows, eval_rows: Rows) -> Json:
    return {
        "model_name_or_path": args.model_name_or_path,
        "dataset": {
            "dataset_dir": str(args.dataset_dir),
            **row_counts(train_rows, eval_rows),
            "summary": read_summary(args.dataset_dir),
        },
        "training": {
            "num_train_epochs": args.num_train_epochs,
            "learning_rate": args.learning_rate,
            "beta": args.beta,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "per_device_eval_batch_size": args.per_device_eval_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "max_length": args.max_length,
            "seed": args.seed,
            "bf16": args.bf16,
        },
        "lora": {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "target_modules": LORA_TARGET_MODULES,
        },
        "wandb": {
            "project": args.wandb_project,
            "run_name": args.run_name,
        },
    }


def save_metadata(args: argparse.Namespace, train_rows: Rows, eval_rows: Rows) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "metadata.json"
    content = json.dumps(
        metadata(args, train_rows, eval_rows), ensure_ascii=False, indent=2
    )
    path.write_text(content + "\n", encoding="utf-8")


def configure_wandb(args: argparse.Namespace) -> None:
    os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
    os.environ.setdefault("WANDB_NAME", args.run_name)
    os.environ.setdefault("WANDB_DIR", str(args.output_dir))


def make_dpo_config(args: argparse.Namespace, torch: Any) -> Any:
    from trl import DPOConfig

    return DPOConfig(
        output_dir=str(args.output_dir),
        beta=args.beta,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_length=args.max_length,
        logging_steps=args.logging_steps,
        logging_first_step=True,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=args.seed,
        bf16=args.bf16,
        report_to="wandb",
        run_name=args.run_name,
        gradient_checkpointing=True,
        model_init_kwargs={
            "torch_dtype": torch.bfloat16 if args.bf16 else torch.float32,
            "trust_remote_code": True,
        },
    )


def make_lora_config(args: argparse.Namespace) -> Any:
    from peft import LoraConfig

    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LORA_TARGET_MODULES,
    )


def make_trainer(args: argparse.Namespace, train_rows: Rows, eval_rows: Rows) -> Any:
    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer
    from trl import DPOTrainer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return DPOTrainer(
        model=args.model_name_or_path,
        args=make_dpo_config(args, torch),
        train_dataset=Dataset.from_list(train_rows),
        eval_dataset=Dataset.from_list(eval_rows),
        processing_class=tokenizer,
        peft_config=make_lora_config(args),
    )


def main() -> None:
    args = parse_args()
    train_rows, eval_rows = load_preference_rows(args)
    print(json.dumps(row_counts(train_rows, eval_rows), ensure_ascii=False))
    save_metadata(args, train_rows, eval_rows)
    configure_wandb(args)
    trainer = make_trainer(args, train_rows, eval_rows)
    trainer.train()
    trainer.save_model(str(args.output_dir / "checkpoint-best"))


if __name__ == "__main__":
    main()
