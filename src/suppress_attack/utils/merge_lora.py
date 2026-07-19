from __future__ import annotations

import argparse
from typing import Any, cast

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, args.adapter)
    model = cast(Any, model).merge_and_unload()
    model.save_pretrained(args.output, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.save_pretrained(args.output)


if __name__ == "__main__":
    main()
