# LLM Degeneration Mitigation

Code, preference datasets, and LoRA adapters for reducing excessive output
length and token-level repetition in
[Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) with DPO.

## Overview

1. Generate degeneration-inducing prompts with
   [LoopLLM](https://github.com/neuron-insight-lab/LoopLLM).
2. Build preference pairs using three data-construction strategies.
3. Train LoRA adapters with DPO.
4. Evaluate the trained models on MATH-500, IFEval, and MuSR.

## Main results

The LoopLLM attack reached a 54.9% success rate on the MATH training split and
54.2% on the 500-example test split.

| Model | Baseline prompt | Attack prompt | Drop |
| --- | ---: | ---: | ---: |
| Qwen3-0.6B | 71.10% | 66.45% | 4.65 pp |
| Naive | 72.30% | 70.00% | 2.30 pp |
| MATH solution | 68.75% | 68.45% | 0.30 pp |
| Filtering | **72.45%** | **71.20%** | 1.25 pp |

MATH-500 pass@1 (`n=4`). Filtering achieved the highest attacked-prompt
accuracy. On the base model, the attack increased mean output length from
4,212 to 4,908 tokens and Rep-4 from 0.347 to 0.380; with Filtering, both
remained stable (4,716 to 4,672 tokens and 0.384 to 0.377).

## Preference datasets

| Variant | Train | Validation | Total |
| --- | ---: | ---: | ---: |
| Naive | 11,400 | 600 | 12,000 |
| MATH solution | 6,356 | 334 | 6,690 |
| Filtering | 3,443 | 181 | 3,624 |

Each JSONL row contains `id`, `variant`, `prompt`, `chosen`, and `rejected`.
The source problems come from the [MATH dataset](https://github.com/hendrycks/math).

## Usage

Install the Python 3.13 environment:

```bash
uv sync
```

Run the attack with a local MATH JSONL file:

```bash
uv run python src/attack/multi_prompt_gcg.py \
  --config configs/attack/multi_prompt_gcg.yaml \
  data_path=/path/to/math/train.jsonl
```

Train an adapter:

```bash
uv run python src/suppress_attack/dpo/train.py \
  --dataset-dir data/processed/naive \
  --output-dir outputs/dpo/naive
```

Raw source data, raw attack outputs, and merged model weights are not included.
Licensed under the Apache License 2.0. See `LICENSE`.
