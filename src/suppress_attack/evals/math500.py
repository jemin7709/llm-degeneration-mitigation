from __future__ import annotations

import json
import os
from pathlib import Path

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc

NUM_SAMPLES = 4
PROMPT_SOURCE = os.environ["MATH500_PROMPT_SOURCE"]
TASK_NAME = f"math500_{PROMPT_SOURCE}"
ATTACK_DIR = Path(os.environ["MATH500_ATTACK_DIR"])


def load_attack_result(index: int) -> dict:
    with open(ATTACK_DIR / f"res_{index}.json", encoding="utf-8") as f:
        return json.load(f)


def baseline_prompt(line: dict) -> str:
    return load_attack_result(int(line["__index"]))["-1"]["baseline_prompt"].strip()


def attack_prompt(line: dict) -> str:
    result = load_attack_result(int(line["__index"]))
    last_step = max(int(key) for key in result if key != "-1")
    first_success_step = result[str(last_step)]["first_success_step"]
    step = int(first_success_step) if first_success_step is not None else last_step
    return result[str(step)]["adv_prompt"].strip()


def math_500_prompt(line: dict, task_name: str | None = None) -> Doc:
    if PROMPT_SOURCE == "baseline":
        query = baseline_prompt(line)
    elif PROMPT_SOURCE == "attack":
        query = attack_prompt(line)
    else:
        raise ValueError("MATH500_PROMPT_SOURCE must be 'baseline' or 'attack'")
    return Doc(
        task_name=task_name or "",
        query=query,
        choices=[f"ANSWER: {line['solution']}"],
        gold_index=0,
        specific={
            "solution": line.get("solution", ""),
            "answer": line["answer"],
            "unique_id": line.get("unique_id", ""),
        },
        id=line.get("unique_id", str(line.get("__index", ""))),
    )


math_500 = LightevalTaskConfig(
    name=TASK_NAME,
    prompt_function=math_500_prompt,
    hf_repo="HuggingFaceH4/MATH-500",
    hf_subset="default",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    metrics=[
        Metrics.pass_at_k_math(sample_params={"k": 1, "n": NUM_SAMPLES}),
        Metrics.pass_at_k_math(sample_params={"k": NUM_SAMPLES, "n": NUM_SAMPLES}),
    ],
    version=6,
)

TASKS_TABLE = [math_500]
