from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .data import index_range, read_prompts
from .model import generation_config_metadata

if TYPE_CHECKING:
    from .runtime import ColocatedRuntime
    from .suffix import SuffixManager


@dataclass
class PromptState:
    dataset_index: int
    group_id: int
    group_size: int
    prompt: str
    suffix_manager: SuffixManager
    results: dict[str, dict[str, Any]]
    first_success_step: int | None = None

    @property
    def adv_prompt(self) -> str:
        return f"{self.prompt} {self.suffix_manager.adv_suffix.strip()}"


def initial_record(
    state: PromptState,
    baseline: dict[str, Any],
    initial: dict[str, Any],
    group_success_rate: float,
    ever_group_success_rate: float,
) -> dict[str, Any]:
    return {
        "baseline_prompt": state.prompt,
        "baseline_answer": baseline["answer"],
        "baseline_answers": baseline["answers"],
        "baseline_output_len": baseline["avg_len"],
        "baseline_output_lengths": baseline["output_lengths"],
        "baseline_cap_hits": baseline["cap_hits"],
        "baseline_success_rate": baseline["success_rate"],
        "baseline_success": baseline["success"],
        "baseline_entropy_low": baseline["entropy_low"],
        "prompt": state.adv_prompt,
        "answer": initial["answer"],
        "answers": initial["answers"],
        "output_len": initial["avg_len"],
        "output_lengths": initial["output_lengths"],
        "cap_hits": initial["cap_hits"],
        "success_rate": initial["success_rate"],
        "avg_len": initial["avg_len"],
        "adv_suffix": state.suffix_manager.adv_suffix,
        "adv_prompt": state.adv_prompt,
        "group_id": state.group_id,
        "group_size": state.group_size,
        "group_success_rate": group_success_rate,
        "ever_group_success_rate": ever_group_success_rate,
        "group_current_loss": 0.0,
        "evaluated": True,
        "success": initial["success"],
        "entropy_low": initial["entropy_low"],
        "first_success_step": state.first_success_step,
        "time": 0.0,
    }


def step_record(
    state: PromptState,
    current_loss: float,
    group_current_loss: float,
    group_success_rate: float,
    ever_group_success_rate: float,
    evaluation: dict[str, Any] | None,
    duration: float,
) -> dict[str, Any]:
    record = {
        "prompt": state.prompt,
        "adv_suffix": state.suffix_manager.adv_suffix,
        "adv_prompt": state.adv_prompt,
        "current_losses": current_loss,
    }
    if evaluation is not None:
        record.update(
            {
                "answer": evaluation["answer"],
                "answers": evaluation["answers"],
                "output_lengths": evaluation["output_lengths"],
                "cap_hits": evaluation["cap_hits"],
                "success_rate": evaluation["success_rate"],
                "avg_len": evaluation["avg_len"],
            }
        )
    record.update(
        {
            "time": duration,
            "group_id": state.group_id,
            "group_size": state.group_size,
            "group_success_rate": group_success_rate,
            "ever_group_success_rate": ever_group_success_rate,
            "group_current_loss": group_current_loss,
            "evaluated": evaluation is not None,
            "success": evaluation["success"] if evaluation is not None else False,
            "entropy_low": evaluation["entropy_low"] if evaluation is not None else False,
            "first_success_step": state.first_success_step,
        }
    )
    return record


def save_state(save_dir: Path, state: PromptState) -> None:
    with (save_dir / f"res_{state.dataset_index}.json").open("w", encoding="utf-8") as handle:
        json.dump(state.results, handle, indent=2, ensure_ascii=False)


def save_metadata(cfg: dict[str, Any], runtime: ColocatedRuntime) -> None:
    start, end = index_range(cfg, len(read_prompts(cfg)))
    metadata = {
        "config": cfg,
        "generation_config": generation_config_metadata(runtime.model.generation_config),
    }
    path = Path(cfg["save_dir"]) / f"metadata_{start}_{end}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def resume_indices(save_dir: Path, start_index: int, end_index: int) -> list[int]:
    existing = sorted(
        index
        for index in range(start_index, end_index)
        if (save_dir / f"res_{index}.json").exists()
    )
    rerun_from = existing[-1] if existing else start_index
    return list(range(rerun_from, end_index))
