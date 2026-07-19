from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import torch
from transformers import set_seed

from gcg.config import configure_logging, load_config
from gcg.data import index_range, prompt_chunks, read_prompts
from gcg.evaluation import (
    annotate_baseline_success,
    annotate_success,
    ever_group_success_rate_of,
    generate_batch,
    group_success_rate_of,
)
from gcg.model import build_initial_suffix, get_nonascii_toks
from gcg.optimizer import choose_suffixes
from gcg.records import (
    PromptState,
    initial_record,
    resume_indices,
    save_metadata,
    save_state,
    step_record,
)
from gcg.runtime import ColocatedRuntime
from gcg.suffix import SuffixManager

LOGGER = logging.getLogger(__name__)


def main() -> None:
    configure_logging()
    cfg = load_config()
    set_seed(int(cfg["seed"]))
    runtime = ColocatedRuntime(cfg)
    save_metadata(cfg, runtime)
    run_attack(cfg, runtime)


def run_attack(cfg: dict[str, Any], runtime: ColocatedRuntime) -> None:
    prompts = read_prompts(cfg)
    start_index, end_index = index_range(cfg, len(prompts))
    save_dir = Path(cfg["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    indices = resume_indices(save_dir, start_index, end_index)
    if not indices:
        LOGGER.info("No prompts to run for %s:%s", start_index, end_index)
        return

    initial_suffix, initial_token_id = build_initial_suffix(runtime.tokenizer, int(cfg["adv_len"]))
    runtime.activate_hf()
    not_allowed_tokens = get_nonascii_toks(runtime.tokenizer, runtime.device)

    for group_id, group_indices in prompt_chunks(indices, int(cfg["prompt_batch_size"])):
        LOGGER.info("group %s indices %s", group_id, group_indices)
        states = initialize_group(
            cfg,
            runtime,
            prompts,
            group_id,
            group_indices,
            initial_suffix,
            initial_token_id,
            save_dir,
        )
        optimize_group(cfg, runtime, states, not_allowed_tokens, save_dir)
        torch.cuda.empty_cache()


def initialize_group(
    cfg: dict[str, Any],
    runtime: ColocatedRuntime,
    prompts: list[str],
    group_id: int,
    group_indices: list[int],
    initial_suffix: str,
    initial_token_id: int,
    save_dir: Path,
) -> list[PromptState]:
    states = [
        make_prompt_state(
            cfg,
            runtime,
            prompts[index],
            index,
            group_id,
            len(group_indices),
            initial_suffix,
            initial_token_id,
        )
        for index in group_indices
    ]

    runtime.activate_vllm()
    baseline = generate_batch(cfg, runtime, [state.prompt for state in states])
    initial = generate_batch(cfg, runtime, [state.adv_prompt for state in states])
    runtime.activate_hf()

    annotate_baseline_success(cfg, runtime, states, baseline)
    annotate_success(cfg, runtime, states, initial, step=-1)
    group_success_rate = group_success_rate_of(initial)
    ever_group_success_rate = ever_group_success_rate_of(states)

    for state, baseline_result, initial_result in zip(states, baseline, initial):
        state.results["-1"] = initial_record(
            state,
            baseline_result,
            initial_result,
            group_success_rate,
            ever_group_success_rate,
        )
        save_state(save_dir, state)

    return states


def make_prompt_state(
    cfg: dict[str, Any],
    runtime: ColocatedRuntime,
    prompt: str,
    dataset_index: int,
    group_id: int,
    group_size: int,
    initial_suffix: str,
    initial_token_id: int,
) -> PromptState:
    suffix_manager = SuffixManager(runtime.tokenizer, prompt, int(cfg["adv_len"]))
    suffix_manager.update(adv_suffix=initial_suffix)
    suffix_manager.adv_token_id = initial_token_id
    return PromptState(
        dataset_index=dataset_index,
        group_id=group_id,
        group_size=group_size,
        prompt=prompt,
        suffix_manager=suffix_manager,
        results={},
    )


def optimize_group(
    cfg: dict[str, Any],
    runtime: ColocatedRuntime,
    states: list[PromptState],
    not_allowed_tokens: torch.Tensor,
    save_dir: Path,
) -> None:
    for step in range(int(cfg["steps"])):
        start_time = time.time()
        suffixes, losses, group_loss = choose_suffixes(cfg, runtime, states, not_allowed_tokens)
        for state, suffix in zip(states, suffixes):
            state.suffix_manager.update(adv_suffix=suffix)

        evaluations = None
        group_success_rate = 0.0
        ever_group_success_rate = ever_group_success_rate_of(states)
        if should_evaluate(cfg, step):
            runtime.activate_vllm()
            evaluations = generate_batch(cfg, runtime, [state.adv_prompt for state in states])
            runtime.activate_hf()
            annotate_success(cfg, runtime, states, evaluations, step)
            group_success_rate = group_success_rate_of(evaluations)
            ever_group_success_rate = ever_group_success_rate_of(states)

        duration = time.time() - start_time
        for i, state in enumerate(states):
            evaluation = evaluations[i] if evaluations is not None else None
            state.results[str(step)] = step_record(
                state,
                losses[i],
                group_loss,
                group_success_rate,
                ever_group_success_rate,
                evaluation,
                duration,
            )
            save_state(save_dir, state)

        if ever_group_success_rate >= float(cfg["success_rate_threshold"]):
            LOGGER.info("group %s reached success threshold", states[0].group_id)
            break


def should_evaluate(cfg: dict[str, Any], step: int) -> bool:
    return (step + 1) % int(cfg["eval_interval"]) == 0 or step == int(cfg["steps"]) - 1


if __name__ == "__main__":
    main()
