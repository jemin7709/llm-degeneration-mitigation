from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from .model import as_list, chat_prompt_ids, vllm_tokens_prompt

if TYPE_CHECKING:
    from .records import PromptState
    from .runtime import ColocatedRuntime


def generate_batch(cfg: dict[str, Any], runtime: ColocatedRuntime, prompts: list[str]) -> list[dict[str, Any]]:
    from vllm import SamplingParams

    generation_config = runtime.model.generation_config
    sampling_params = SamplingParams(
        n=int(cfg["sample_times"]),
        max_tokens=int(cfg["max_new_tokens"]),
        seed=int(cfg["seed"]),
        temperature=float(cfg["temperature"]),
        top_p=float(cfg["top_p"]),
        top_k=int(cfg["top_k"]),
        stop_token_ids=as_list(generation_config.eos_token_id),
    )
    outputs = runtime.llm.generate(
        [vllm_tokens_prompt(runtime.tokenizer, prompt) for prompt in prompts],
        sampling_params,
        use_tqdm=False,
    )
    return [summarize_output(int(cfg["max_new_tokens"]), output) for output in outputs]


def summarize_output(max_new_tokens: int, request_output) -> dict[str, Any]:
    answers = [output.text.strip() for output in request_output.outputs]
    lengths = [len(output.token_ids) for output in request_output.outputs]
    cap_hits = [length >= max_new_tokens - 5 for length in lengths]
    longest = max(range(len(lengths)), key=lengths.__getitem__)
    return {
        "answer": answers[longest],
        "answers": answers,
        "output_lengths": lengths,
        "cap_hits": cap_hits,
        "avg_len": sum(lengths) / len(lengths),
        "success_rate": sum(cap_hits) / len(cap_hits),
    }


def annotate_success(
    cfg: dict[str, Any],
    runtime: ColocatedRuntime,
    states: list[PromptState],
    evaluations: list[dict[str, Any]],
    step: int,
) -> None:
    for state, evaluation in zip(states, evaluations):
        state.suffix_manager.update(answer=evaluation["answer"])
        entropy_low = False
        if evaluation["success_rate"] >= float(cfg["prompt_success_rate_threshold"]):
            input_ids = state.suffix_manager.get_input_ids().to(runtime.device).unsqueeze(0)
            entropy_low = is_entropy_low(runtime.model, input_ids, float(cfg["entropy_threshold"]))
        set_success_fields(cfg, evaluation, entropy_low)
        if evaluation["success"] and state.first_success_step is None:
            state.first_success_step = step


def annotate_baseline_success(
    cfg: dict[str, Any],
    runtime: ColocatedRuntime,
    states: list[PromptState],
    evaluations: list[dict[str, Any]],
) -> None:
    for state, evaluation in zip(states, evaluations):
        entropy_low = False
        if evaluation["success_rate"] >= float(cfg["prompt_success_rate_threshold"]):
            input_ids = chat_prompt_ids(
                runtime.tokenizer,
                state.prompt,
                assistant_content=evaluation["answer"],
                return_tensors="pt",
            ).to(runtime.device)
            entropy_low = is_entropy_low(runtime.model, input_ids, float(cfg["entropy_threshold"]))
        set_success_fields(cfg, evaluation, entropy_low)


def set_success_fields(cfg: dict[str, Any], evaluation: dict[str, Any], entropy_low: bool) -> None:
    evaluation["entropy_low"] = entropy_low
    evaluation["success"] = (
        evaluation["success_rate"] >= float(cfg["prompt_success_rate_threshold"])
        and entropy_low
    )


@torch.inference_mode()
def is_entropy_low(model, input_ids: torch.Tensor, threshold: float) -> bool:
    logits = model(input_ids).logits
    logits_ans = logits[0, -100:, :]
    probs = torch.softmax(logits_ans, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
    return entropy.mean(dim=-1).item() < threshold


def group_success_rate_of(evaluations: list[dict[str, Any]]) -> float:
    return sum(evaluation["success"] for evaluation in evaluations) / len(evaluations)


def ever_group_success_rate_of(states: list[PromptState]) -> float:
    return sum(state.first_success_step is not None for state in states) / len(states)
