from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn as nn

if TYPE_CHECKING:
    from .records import PromptState
    from .runtime import ColocatedRuntime
    from .suffix import SuffixManager


def choose_suffixes(
    cfg: dict[str, Any],
    runtime: ColocatedRuntime,
    states: list[PromptState],
    not_allowed_tokens: torch.Tensor,
) -> tuple[list[str], list[float], float]:
    suffixes = []
    losses = []
    for state in states:
        suffix, loss = choose_prompt_suffix(cfg, runtime, state, not_allowed_tokens)
        suffixes.append(suffix)
        losses.append(loss)
    return suffixes, losses, float(sum(losses))


def choose_prompt_suffix(
    cfg: dict[str, Any],
    runtime: ColocatedRuntime,
    state: PromptState,
    not_allowed_tokens: torch.Tensor,
) -> tuple[str, float]:
    input_ids = state.suffix_manager.get_input_ids().to(runtime.device)
    grad = compute_prompt_gradient(runtime.model, input_ids, state.suffix_manager)
    control_tokens = input_ids[state.suffix_manager._control_slice].to(runtime.device)

    with torch.no_grad():
        candidate_tokens = sample_control(
            control_tokens,
            grad,
            int(cfg["num_candidate"]),
            int(cfg["topk"]),
            not_allowed_tokens,
        )
        candidate_suffixes, candidate_ids = get_filtered_candidates(
            runtime.tokenizer,
            candidate_tokens,
            control_tokens,
        )
        losses = get_all_losses(
            runtime.model,
            runtime.tokenizer,
            input_ids,
            candidate_ids,
            state.suffix_manager,
            int(cfg["once_forward_batch"]),
        )
        best = int(losses.argmin().item())
        return candidate_suffixes[best], float(losses[best].item())


def compute_prompt_gradient(model, input_ids: torch.Tensor, suffix_manager: SuffixManager) -> torch.Tensor:
    control_slice = suffix_manager._control_slice
    target_slice = suffix_manager._target_slice
    special_id = suffix_manager.adv_token_id
    embed_weights = model.get_input_embeddings().weight

    one_hot = torch.zeros(
        input_ids[control_slice].shape[0],
        embed_weights.shape[0],
        device=model.device,
        dtype=embed_weights.dtype,
    )
    one_hot.scatter_(1, input_ids[control_slice].unsqueeze(1), 1)
    one_hot.requires_grad_()

    embeds = embed_weights[input_ids].detach()
    full_embeds = torch.cat(
        [
            embeds[: control_slice.start],
            one_hot @ embed_weights,
            embeds[control_slice.stop :],
        ],
        dim=0,
    ).unsqueeze(0)
    logits = model(inputs_embeds=full_embeds, use_cache=False).logits
    loss = target_loss(logits, target_slice, special_id)
    loss.backward()

    assert one_hot.grad is not None
    grad = one_hot.grad.clone()
    grad = grad / grad.norm(dim=-1, keepdim=True)
    model.zero_grad(set_to_none=True)
    return grad


def target_loss(logits: torch.Tensor, target_slice: slice, special_id: int) -> torch.Tensor:
    logits_t = logits[:, target_slice.start - 1 : -1, :]
    probs = torch.softmax(logits_t, dim=-1)
    special_probs = probs[:, :, special_id]
    return nn.BCELoss(reduction="none")(
        special_probs,
        torch.ones_like(special_probs),
    ).mean(dim=-1)


@torch.inference_mode()
def get_all_losses(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    candidate_ids: torch.Tensor,
    suffix_manager: SuffixManager,
    batch_size: int,
) -> torch.Tensor:
    control_slice = suffix_manager._control_slice
    target_slice = suffix_manager._target_slice
    special_id = suffix_manager.adv_token_id

    base_ids = input_ids.to(model.device)
    candidates = candidate_ids.to(model.device)
    ids = base_ids.unsqueeze(0).expand(candidates.shape[0], -1).clone()
    ids[:, control_slice] = candidates

    losses = []
    for start in range(0, ids.shape[0], batch_size):
        batch_ids = ids[start : start + batch_size]
        logits = model(input_ids=batch_ids, use_cache=False).logits
        losses.append(target_loss(logits, target_slice, special_id))
    return torch.cat(losses, dim=0)


def sample_control(
    control_tokens: torch.Tensor,
    grad: torch.Tensor,
    num_candidates: int,
    topk: int,
    not_allowed_tokens: torch.Tensor,
) -> torch.Tensor:
    grad[:, not_allowed_tokens] = np.inf
    top_indices = (-grad).topk(topk, dim=1).indices
    base = control_tokens.to(grad.device).repeat(num_candidates, 1)
    positions = torch.arange(
        0,
        len(control_tokens),
        len(control_tokens) / num_candidates,
        device=grad.device,
    ).type(torch.int64)
    values = torch.gather(
        top_indices[positions],
        1,
        torch.randint(0, topk, (num_candidates, 1), device=grad.device),
    )
    return base.scatter_(1, positions.unsqueeze(-1), values).unique(dim=0)


def get_filtered_candidates(
    tokenizer,
    candidate_tokens: torch.Tensor,
    current_tokens: torch.Tensor,
) -> tuple[list[str], torch.Tensor]:
    strings = []
    ids = []
    seen = set()
    expected_len = len(current_tokens)
    current = tokenizer.decode(current_tokens, skip_special_tokens=True)

    for candidate in candidate_tokens:
        text = tokenizer.decode(candidate, skip_special_tokens=True)
        encoded = tokenizer(text, add_special_tokens=False).input_ids
        if text != current and len(encoded) == expected_len and text not in seen:
            strings.append(text)
            ids.append(encoded)
            seen.add(text)

    if not strings:
        raise RuntimeError("no valid GCG candidates")
    return strings, torch.tensor(ids, device=candidate_tokens.device, dtype=torch.long)
