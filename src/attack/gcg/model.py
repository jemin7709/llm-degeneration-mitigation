from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

if TYPE_CHECKING:
    from vllm.inputs import TokensPrompt


def load_model_and_tokenizer(model_name: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="cpu",
        trust_remote_code=True,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    model.generation_config.pad_token_id = model.generation_config.pad_token_id or tokenizer.pad_token_id
    model.generation_config.do_sample = True
    model.generation_config.temperature = 0.6
    model.generation_config.top_p = 0.95
    model.generation_config.top_k = 20
    return model, tokenizer


def warmup_cuda(device: torch.device) -> None:
    free_bytes, _ = torch.cuda.mem_get_info(device)
    warmup_bytes = min(1024**3, max(1, free_bytes - 512 * 1024**2))
    dummy_tensor = torch.ones(warmup_bytes, dtype=torch.uint8, device=device)
    torch.cuda.synchronize()
    time.sleep(1)
    del dummy_tensor
    torch.cuda.empty_cache()


def chat_prompt_ids(
    tokenizer,
    user_content: str,
    assistant_content: str | None = None,
    add_generation_prompt: bool = False,
    return_tensors: str | None = None,
):
    messages = [{"role": "user", "content": user_content}]
    if assistant_content is not None:
        messages.append({"role": "assistant", "content": assistant_content})
    return tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        return_tensors=return_tensors,
        return_dict=False,
    )


def vllm_tokens_prompt(tokenizer, user_prompt: str) -> TokensPrompt:
    prompt_ids = chat_prompt_ids(
        tokenizer,
        user_prompt,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    return {"prompt_token_ids": prompt_ids[0].tolist()}


def build_initial_suffix(tokenizer, adv_len: int) -> tuple[str, int]:
    token_id = tokenizer.encode("* " * 20)[-5]
    return tokenizer.decode([token_id] * adv_len), token_id


def get_nonascii_toks(tokenizer, device: torch.device) -> torch.Tensor:
    nonascii = []
    for token_id in range(tokenizer.vocab_size):
        text = tokenizer.decode([token_id])
        if not (text.isascii() and text.isprintable()):
            nonascii.append(token_id)
    nonascii.extend(tokenizer.all_special_ids)

    token = "* "
    seen = set()
    while True:
        token_id = tokenizer(token, add_special_tokens=False).input_ids[-1]
        if token_id in seen:
            break
        seen.add(token_id)
        nonascii.append(token_id)
        token += " "
    return torch.tensor(nonascii, device=device)


def generation_config_metadata(generation_config) -> dict[str, Any]:
    keys = ["do_sample", "max_new_tokens", "temperature", "top_p", "top_k", "eos_token_id", "pad_token_id"]
    return {key: getattr(generation_config, key, None) for key in keys}


def as_list(value) -> list[int] | None:
    if value is None:
        return None
    return value if isinstance(value, list) else [value]
