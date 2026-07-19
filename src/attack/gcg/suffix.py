from __future__ import annotations

import torch

from .model import chat_prompt_ids


class SuffixManager:
    def __init__(self, tokenizer, instruction: str, adv_len: int):
        self.tokenizer = tokenizer
        self.instruction = instruction
        self.target: str | None = None
        self.adv_len = adv_len
        self.adv_token_id = tokenizer.encode("* " * 20)[-5]
        self.adv_suffix = tokenizer.decode([self.adv_token_id] * adv_len)
        self._init_slices()

    def _init_slices(self) -> None:
        adv_prompt = f"{self.instruction} {self.adv_suffix.strip()}"
        prompt_ids = chat_prompt_ids(
            self.tokenizer,
            adv_prompt,
            add_generation_prompt=True,
        )
        prefix_len = (
            max(i for i, token_id in enumerate(prompt_ids) if token_id == self.adv_token_id)
            - self.adv_len
            + 1
        )
        self._control_slice = slice(prefix_len, prefix_len + self.adv_len)
        self._target_slice = slice(len(prompt_ids), None)

    def get_input_ids(self) -> torch.Tensor:
        adv_prompt = f"{self.instruction} {self.adv_suffix.strip()}"
        return chat_prompt_ids(
            self.tokenizer,
            adv_prompt,
            assistant_content=self.target,
            return_tensors="pt",
        )[0]

    def update(self, adv_suffix: str | None = None, answer: str | None = None) -> None:
        if adv_suffix is not None:
            self.adv_suffix = adv_suffix
        if answer is not None:
            ids = self.tokenizer.encode(answer, add_special_tokens=False)
            self.target = self.tokenizer.decode(ids[:1024]) if len(ids) > 1024 else answer
