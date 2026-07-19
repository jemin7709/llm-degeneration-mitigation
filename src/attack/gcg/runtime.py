from __future__ import annotations

import os
from typing import Any

import torch
from accelerate import Accelerator

from .model import load_model_and_tokenizer, warmup_cuda

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


class ColocatedRuntime:
    def __init__(self, cfg: dict[str, Any]):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required")

        from vllm import LLM

        self.cfg = cfg
        self.accelerator = Accelerator()
        self.device = self.accelerator.device
        self.active = "loading"

        self.model, self.tokenizer = load_model_and_tokenizer(cfg["model_name"])
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.model.generation_config.max_new_tokens = int(cfg["max_new_tokens"])

        warmup_cuda(self.device)

        self.llm = LLM(
            model=cfg["model_name"],
            trust_remote_code=True,
            generation_config="vllm",
            enable_sleep_mode=True,
            gpu_memory_utilization=float(cfg["vllm_gpu_memory_utilization"]),
            seed=int(cfg["seed"]),
        )
        self.llm.sleep(level=int(cfg["vllm_sleep_level"]))
        self.active = "sleeping"

    def activate_hf(self) -> None:
        if self.active == "hf":
            return
        if self.active == "vllm":
            torch.cuda.synchronize()
            self.llm.sleep(level=int(self.cfg["vllm_sleep_level"]))
        torch.cuda.empty_cache()
        self.model.to(self.device)
        self.active = "hf"

    def activate_vllm(self) -> None:
        if self.active == "vllm":
            return
        if self.active == "hf":
            torch.cuda.synchronize()
            self.model.to("cpu")
        torch.cuda.empty_cache()
        self.llm.wake_up()
        self.active = "vllm"
