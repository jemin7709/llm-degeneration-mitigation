from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import yaml


def load_config() -> dict[str, Any]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/attack/multi_prompt_gcg.yaml")
    parser.add_argument("--cfg", choices=["job"])
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    with Path(args.config).open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    apply_overrides(cfg, args.overrides)
    cfg["save_dir"] = str(resolve_save_dir(cfg))
    if args.cfg == "job":
        print(yaml.safe_dump(cfg, sort_keys=False), end="")
        raise SystemExit(0)
    return cfg


def apply_overrides(cfg: dict[str, Any], overrides: list[str]) -> None:
    for override in overrides:
        key, value = override.split("=", 1)
        cfg[key] = parse_value(value)


def parse_value(value: str):
    if value in {"null", "None"}:
        return None
    if value in {"true", "false"}:
        return value == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def resolve_save_dir(cfg: dict[str, Any]) -> Path:
    if cfg.get("save_dir"):
        return Path(cfg["save_dir"])
    name = f"{cfg['model_tag']}_{cfg['data_name']}_multi_prompt_gcg_b{cfg['prompt_batch_size']}_s{cfg['seed']}"
    return Path(cfg["root_dir"]) / name


def configure_logging() -> None:
    logging.basicConfig(
        format="[%(asctime)s] %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
        level=logging.INFO,
    )
