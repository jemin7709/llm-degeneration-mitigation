from __future__ import annotations

import json
from pathlib import Path
from typing import Any

MATH_QUERY_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form Answer: $ANSWER (without quotes) where $ANSWER is the answer to the problem.

{Question}

Remember to put your answer on its own line after "Answer:", and you do not need to use a \\boxed command.
""".strip()


def read_prompts(cfg: dict[str, Any]) -> list[str]:
    data_name = cfg["data_name"]
    if data_name == "all":
        return read_all_prompts(Path(cfg["data_path"]))
    if data_name not in {"math_train", "math_test"}:
        raise ValueError("data_name must be math_train, math_test, or all")
    return read_math_prompts(Path(cfg["data_path"]))


def read_all_prompts(path: Path) -> list[str]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [row["instruction"] for row in rows]


def read_math_prompts(path: Path) -> list[str]:
    prompts = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                prompts.append(MATH_QUERY_TEMPLATE.format(Question=item["problem"]))
    return prompts


def index_range(cfg: dict[str, Any], data_len: int) -> tuple[int, int]:
    start = int(cfg["start_index"])
    if cfg.get("end_index") is not None:
        end = int(cfg["end_index"])
    elif cfg.get("limit") is not None:
        end = start + int(cfg["limit"])
    else:
        end = data_len
    return start, min(end, data_len)


def prompt_chunks(indices: list[int], batch_size: int):
    for start in range(0, len(indices), batch_size):
        chunk = indices[start : start + batch_size]
        yield chunk[0] // batch_size, chunk
