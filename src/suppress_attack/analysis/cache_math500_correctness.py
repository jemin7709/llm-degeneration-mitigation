from __future__ import annotations

import argparse
import contextlib
import io
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Protocol, cast

import pandas as pd
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.requests import Doc

OUTPUT_ROOT = Path("outputs/eval_math/seed23")
SAVE_DIR = Path("outputs/analysis/math500_correctness")

MODELS = [
    ("base-math500", "base"),
    ("dpo-naive-", "naive"),
    ("dpo-data_filtering_rep4_053", "data_filtering_rep4_053"),
    ("dpo-data_filtering-", "data_filtering"),
    ("dpo-math_solution-", "math_solution"),
]
PROMPTS = [("baseline", "baseline"), ("attack", "attack")]


def match_label(name: str, patterns: list[tuple[str, str]]) -> str | None:
    return next((value for key, value in patterns if key in name), None)


class SampleLevelFn(Protocol):
    def preprocess(self, text: str) -> str: ...
    def compute_score(self, *, doc: Any, model_response: Any) -> Any: ...


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    if hasattr(value, "tolist"):
        converted = value.tolist()
        return list(converted) if isinstance(converted, (list, tuple)) else [converted]
    return [value]


def model_texts(model_response: dict[str, Any], key: str, num_samples: int) -> list[str]:
    texts = ["" if text is None else str(text) for text in as_list(model_response.get(key))]
    return (texts + [""] * num_samples)[:num_samples]


def score_predictions(model_response: dict[str, Any], num_samples: int) -> list[str]:
    preds = model_texts(model_response, "text_post_processed", num_samples)
    return preds if any(preds) else model_texts(model_response, "text", num_samples)


def output_token_counts(model_response: dict[str, Any], num_samples: int) -> list[int | None]:
    counts = [
        None if tokens is None else len(tokens)
        for tokens in as_list(model_response.get("output_tokens"))[:num_samples]
    ]
    return (counts + [None] * num_samples)[:num_samples]


def infer_num_samples(detail: pd.DataFrame) -> int:
    if detail.empty:
        raise ValueError("Cannot infer num_samples from an empty detail parquet")
    model_response = cast(dict[str, Any], detail.iloc[0]["model_response"])
    for key in ("output_tokens", "text_post_processed", "text"):
        values = as_list(model_response.get(key))
        if values:
            return len(values)
    raise ValueError("Cannot infer num_samples from model_response")


def problem_id(doc: dict[str, Any], fallback: int) -> str:
    specific = doc.get("specific") or {}
    return str(doc.get("id") or specific.get("unique_id") or fallback)


def build_score_one(num_samples: int) -> Callable[[dict[str, Any], str], float]:
    metric = Metrics.pass_at_k_math(sample_params={"k": 1, "n": num_samples})
    pass_fn = cast(SampleLevelFn, metric.sample_level_fn)

    def score_one(doc_dict: dict[str, Any], pred: str) -> float:
        choices = [pass_fn.preprocess(str(choice)) for choice in as_list(doc_dict["choices"])]
        processed_pred = pass_fn.preprocess(pred)
        doc = Doc(
            task_name=str(doc_dict.get("task_name", "")),
            query=doc_dict.get("query", ""),
            choices=choices,
            gold_index=int(doc_dict.get("gold_index", 0)),
        )
        model_response = SimpleNamespace(final_text=[processed_pred], text=[processed_pred])
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return float(pass_fn.compute_score(doc=doc, model_response=model_response))

    return score_one


def score_run(run_dir: Path, model: str, prompt: str) -> pd.DataFrame:
    detail_path = sorted((run_dir / "details").glob("**/*.parquet"))[-1]
    detail = pd.read_parquet(detail_path)
    num_samples = infer_num_samples(detail)
    score_one = build_score_one(num_samples)
    rows = []

    for problem_idx, row in enumerate(detail.itertuples(index=False)):
        doc = cast(dict[str, Any], row.doc)
        model_response = cast(dict[str, Any], row.model_response)
        preds = score_predictions(model_response, num_samples)
        token_counts = output_token_counts(model_response, num_samples)
        for generation_idx, (pred, token_count) in enumerate(zip(preds, token_counts, strict=True)):
            rows.append(
                {
                    "model": model,
                    "prompt": prompt,
                    "run_dir": run_dir.name,
                    "detail_path": str(detail_path),
                    "problem_idx": problem_idx,
                    "problem_id": problem_id(doc, problem_idx),
                    "generation_idx": generation_idx,
                    "is_correct": score_one(doc, pred) >= 1.0,
                    "token_count": token_count,
                }
            )

    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--save-dir", type=Path, default=SAVE_DIR)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    save_dir = args.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    frames = []

    for run_dir in sorted(path for path in args.output_root.iterdir() if path.is_dir()):
        model = match_label(run_dir.name, MODELS)
        prompt = match_label(run_dir.name, PROMPTS)
        if not model or not prompt:
            continue

        cache_path = save_dir / f"{model}_{prompt}.parquet"
        if cache_path.exists() and not args.force:
            print(f"reuse {cache_path}")
            frames.append(pd.read_parquet(cache_path))
            continue

        print(f"score {model}/{prompt}", flush=True)
        frame = score_run(run_dir, model, prompt)
        frame.to_parquet(cache_path, index=False)
        frames.append(frame)

    correctness = pd.concat(frames, ignore_index=True)
    correctness.to_parquet(save_dir / "completion_correctness.parquet", index=False)
    correctness.to_csv(save_dir / "completion_correctness.csv", index=False)


if __name__ == "__main__":
    main()
