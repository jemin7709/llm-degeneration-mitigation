from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter
from pathlib import Path


ANSWER_RE = re.compile(r"(?i)Answer\s*:\s*(.+)")
SETTINGS = [
    "naive",
    "math_solution",
    "data_filtering",
]
MATH_SCORER = {"name": "lighteval_math_extractive_match", "precision": 6}
MATH_METRIC = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--setting", choices=SETTINGS, required=True)
    parser.add_argument("--math-data", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/preference_datasets"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--validation-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=23)
    return parser.parse_args()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def result_paths(result_dir, limit):
    paths = sorted(
        result_dir.glob("res_*.json"),
        key=lambda p: int(p.stem.removeprefix("res_")),
    )
    paths = paths[:limit] if limit else paths
    if not paths:
        raise SystemExit(f"No res_*.json files in {result_dir}")
    return paths


def load_math(path):
    if path.suffix == ".csv":
        with path.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    else:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    out = {}
    for i, row in enumerate(rows):
        solution = str(row.get("solution") or row.get("Answer") or "").strip()
        answer = (
            str(row.get("answer") or row.get("final_answer") or "").strip()
            or extract_answer(solution)
        )
        out[int(row.get("index", i))] = {"solution": solution, "answer": answer}
    return out


def result_outputs(data):
    rows = []
    base = data["-1"]
    if str(base.get("baseline_answer", "")).strip():
        rows.append(make_output(
            source="baseline_answer",
            step="baseline",
            prompt=base["baseline_prompt"],
            answer=base["baseline_answer"],
            success=base.get("baseline_success", False),
        ))
    for key in sorted((key for key in data if key != "-1"), key=int):
        step = data[key]
        if str(step.get("answer", "")).strip():
            rows.append(make_output(
                source="attack_answer",
                step=int(key),
                prompt=step.get("adv_prompt") or step["prompt"],
                answer=step["answer"],
                success=step.get("success", False),
            ))
    return rows


def make_output(source, step, prompt, answer, success):
    return {
        "source": source,
        "step": step,
        "prompt": str(prompt).strip(),
        "answer": str(answer).strip(),
        "success": bool(success),
    }


def choose(setting, rows, math_row):
    if setting == "naive":
        return choose_naive(rows)
    if setting == "math_solution":
        return choose_math_solution(rows, math_row)
    return choose_data_filtering(rows, math_row)
def choose_naive(rows):
    chosen = first(rows, "baseline_answer")
    rejected = first(rows, "attack_answer")
    if chosen is None:
        return None, rejected, "missing_baseline_answer"
    if rejected is None:
        return chosen, None, "missing_attack_answer"
    return chosen, rejected, None


def choose_math_solution(rows, math_row):
    rejected = successful(rows)
    if math_row is None or rejected is None:
        reason = "missing_math" if math_row is None else "missing_success_reject"
        return None, rejected, reason
    chosen = {
        "source": "math_solution",
        "step": "gold",
        "answer": gold_solution(math_row),
    }
    return chosen, rejected, None


def choose_data_filtering(rows, math_row):
    rejected = successful_attack(rows)
    if math_row is None or rejected is None:
        reason = "missing_math" if math_row is None else "missing_attack_success_reject"
        return None, rejected, reason
    chosen = next(
        (
            row
            for row in correct_candidates(rows, rejected)
            if correct(math_row["answer"], row["answer"])
            and has_think_tags(row["answer"])
        ),
        None,
    )
    if chosen is None:
        return None, rejected, "missing_correct_non_success_with_think_tags"
    return chosen, rejected, None
def has_think_tags(answer):
    return "<think>" in answer and "</think>" in answer


def correct_candidates(rows, rejected):
    for row in rows:
        if row["success"]:
            continue
        if after_rejected_attack(row, rejected):
            continue
        yield row


def after_rejected_attack(row, rejected):
    return (
        row["source"] == "attack_answer"
        and isinstance(rejected["step"], int)
        and row["step"] >= rejected["step"]
    )


def first(rows, source):
    return next((row for row in rows if row["source"] == source), None)


def successful(rows):
    return successful_attack(rows) or next((row for row in rows if row["success"]), None)


def successful_attack(rows):
    return next(
        (row for row in rows if row["source"] == "attack_answer" and row["success"]),
        None,
    )


def gold_solution(row):
    return (
        f"<think>\n{row['solution']}\n</think>\n"
        f"Answer: \\boxed{{{row['answer']}}}"
    )


def correct(gold, prediction):
    return math_metric().compute(math_doc(gold), math_response(prediction)) == 1.0


def math_metric():
    global MATH_METRIC
    if MATH_METRIC is None:
        from lighteval.metrics.dynamic_metrics import MultilingualExtractiveMatchMetric
        from lighteval.metrics.utils.extractive_match_utils import (
            ExprExtractionConfig,
            LatexExtractionConfig,
        )
        from lighteval.utils.language import Language

        MATH_METRIC = MultilingualExtractiveMatchMetric(
            language=Language.ENGLISH,
            gold_extraction_target=[ExprExtractionConfig(), LatexExtractionConfig()],
            pred_extraction_target=[ExprExtractionConfig(), LatexExtractionConfig()],
            precision=6,
        )
    return MATH_METRIC


def math_doc(gold):
    from lighteval.tasks.requests import Doc

    return Doc(task_name="math", query="", choices=[gold], gold_index=0)


def math_response(prediction):
    from lighteval.models.model_output import ModelResponse

    return ModelResponse(text=[prediction])


def extract_answer(text):
    boxed = boxed_answer(text)
    if boxed:
        return boxed
    matches = ANSWER_RE.findall(text)
    return matches[-1].strip() if matches else str(text).strip()


def boxed_answer(text):
    start = str(text).rfind(r"\boxed{")
    if start < 0:
        return None
    depth, chars = 1, []
    for char in str(text)[start + len(r"\boxed{") :]:
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return "".join(chars).strip()
        chars.append(char)
    return None


def audit(index, path, chosen, rejected):
    return {
        "index": index,
        "path": str(path),
        "prompt": rejected["prompt"],
        "chosen": chosen["answer"],
        "chosen_source": chosen["source"],
        "chosen_step": chosen["step"],
        "rejected": rejected["answer"],
        "rejected_source": rejected["source"],
        "reject_step": rejected["step"],
    }


def build(paths, math_rows, setting):
    examples = []
    skipped = Counter()
    for path in paths:
        index = int(path.stem.removeprefix("res_"))
        rows = result_outputs(read_json(path))
        chosen, rejected, skip_reason = choose(setting, rows, math_rows.get(index))
        if skip_reason:
            skipped[skip_reason] += 1
            continue
        examples.append(audit(index, path, chosen, rejected))
    return examples, {
        "files": len(paths),
        "kept": len(examples),
        "skipped": sum(skipped.values()),
        "skipped_by_reason": dict(sorted(skipped.items())),
    }


def split(rows, ratio, seed):
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    n = max(1, round(len(rows) * ratio)) if ratio > 0 and len(rows) > 1 else 0
    return rows[n:], rows[:n]


def dpo(row):
    return {
        "prompt": [{"role": "user", "content": row["prompt"]}],
        "chosen": [{"role": "assistant", "content": row["chosen"]}],
        "rejected": [{"role": "assistant", "content": row["rejected"]}],
    }


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")


def save(args, examples, stats):
    out = args.output_root / args.result_dir.name / args.setting
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"Output dir already exists: {out}")
    train, valid = split(examples, args.validation_ratio, args.seed)
    write_jsonl(out / "train.jsonl", [dpo(row) for row in train])
    write_jsonl(out / "validation.jsonl", [dpo(row) for row in valid])
    write_jsonl(out / "audit.jsonl", audit_rows(train, valid))
    summary = summary_json(args, out, examples, train, valid, stats)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary["counts"] | {"output_dir": str(out)}))


def audit_rows(train, valid):
    return (
        [{**row, "split": "train"} for row in train]
        + [{**row, "split": "validation"} for row in valid]
    )


def summary_json(args, out, examples, train, valid, stats):
    summary = {
        "setting": args.setting,
        "result_dir": str(args.result_dir),
        "math_data": str(args.math_data),
        "output_dir": str(out),
        "counts": {
            "examples": len(examples),
            "train": len(train),
            "validation": len(valid),
        },
        "stats": stats,
    }
    if args.setting == "data_filtering":
        summary["scorer"] = MATH_SCORER
    return summary


def main():
    parsed = parse_args()
    examples, stats = build(
        result_paths(parsed.result_dir, parsed.limit),
        load_math(parsed.math_data),
        parsed.setting,
    )
    save(parsed, examples, stats)


if __name__ == "__main__":
    main()
