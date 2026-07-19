from __future__ import annotations

import argparse
import contextlib
import html
import io
import json
from collections import Counter
from itertools import groupby
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Protocol, cast

import numpy as np
import pandas as pd

ScoreOne = Callable[
    [dict[str, Any], str], tuple[float, str, str, str, list[dict[str, Any]]]
]


class SampleLevelFn(Protocol):
    def preprocess(self, text: str) -> str: ...

    def compute_score(self, *, doc: Any, model_response: Any) -> Any: ...


QUERY_PREFIX = (
    "Solve the following math problem step by step. The last line of your response should be of the form "
    "Answer: $ANSWER (without quotes) where $ANSWER is the answer to the problem."
)
REMEMBER_PREFIX = (
    'Remember to put your answer on its own line after "Answer:", '
    "and you do not need to use a \\boxed command."
)
MAX_REGEX_CANDIDATES = 80


def as_list(x: Any) -> list[Any]:
    """Convert parquet/numpy/list/scalar values to a Python list."""
    if x is None:
        return []
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def esc(x: Any) -> str:
    return html.escape("" if x is None else str(x), quote=False)


def esc_attr(x: Any) -> str:
    return html.escape("" if x is None else str(x), quote=True)


def raw_text(x: Any) -> str:
    text = "" if x is None else str(x)
    return "\n".join(
        json.dumps(line, ensure_ascii=False)[1:-1] for line in text.split("\n")
    )


def esc_raw(x: Any) -> str:
    return esc(raw_text(x))


def esc_raw_attr(x: Any) -> str:
    return esc_attr(raw_text(x))


def answer_span(match: Any, fallback: str) -> tuple[int, int] | None:
    if not fallback:
        return None

    group_spans: list[tuple[int, int, str, str]] = []
    for name, value in match.groupdict().items():
        if not value:
            continue
        start = match.start(name)
        end = match.end(name)
        if start == -1 or end == -1:
            continue
        group_spans.append((start, end, name, value))

    answer_groups = [
        span
        for span in group_spans
        if any(part in span[2].lower() for part in ("latex", "expr", "integer", "decimal"))
    ]
    for start, end, name, value in sorted(
        answer_groups, key=lambda span: span[1], reverse=True
    ):
        if "latex" in name.lower():
            return start, end
        offset = value.rfind(fallback)
        if offset != -1:
            return start + offset, start + offset + len(fallback)
        if fallback in raw_text(value):
            return start, end
    if answer_groups:
        start, end, _, _ = max(answer_groups, key=lambda span: span[1])
        return start, end

    match_text = match.group(0)
    offset = match_text.rfind(fallback)
    if offset != -1:
        return match.start() + offset, match.start() + offset + len(fallback)
    for start, end, _, value in sorted(
        group_spans, key=lambda span: span[1], reverse=True
    ):
        if fallback in raw_text(value):
            return start, end
    return None


def collect_lighteval_candidates(
    pred: str,
    pred_extraction_regexes: Any,
    extract_match_fn: Any,
    extraction_mode: str,
    timeout_seconds: int,
) -> list[dict[str, Any]]:
    all_patterns = [
        (pattern, target_type, priority)
        for target_patterns, target_type in pred_extraction_regexes
        for pattern, priority in target_patterns
    ]
    fallback_candidates: list[dict[str, Any]] = []
    for _, patterns_group in groupby(
        sorted(all_patterns, key=lambda x: x[2]), key=lambda x: x[2]
    ):
        matches_with_pos = [
            (match, match.start(), match.end(), target_type)
            for pattern, target_type, _ in patterns_group
            for match in pattern.finditer(pred)
        ]
        matches_with_pos = sorted(
            matches_with_pos, key=lambda x: (x[2], -x[1]), reverse=True
        )
        if not matches_with_pos:
            continue

        display_matches = matches_with_pos[:MAX_REGEX_CANDIDATES]
        candidates = [
            {
                "start": start,
                "end": end,
                "answer_start": None,
                "answer_end": None,
                "fallback": "",
            }
            for _, start, end, _ in display_matches
        ]
        fallback_candidates = candidates

        for idx, (match, _, _, target_type) in enumerate(matches_with_pos):
            extracted_match, str_fallback = extract_match_fn(
                match, target_type, timeout_seconds
            )
            if idx < len(candidates):
                candidates[idx]["fallback"] = str_fallback
            if extracted_match is not None:
                span = answer_span(match, str_fallback)
                if idx >= len(candidates):
                    candidates.append(
                        {
                            "start": match.start(),
                            "end": match.end(),
                            "answer_start": None,
                            "answer_end": None,
                            "fallback": str_fallback,
                        }
                    )
                    idx = len(candidates) - 1
                if span is not None:
                    candidates[idx]["answer_start"] = span[0]
                    candidates[idx]["answer_end"] = span[1]
                return candidates
            if extraction_mode == "first_match":
                return candidates

    return fallback_candidates


def merge_spans(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    spans = sorted(candidates, key=lambda item: (item["start"], item["end"]))
    merged: list[dict[str, Any]] = []
    for span in spans:
        if not merged or span["start"] > merged[-1]["end"]:
            merged.append(dict(span))
            continue
        prev = merged[-1]
        prev["end"] = max(prev["end"], span["end"])
        if span.get("answer_start") is not None:
            prev["answer_start"] = span["answer_start"]
            prev["answer_end"] = span["answer_end"]
            prev["fallback"] = span.get("fallback", "")
    return merged


def render_candidate(text: str, candidate: dict[str, Any]) -> str:
    answer_start = candidate.get("answer_start")
    answer_end = candidate.get("answer_end")
    if answer_start is None or answer_end is None:
        return esc_raw(text[candidate["start"] : candidate["end"]])

    return (
        esc_raw(text[candidate["start"] : answer_start])
        + f'<span class="actual-extraction">{esc_raw(text[answer_start:answer_end])}</span>'
        + esc_raw(text[answer_end : candidate["end"]])
    )


def highlight_extracted_answer(text: str, candidates: list[dict[str, Any]]) -> str:
    candidates = merge_spans(candidates)
    parts: list[str] = []
    cursor = 0
    for candidate in candidates:
        start = candidate["start"]
        end = candidate["end"]
        if start < cursor:
            continue
        parts.append(esc_raw(text[cursor:start]))
        title = candidate.get("fallback") or text[start:end]
        parts.append(
            f'<span class="answer-candidate-line" title="LightEval regex match: {esc_raw_attr(title)}">'
            f"{render_candidate(text, candidate)}</span>"
        )
        cursor = end
    parts.append(esc_raw(text[cursor:]))
    return "".join(parts)


def fmt_extracted(values: Any) -> str:
    vals = as_list(values)
    out: list[str] = []
    for v in vals:
        s = str(v).strip()
        if s and s not in out:
            out.append(s)
    return " | ".join(out) if out else "없음"


def metric_value(metric_dict: dict[str, Any], k: int, n: int) -> float | None:
    key = f"pass@k:k={k}&n={n}"
    if key in metric_dict:
        return float(metric_dict[key])
    # LightEval version/serialization differences에 대한 약한 fallback
    for kk, vv in metric_dict.items():
        if f"k={k}" in kk and f"n={n}" in kk:
            return float(vv)
    return None


def parse_math500_query(query: str) -> tuple[str, str]:
    """Return the problem text and suffix appended after the fixed MATH instruction."""
    q = (query or "").strip()
    problem, suffix = q.rsplit(REMEMBER_PREFIX, 1)
    problem = problem.strip()
    if problem.startswith(QUERY_PREFIX):
        problem = problem[len(QUERY_PREFIX) :].strip()
    return problem, suffix.strip()


def model_texts(
    model_response: dict[str, Any], key: str, num_samples: int
) -> list[str]:
    texts = as_list(model_response.get(key))
    texts = ["" if text is None else str(text) for text in texts]
    if len(texts) < num_samples:
        texts.extend([""] * (num_samples - len(texts)))
    return texts[:num_samples]


def get_score_predictions(
    model_response: dict[str, Any], num_samples: int
) -> list[str]:
    preds = model_texts(model_response, "text_post_processed", num_samples)
    if not any(preds):
        preds = model_texts(model_response, "text", num_samples)
    return preds


def get_display_predictions(
    model_response: dict[str, Any], num_samples: int
) -> list[str]:
    return model_texts(model_response, "text", num_samples)


def get_output_token_counts(
    model_response: dict[str, Any], num_samples: int
) -> list[int | None]:
    token_rows = as_list(model_response.get("output_tokens"))
    counts = [None if row is None else len(row) for row in token_rows[:num_samples]]
    if len(counts) < num_samples:
        counts.extend([None] * (num_samples - len(counts)))
    return counts


def infer_num_samples(df: pd.DataFrame) -> int:
    if df.empty:
        raise ValueError("Cannot infer num_samples from an empty parquet")
    model_response = dict(df.iloc[0]["model_response"])
    for key in ("output_tokens", "text_post_processed", "text"):
        values = as_list(model_response.get(key))
        if values:
            return len(values)
    raise ValueError("Cannot infer num_samples from model_response")


def build_lighteval_single_scorer(
    num_samples: int,
    show_warnings: bool = False,
) -> ScoreOne:
    """
    Build a per-response scorer that mirrors Metrics.pass_at_k_math.

    It uses the same PassAtK sample_scoring_function, but calls it on one
    prediction at a time so the HTML can mark R1/R2/R3/R4 as correct/incorrect.
    """
    try:
        from lighteval.metrics.metrics import Metrics
        from lighteval.metrics.utils.extractive_match_utils import (
            extract_match,
            get_extraction_regexes,
        )
        from lighteval.tasks.requests import Doc
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "lighteval가 필요합니다. 예: pip install 'lighteval[math]'"
        ) from e

    metric = Metrics.pass_at_k_math(sample_params={"k": 1, "n": num_samples})
    pass_fn_obj = getattr(metric, "sample_level_fn", None)
    if pass_fn_obj is None or not hasattr(pass_fn_obj, "compute_score"):
        raise RuntimeError(
            "현재 lighteval 버전에서 Metrics.pass_at_k_math(...).sample_level_fn.compute_score를 찾지 못했습니다."
    )
    pass_fn = cast(SampleLevelFn, pass_fn_obj)
    extraction_metric = cast(Any, getattr(pass_fn.compute_score, "__self__", None))
    if extraction_metric is None:
        raise RuntimeError("LightEval 추출 metric 객체를 찾지 못했습니다.")

    def score_one(
        doc_dict: dict[str, Any], pred: str
    ) -> tuple[float, str, str, str, list[dict[str, Any]]]:
        choices = [str(x) for x in as_list(doc_dict.get("choices"))]
        gold_index = int(doc_dict.get("gold_index", 0))

        # PassAtK.compute 내부의 preprocess 경로를 그대로 반영합니다.
        processed_choices = [pass_fn.preprocess(c) for c in choices]
        processed_pred = pass_fn.preprocess(pred)

        doc = Doc(
            task_name=str(doc_dict.get("task_name", "")),
            query=doc_dict.get("query", ""),
            choices=processed_choices,
            gold_index=gold_index,
        )
        pred_extraction_regexes = get_extraction_regexes(
            doc,
            extraction_metric.pred_extraction_target,
            extraction_metric.language,
        )
        extraction_candidates = collect_lighteval_candidates(
            processed_pred,
            pred_extraction_regexes,
            extract_match,
            extraction_metric.extraction_mode,
            extraction_metric.timeout_seconds,
        )
        model_response = SimpleNamespace(
            final_text=[processed_pred], text=[processed_pred]
        )

        if show_warnings:
            score = float(pass_fn.compute_score(doc=doc, model_response=model_response))
        else:
            with (
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                score = float(
                    pass_fn.compute_score(doc=doc, model_response=model_response)
                )
        specific = getattr(doc, "specific", None) or {}
        extracted_pred = fmt_extracted(specific.get("extracted_predictions"))
        extracted_gold = fmt_extracted(specific.get("extracted_golds"))
        return score, extracted_pred, extracted_gold, processed_pred, extraction_candidates

    return score_one


def response_html(i: int, resp: dict[str, Any]) -> str:
    ok = bool(resp["correct"])
    cls = "correct" if ok else "incorrect"
    label = "정답" if ok else "오답"
    mark = "✅" if ok else "❌"
    text = resp["text"]
    highlight_text = str(resp.get("highlight_text", text))
    token_count = resp.get("output_token_count")
    token_label = (
        f"{token_count:,} tokens" if token_count is not None else "tokens: 없음"
    )
    return f"""
        <details class="response {cls}">
          <summary>
            <span class="answer-badge {cls}">{mark} {label}</span>
            응답 {i} · 추출 답: <code>{esc_raw(resp["extracted_pred"])}</code> · {token_label}
          </summary>
          <h4>추출 하이라이트</h4>
          <pre class="highlighted-output">{highlight_extracted_answer(highlight_text, resp["extraction_candidates"])}</pre>
          <h4>Raw 출력</h4>
          <pre class="raw-output">{esc_raw(text)}</pre>
        </details>"""


def item_html(item: dict[str, Any]) -> str:
    status = item["status"]
    badge_text = {
        "all": f"{item['correct_count']}/{item['num_samples']} 정답",
        "partial": f"{item['correct_count']}/{item['num_samples']} 부분 정답",
        "fail": f"0/{item['num_samples']} 실패",
    }[status]

    answer_summary = " | ".join(
        f'<span class="mini {"correct" if r["correct"] else "incorrect"}">'
        f"R{j}: {'✅' if r['correct'] else '❌'} <code>{esc_raw(r['extracted_pred'])}</code>"
        f"</span>"
        for j, r in enumerate(item["responses"], start=1)
    )
    responses = "\n".join(
        response_html(i, r) for i, r in enumerate(item["responses"], start=1)
    )
    corrects = ",".join("1" if r["correct"] else "0" for r in item["responses"])
    search_blob = " ".join(
        [
            str(item["row_no"]),
            str(item.get("id", "")),
            item.get("task_name", ""),
            raw_text(item["problem"]),
            raw_text(item["suffix"]),
            raw_text(item["gold_solution"]),
            raw_text(item["gold_extracted"]),
            " ".join(raw_text(r["extracted_pred"]) for r in item["responses"]),
        ]
    ).lower()

    return f"""
    <details class="item {status}" data-status="{status}" data-corrects="{esc_attr(corrects)}" data-search="{esc_attr(search_blob)}">
      <summary>
        <span class="badge {status}">{badge_text}</span>
        <span class="mono">#{item["row_no"]:03d} · id={esc(item.get("id", ""))}</span>
        <span class="metric">correct={item["correct_count"]}/{item["num_samples"]} · pass@1={item["pass1"]:.2f} · pass@{item["num_samples"]}={item["passk"]:.0f}</span>
        <span class="preview">{esc_raw(item["problem"])}</span>
      </summary>
      <section class="grid">
        <div><h3>문제</h3><pre>{esc_raw(item["problem"])}</pre></div>
        <div><h3>공격 suffix</h3><pre>{esc_raw(item["suffix"] or "(없음)")}</pre></div>
      </section>

      <h3>정답 / Gold</h3>
      <p><b>Gold answer:</b> <code>{esc_raw(item["gold_answer"])}</code></p>
      <p><b>LightEval 채점용 gold answer:</b> <code>{esc_raw(item["gold_extracted"])}</code></p>
      <pre>{esc_raw(item["gold_solution"])}</pre>

      <h3>모델 응답 요약</h3>
      <p class="answers">{answer_summary}</p>

      {responses}
    </details>"""


def render_html(
    items: list[dict[str, Any]],
    source_name: str,
    num_samples: int,
    eval_errors: int,
    mismatches: list[str],
) -> str:
    n_items = len(items)
    dist = Counter(item["correct_count"] for item in items)
    mean_pass1 = sum(item["pass1"] for item in items) / max(n_items, 1)
    passk_success = sum(1 for item in items if item["passk"] >= 1.0)
    full_fail = dist.get(0, 0)
    max_len = max(
        (
            r["output_token_count"]
            for item in items
            for r in item["responses"]
            if r["output_token_count"] is not None
        ),
        default=0,
    )

    dist_rows = []
    for c in range(num_samples, -1, -1):
        cnt = dist.get(c, 0)
        pct = 100.0 * cnt / max(n_items, 1)
        dist_rows.append(
            f"<tr><td>{c}/{num_samples}</td><td>{cnt}</td>"
            f'<td><div class="bar"><span style="width:{pct:.1f}%"></span></div> {pct:.1f}%</td></tr>'
        )

    mismatch_note = ""
    if mismatches:
        shown = "\n".join(mismatches[:20])
        more = f"\n... and {len(mismatches) - 20} more" if len(mismatches) > 20 else ""
        mismatch_note = f"""
        <details class="warning" open>
          <summary>주의: parquet 저장 metric과 재평가 결과가 다른 row {len(mismatches)}개</summary>
          <pre>{esc(shown + more)}</pre>
        </details>"""

    options = [
        '<option value="">응답 정오답 필터 없음</option>',
        '<option value="has_wrong">오답 응답 포함</option>',
        '<option value="has_correct">정답 응답 포함</option>',
    ]
    for i in range(1, num_samples + 1):
        options.append(f'<option value="r{i}_correct">R{i} 정답</option>')
        options.append(f'<option value="r{i}_wrong">R{i} 오답</option>')

    body = "\n".join(item_html(item) for item in items)

    return f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>MATH500 Attack Result Viewer</title>
<style>
  :root {{ --border:#d0d7de; --muted:#57606a; --bg:#f6f8fa; --ok:#1a7f37; --warn:#9a6700; --bad:#cf222e; }}
  body {{ font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; margin:24px; color:#24292f; }}
  h1 {{ margin-bottom:4px; }}
  h3 {{ margin-bottom:6px; }}
  .sub {{ color:var(--muted); margin-top:0; }}
  .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:12px; margin:18px 0; }}
  .card {{ border:1px solid var(--border); border-radius:10px; padding:14px; background:white; }}
  .card .label {{ color:var(--muted); font-size:13px; }}
  .card .value {{ font-size:24px; font-weight:700; margin-top:4px; }}
  .toolbar {{ position:sticky; top:0; z-index:2; background:white; padding:12px 0; border-bottom:1px solid var(--border); display:flex; gap:8px; align-items:center; flex-wrap:wrap; }}
  input, select, button {{ padding:8px 10px; border:1px solid var(--border); border-radius:8px; font-size:14px; background:white; }}
  input {{ min-width:min(520px,80vw); flex:1; }}
  button {{ cursor:pointer; }}
  details.item {{ border:1px solid var(--border); border-radius:10px; margin:10px 0; background:white; }}
  details.item > summary {{ cursor:pointer; padding:12px 14px; display:flex; gap:10px; align-items:center; flex-wrap:wrap; }}
  details.item[open] > summary {{ border-bottom:1px solid var(--border); background:var(--bg); }}
  .badge {{ border-radius:999px; padding:3px 8px; font-size:12px; font-weight:700; color:white; }}
  .badge.all {{ background:var(--ok); }} .badge.partial {{ background:var(--warn); }} .badge.fail {{ background:var(--bad); }}
  .answer-badge {{ border-radius:999px; padding:2px 7px; font-size:12px; font-weight:700; color:white; margin-right:6px; }}
  .answer-badge.correct {{ background:var(--ok); }} .answer-badge.incorrect {{ background:var(--bad); }}
  .mini {{ display:inline-block; border:1px solid var(--border); border-radius:8px; padding:6px 8px; margin:3px 4px 3px 0; }}
  .mini.correct {{ border-color:#8cce9f; background:#dafbe1; }}
  .mini.incorrect {{ border-color:#ffb3b8; background:#ffebe9; }}
  .mono {{ font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; }}
  .metric {{ color:var(--muted); }}
  .preview {{ flex-basis:100%; margin-left:2px; }}
  section.grid {{ display:grid; grid-template-columns:minmax(0,1fr) minmax(0,1fr); gap:12px; padding:14px; }}
  details.item > h3, details.item > p, details.item > pre, details.item > details, details.item > section + h3 {{ margin-left:14px; margin-right:14px; }}
  pre {{ background:#f6f8fa; border:1px solid var(--border); border-radius:8px; padding:12px; overflow:auto; white-space:pre-wrap; line-height:1.45; font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; font-size:13px; }}
  .response {{ margin:10px 14px; border:1px solid var(--border); border-radius:8px; }}
  .response.correct {{ border-left:5px solid var(--ok); }}
  .response.incorrect {{ border-left:5px solid var(--bad); }}
  .response > summary {{ padding:10px; cursor:pointer; background:#fbfbfb; }}
  .response > h4 {{ margin:10px 10px 6px; font-size:13px; color:var(--muted); }}
  .response > pre {{ margin:0; border:none; border-top:1px solid var(--border); border-radius:0; }}
  .response > pre:last-child {{ border-radius:0 0 8px 8px; }}
  .raw-output {{ background:#fff; }}
  table {{ border-collapse:collapse; margin:12px 0; min-width:360px; }}
  th, td {{ border:1px solid var(--border); padding:8px 10px; text-align:left; }}
  th {{ background:var(--bg); }}
  .bar {{ display:inline-block; width:160px; height:10px; background:#eaeef2; border-radius:999px; overflow:hidden; vertical-align:middle; margin-right:8px; }}
  .bar span {{ display:block; height:100%; background:#6e7781; }}
  code {{ background:#f6f8fa; padding:1px 4px; border-radius:4px; }}
  .answer-candidate-line {{ display:block; background:#fff3b0; border-left:4px solid #d29922; padding-left:8px; margin:2px 0; }}
  .actual-extraction {{ background:#fd7e14; color:white; font-weight:700; padding:0 2px; border-radius:3px; }}
  .warning {{ border:1px solid var(--warn); border-radius:8px; padding:10px 14px; margin:12px 0; background:#fff8c5; }}
  @media (max-width:900px) {{ section.grid {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body>
<h1>MATH500 Attack Result Viewer</h1>
<p class="sub">원본: <code>{esc(source_name)}</code> · LightEval <code>Metrics.pass_at_k_math</code> 방식으로 응답별 재평가</p>
<div class="cards">
  <div class="card"><div class="label">문제 수</div><div class="value">{n_items}</div></div>
  <div class="card"><div class="label">샘플 수</div><div class="value">{num_samples} / 문제</div></div>
  <div class="card"><div class="label">평균 pass@1</div><div class="value">{mean_pass1 * 100:.1f}%</div></div>
  <div class="card"><div class="label">pass@{num_samples} 성공</div><div class="value">{passk_success}/{n_items}</div></div>
  <div class="card"><div class="label">전체 실패</div><div class="value">{full_fail}</div></div>
  <div class="card"><div class="label">최대 응답 토큰 수</div><div class="value">{max_len:,}</div></div>
  <div class="card"><div class="label">평가 오류</div><div class="value">{eval_errors}</div></div>
</div>
{mismatch_note}
<h2>정답 샘플 수 분포</h2>
<table><thead><tr><th>정답 샘플 수</th><th>문제 수</th><th>비율</th></tr></thead><tbody>{"".join(dist_rows)}</tbody></table>
<div class="toolbar">
  <input id="q" placeholder="id, 문제, attack suffix, raw 추출 답으로 검색" oninput="applyFilter()">
  <select id="status" onchange="applyFilter()">
    <option value="">전체 문제</option>
    <option value="fail">0/{num_samples} 실패</option>
    <option value="partial">부분 정답</option>
    <option value="all">{num_samples}/{num_samples} 정답</option>
  </select>
  <select id="respfilter" onchange="applyFilter()">
    {"".join(options)}
  </select>
  <button onclick="expandVisible()">보이는 문제 펼치기</button>
  <button onclick="expandWrongResponses()">오답 응답만 펼치기</button>
  <button onclick="collapseAll()">모두 접기</button>
  <span><b id="shown">{n_items}</b> / {n_items} 표시</span>
</div>
<div id="items">
{body}
</div>
<script>
function responseFilterOK(corrects, filt) {{
  if (!filt) return true;
  const arr = corrects.split(',');
  if (filt === 'has_wrong') return arr.includes('0');
  if (filt === 'has_correct') return arr.includes('1');
  const m = filt.match(/^r(\\d+)_(correct|wrong)$/);
  if (m) {{
    const idx = parseInt(m[1], 10) - 1;
    const want = m[2] === 'correct' ? '1' : '0';
    return arr[idx] === want;
  }}
  return true;
}}
function applyFilter() {{
  const q = document.getElementById('q').value.toLowerCase();
  const status = document.getElementById('status').value;
  const respfilter = document.getElementById('respfilter').value;
  let shown = 0;
  document.querySelectorAll('details.item').forEach(el => {{
    let ok = true;
    if (q && !el.dataset.search.includes(q)) ok = false;
    if (status && el.dataset.status !== status) ok = false;
    if (!responseFilterOK(el.dataset.corrects, respfilter)) ok = false;
    el.hidden = !ok;
    if (ok) shown += 1;
  }});
  document.getElementById('shown').textContent = shown;
}}
function expandVisible() {{
  document.querySelectorAll('details.item:not([hidden])').forEach(el => el.open = true);
}}
function expandWrongResponses() {{
  document.querySelectorAll('details.item:not([hidden])').forEach(item => {{
    item.open = true;
    item.querySelectorAll('details.response').forEach(resp => {{
      resp.open = resp.classList.contains('incorrect');
    }});
  }});
}}
function collapseAll() {{
  document.querySelectorAll('details').forEach(el => el.open = false);
}}
</script>
</body>
</html>"""


def response_status(correct_count: int, num_samples: int) -> str:
    if correct_count == num_samples:
        return "all"
    if correct_count == 0:
        return "fail"
    return "partial"


def build_response(
    doc_dict: dict[str, Any],
    pred: str,
    output_token_count: int | None,
    score_one: ScoreOne,
) -> tuple[dict[str, Any], str | None, bool]:
    try:
        (
            score,
            extracted_pred,
            extracted_gold,
            processed_pred,
            extraction_candidates,
        ) = score_one(doc_dict, pred)
        correct = score >= 1.0
        eval_error = False
    except Exception as e:
        score = 0.0
        correct = False
        extracted_pred = f"EVAL ERROR: {type(e).__name__}: {e}"
        extracted_gold = "없음"
        processed_pred = pred
        extraction_candidates = []
        eval_error = True

    gold_candidate = extracted_gold if extracted_gold != "없음" else None
    return (
        {
            "text": pred,
            "highlight_text": processed_pred,
            "score": score,
            "correct": correct,
            "extracted_pred": extracted_pred,
            "extracted_gold": extracted_gold,
            "extraction_candidates": extraction_candidates,
            "output_token_count": output_token_count,
        },
        gold_candidate,
        eval_error,
    )


def metric_mismatches(
    row_no: int,
    doc_id: Any,
    metric_dict: dict[str, Any],
    pass1: float,
    passk: float,
    num_samples: int,
) -> list[str]:
    mismatches: list[str] = []
    reported_pass1 = metric_value(metric_dict, 1, num_samples)
    reported_passk = metric_value(metric_dict, num_samples, num_samples)
    if reported_pass1 is not None and abs(reported_pass1 - pass1) > 1e-6:
        mismatches.append(
            f"row #{row_no} id={doc_id}: reported pass@1={reported_pass1}, recomputed={pass1}"
        )
    if reported_passk is not None and abs(reported_passk - passk) > 1e-6:
        mismatches.append(
            f"row #{row_no} id={doc_id}: reported pass@{num_samples}={reported_passk}, recomputed={passk}"
        )
    return mismatches


def build_item(
    row_no: int,
    row: Any,
    num_samples: int,
    score_one: ScoreOne,
) -> tuple[dict[str, Any], int, list[str]]:
    doc_dict = dict(row["doc"])
    metric_dict = dict(row["metric"] or {})
    model_response = dict(row["model_response"])

    problem, suffix = parse_math500_query(str(doc_dict.get("query", "")))
    choices = [str(x) for x in as_list(doc_dict.get("choices"))]
    gold_index = int(doc_dict.get("gold_index", 0))
    gold_answer = choices[gold_index] if 0 <= gold_index < len(choices) else ""
    specific = dict(doc_dict.get("specific") or {})
    gold_solution = str(specific.get("solution") or gold_answer)

    score_preds = get_score_predictions(model_response, num_samples)
    display_preds = get_display_predictions(model_response, num_samples)
    output_token_counts = get_output_token_counts(model_response, num_samples)
    responses: list[dict[str, Any]] = []
    gold_extracted_candidates: list[str] = []
    eval_errors = 0

    for score_pred, display_pred, output_token_count in zip(
        score_preds, display_preds, output_token_counts, strict=True
    ):
        response, gold_candidate, eval_error = build_response(
            doc_dict,
            score_pred,
            output_token_count,
            score_one,
        )
        response["text"] = display_pred
        responses.append(response)
        if gold_candidate is not None:
            gold_extracted_candidates.append(gold_candidate)
        if eval_error:
            eval_errors += 1

    correct_count = sum(1 for r in responses if r["correct"])
    pass1 = correct_count / num_samples
    passk = 1.0 if correct_count > 0 else 0.0
    doc_id = doc_dict.get("id", "")
    return (
        {
            "row_no": row_no,
            "id": doc_id,
            "task_name": doc_dict.get("task_name", ""),
            "num_samples": num_samples,
            "problem": problem,
            "suffix": suffix,
            "gold_answer": gold_answer,
            "gold_solution": gold_solution,
            "gold_extracted": gold_extracted_candidates[0]
            if gold_extracted_candidates
            else "없음",
            "responses": responses,
            "correct_count": correct_count,
            "pass1": pass1,
            "passk": passk,
            "status": response_status(correct_count, num_samples),
        },
        eval_errors,
        metric_mismatches(row_no, doc_id, metric_dict, pass1, passk, num_samples),
    )


def build_items(
    df: pd.DataFrame,
    num_samples: int,
    score_one: ScoreOne,
) -> tuple[list[dict[str, Any]], int, list[str]]:
    items: list[dict[str, Any]] = []
    eval_errors = 0
    mismatches: list[str] = []
    sorted_rows = sorted(
        df.iterrows(),
        key=lambda indexed_row: int(dict(indexed_row[1]["doc"])["id"]),
    )
    for row_no, (_, row) in enumerate(sorted_rows, start=1):
        item, row_eval_errors, row_mismatches = build_item(
            row_no, row, num_samples, score_one
        )
        items.append(item)
        eval_errors += row_eval_errors
        mismatches.extend(row_mismatches)
    return items, eval_errors, mismatches


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parquet", required=True, help="LightEval details_*.parquet path"
    )
    parser.add_argument("--output", default=None, help="Output HTML path")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Override sample count inferred from the parquet model_response",
    )
    parser.add_argument(
        "--show-lighteval-warnings",
        action="store_true",
        help="Print LightEval extraction diagnostics while rebuilding response labels",
    )
    args = parser.parse_args()

    parquet_path = Path(args.parquet)
    output_path = (
        Path(args.output)
        if args.output
        else parquet_path.with_suffix(".lighteval_view.html")
    )
    df = pd.read_parquet(parquet_path)
    num_samples = args.num_samples or infer_num_samples(df)
    score_one = build_lighteval_single_scorer(
        num_samples, show_warnings=args.show_lighteval_warnings
    )
    items, eval_errors, mismatches = build_items(df, num_samples, score_one)

    output_path.write_text(
        render_html(
            items=items,
            source_name=parquet_path.name,
            num_samples=num_samples,
            eval_errors=eval_errors,
            mismatches=mismatches,
        ),
        encoding="utf-8",
    )
    print(f"wrote: {output_path}")
    print(
        f"items={len(items)}, eval_errors={eval_errors}, metric_mismatches={len(mismatches)}"
    )


if __name__ == "__main__":
    main()
