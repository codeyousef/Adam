#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.losses.alignment import ignorance_penalty, paired_alignment_loss
from src.losses.sigreg import sigreg_loss
from src.models.jepa import JEPAModel
from src.training.phase4 import _proxy_config
from src.utils.data import SimpleTokenizer, make_text_code_pairs, sample_ood_queries, set_seed
from src.utils.retrieval import VectorIndex


RESULTS_TSV = ROOT / "artifacts" / "results.tsv"
RUNS_DIR = ROOT / "artifacts" / "runs"

KNOWN_QUERIES = [
    "Sort a numeric list ascending and return the result.",
    "How can I order an array of integers from smallest to largest?",
    "Read each line from a text file and strip whitespace.",
    "I want to load a file and trim every line in it.",
    "Parse a json string into a javascript object.",
    "Convert this JSON text into a JS variable.",
    "Fetch JSON from an HTTP endpoint in python.",
    "Group dictionaries by a key in python.",
]

OOD_QUERIES = [
    "What is the weather in Tokyo today?",
    "Who was the first president of the United States?",
    "The quick brown fox jumps over the lazy dog.",
    "Name three planets in the solar system.",
    "Explain photosynthesis in one sentence.",
    "Why is the sky blue?",
]


@dataclass
class TargetedExperiment:
    name: str
    seed: int
    recipe: dict[str, Any]


class QueueBuffer:
    def __init__(self, size: int, dim: int, device: str):
        self.buffer = torch.empty(0, dim, device=device)
        self.size = size

    def push(self, x: torch.Tensor) -> None:
        self.buffer = torch.cat([x.detach(), self.buffer], dim=0)[: self.size]

    def get(self) -> torch.Tensor:
        return self.buffer


def ensure_results_header() -> None:
    RESULTS_TSV.parent.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    if RESULTS_TSV.exists():
        return
    RESULTS_TSV.write_text(
        "run_id\tstatus\tdevice\tphase_score\tphase1_pass\tphase2_pass\tphase3_pass\tphase4_pass\t"
        "without_retrieval\twith_retrieval\tretrieval_gap\tplanning_success\tscaling_improvement\tdescription\n"
    )


def parse_results_table() -> list[dict[str, str]]:
    if not RESULTS_TSV.exists():
        return []
    rows: list[dict[str, str]] = []
    with RESULTS_TSV.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            cleaned = {(key or "").strip(): (value or "").strip() for key, value in row.items()}
            if cleaned.get("run_id"):
                rows.append(cleaned)
    return rows


def next_run_index(history_rows: list[dict[str, str]]) -> int:
    max_run = 0
    for row in history_rows:
        prefix = row.get("run_id", "").split("-", 1)[0]
        if prefix.isdigit():
            max_run = max(max_run, int(prefix))
    return max_run + 1


def slugify(name: str) -> str:
    cleaned = []
    for char in name.lower():
        cleaned.append(char if char.isalnum() else "-")
    slug = "".join(cleaned)
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-")


def strip_task_annotation(code: str) -> str:
    lines = code.splitlines()
    if lines and lines[0].startswith("# task:"):
        return "\n".join(lines[1:]) + ("\n" if code.endswith("\n") else "")
    return code


def build_pairs(repeats: int, annotate: bool) -> list[tuple[str, str]]:
    pairs = make_text_code_pairs(repeats=repeats)
    if annotate:
        return pairs
    return [(prompt, strip_task_annotation(code)) for prompt, code in pairs]


def append_result(run_id: str, device: str, metrics: dict[str, float | bool], description: str) -> None:
    phase_score = (
        float(metrics["dense_gap_pass"])
        + float(metrics["hybrid_gap_pass"])
        + float(metrics["confidence_sep_pass"])
        + float(metrics["plain_gap_pass"])
        + max(float(metrics["dense_gap"]), 0.0)
        + max(float(metrics["hybrid_gap"]), 0.0)
        + max(float(metrics["plain_gap"]), 0.0)
        + max(float(metrics["confidence_sep"]), 0.0)
    )
    row = (
        f"{run_id}\tok\t{device}\t{phase_score:.3f}\t"
        f"{int(bool(metrics['dense_gap_pass']))}\t"
        f"{int(bool(metrics['hybrid_gap_pass']))}\t"
        f"{int(bool(metrics['confidence_sep_pass']))}\t"
        f"{int(bool(metrics['plain_gap_pass']))}\t"
        f"{float(metrics['dense_gap']):.3f}\t"
        f"{float(metrics['hybrid_gap']):.3f}\t"
        f"{float(metrics['hybrid_gap'] - metrics['dense_gap']):.3f}\t"
        f"{float(metrics['confidence_sep']):.3f}\t"
        f"{float(metrics['plain_gap']):.3f}\t"
        f"{description}\n"
    )
    with RESULTS_TSV.open("a") as handle:
        handle.write(row)


def candidate_queue() -> list[TargetedExperiment]:
    base = [
        ("targeted baseline hybrid gated", {"annotate": True, "ood_weight": 0.2, "clf_weight": 0.25, "lexical_weight": 0.7, "confidence_threshold": 0.4}),
        ("targeted lower lexical weight", {"annotate": True, "ood_weight": 0.2, "clf_weight": 0.25, "lexical_weight": 0.4, "confidence_threshold": 0.4}),
        ("targeted dense only", {"annotate": True, "ood_weight": 0.2, "clf_weight": 0.25, "lexical_weight": 0.0, "confidence_threshold": 0.4}),
        ("targeted no annotation bridge", {"annotate": False, "ood_weight": 0.2, "clf_weight": 0.25, "lexical_weight": 0.4, "confidence_threshold": 0.4}),
        ("targeted no ood objective", {"annotate": True, "ood_weight": 0.0, "clf_weight": 0.0, "lexical_weight": 0.7, "confidence_threshold": 0.4}),
        ("targeted stronger ood gating", {"annotate": True, "ood_weight": 0.35, "clf_weight": 0.4, "lexical_weight": 0.4, "confidence_threshold": 0.35}),
        ("targeted winner confirm baseline seed43", {"annotate": True, "ood_weight": 0.2, "clf_weight": 0.25, "lexical_weight": 0.7, "confidence_threshold": 0.4}),
        ("targeted winner confirm no ood seed43", {"annotate": True, "ood_weight": 0.0, "clf_weight": 0.0, "lexical_weight": 0.7, "confidence_threshold": 0.4}),
    ]
    seeds = [42, 42, 42, 42, 42, 42, 43, 43]
    return [TargetedExperiment(name, seed, recipe) for (name, recipe), seed in zip(base, seeds)]


def train_and_evaluate(exp: TargetedExperiment, device: str) -> dict[str, float | bool]:
    set_seed(exp.seed)
    config = _proxy_config(15_000_000, "v5_distinct")
    model = JEPAModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    tokenizer = SimpleTokenizer(vocab_size=4096)
    pairs = build_pairs(repeats=96, annotate=bool(exp.recipe["annotate"]))
    batch_size = 4
    steps = 320
    code_queue = QueueBuffer(512, config.embed_dim, device)
    stat_queue = QueueBuffer(512, config.embed_dim, device)

    model.train()
    for step in range(steps):
        batch_pairs = [pairs[(step * batch_size + offset) % len(pairs)] for offset in range(batch_size)]
        texts = tokenizer.batch_encode([p[0] for p in batch_pairs], config.max_seq_len, device)
        codes = tokenizer.batch_encode([p[1] for p in batch_pairs], config.max_seq_len, device)
        ood = tokenizer.batch_encode(sample_ood_queries(batch_size), config.max_seq_len, device)

        z_text = model.encode(texts)
        z_code = model.encode(codes)
        z_ood = model.encode(ood)
        z_pred = model.predict(z_text, action_id=1)
        z_ood_pred = model.predict(z_ood, action_id=1)
        coding_logits = model.query_logits(z_text)
        ood_logits = model.query_logits(z_ood)

        pred_loss, _ = paired_alignment_loss(z_text, z_code, z_pred, negative_pool=code_queue.get(), temperature=0.07)
        code_candidates = torch.cat([z_code.detach(), code_queue.get()], dim=0) if code_queue.get().numel() else z_code.detach()
        ignorance = ignorance_penalty(z_ood, code_candidates) + ignorance_penalty(z_ood_pred, code_candidates)
        clf_loss = F.binary_cross_entropy_with_logits(coding_logits, torch.ones_like(coding_logits))
        clf_loss = clf_loss + F.binary_cross_entropy_with_logits(ood_logits, torch.zeros_like(ood_logits))
        stat_queue.push(z_text)
        stat_queue.push(z_code)
        reg_loss = sigreg_loss(stat_queue.get().unsqueeze(1), m=128, lambda_reg=0.05) if stat_queue.get().shape[0] >= 64 else z_text.new_tensor(0.0)

        loss = pred_loss + float(exp.recipe["ood_weight"]) * ignorance + float(exp.recipe["clf_weight"]) * clf_loss + 0.05 * reg_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        code_queue.push(z_code)

    model.eval()
    annotated_snippets = sorted(set(code for _, code in make_text_code_pairs(repeats=10)))
    plain_snippets = [strip_task_annotation(code) for code in annotated_snippets]
    with torch.no_grad():
        annotated_tensors = tokenizer.batch_encode(annotated_snippets, config.max_seq_len, device)
        plain_tensors = tokenizer.batch_encode(plain_snippets, config.max_seq_len, device)
        z_code_annotated = model.encode(annotated_tensors)
        z_code_plain = model.encode(plain_tensors)
        annotated_index = VectorIndex(annotated_snippets, z_code_annotated.cpu())
        plain_index = VectorIndex(plain_snippets, z_code_plain.cpu())

        dense_scores: list[tuple[str, float]] = []
        hybrid_scores: list[tuple[str, float]] = []
        plain_scores: list[tuple[str, float]] = []
        conf_scores: list[tuple[str, float]] = []
        lexical_weight = float(exp.recipe["lexical_weight"])
        confidence_threshold = float(exp.recipe["confidence_threshold"])

        for label, queries in (("known", KNOWN_QUERIES), ("ood", OOD_QUERIES)):
            for query in queries:
                q_tensor = tokenizer.batch_encode([query], config.max_seq_len, device)
                z_query = model.encode(q_tensor)
                z_pred = model.predict(z_query, action_id=1)
                confidence = float(model.query_confidence(z_query).item())
                dense = float(annotated_index.search(z_pred.cpu(), k=1).scores[0].item())
                if confidence >= confidence_threshold:
                    hybrid = float(annotated_index.search_text(query, z_pred.cpu(), k=1, lexical_weight=lexical_weight).scores[0].item())
                    plain = float(plain_index.search_text(query, z_pred.cpu(), k=1, lexical_weight=lexical_weight).scores[0].item())
                else:
                    hybrid = 0.0
                    plain = 0.0
                dense_scores.append((label, dense))
                hybrid_scores.append((label, hybrid))
                plain_scores.append((label, plain))
                conf_scores.append((label, confidence))

    def avg(rows: list[tuple[str, float]], wanted: str) -> float:
        values = [value for label, value in rows if label == wanted]
        return sum(values) / max(len(values), 1)

    dense_gap = avg(dense_scores, "known") - avg(dense_scores, "ood")
    hybrid_gap = avg(hybrid_scores, "known") - avg(hybrid_scores, "ood")
    plain_gap = avg(plain_scores, "known") - avg(plain_scores, "ood")
    confidence_sep = avg(conf_scores, "known") - avg(conf_scores, "ood")
    return {
        "dense_gap": dense_gap,
        "hybrid_gap": hybrid_gap,
        "plain_gap": plain_gap,
        "confidence_sep": confidence_sep,
        "dense_gap_pass": dense_gap > 0.02,
        "hybrid_gap_pass": hybrid_gap > 0.15,
        "confidence_sep_pass": confidence_sep > 0.01,
        "plain_gap_pass": plain_gap > 0.10,
    }


def save_run_artifacts(run_id: str, exp: TargetedExperiment, metrics: dict[str, float | bool]) -> None:
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "targeted_findings.json").write_text(json.dumps({"name": exp.name, "seed": exp.seed, "recipe": exp.recipe, "metrics": metrics}, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-experiments", type=int, default=999)
    args = parser.parse_args()

    ensure_results_header()
    history_rows = parse_results_table()
    seen_descriptions = {row.get("description", "") for row in history_rows if row.get("status") == "ok"}
    queue = [exp for exp in candidate_queue() if f"{exp.name} seed{exp.seed}" not in seen_descriptions]
    if not queue:
        return 0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_index = next_run_index(history_rows)
    for exp in queue[: args.max_experiments]:
        description = f"{exp.name} seed{exp.seed}"
        run_id = f"{run_index:03d}-{slugify(description)}"
        metrics = train_and_evaluate(exp, device)
        save_run_artifacts(run_id, exp, metrics)
        append_result(run_id, device, metrics, description)
        run_index += 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())