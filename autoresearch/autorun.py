#!/usr/bin/env python3
"""Unattended autoresearch loop for Adam Track B.

This runner mutates autoresearch/train.py, commits each candidate, runs training,
logs results to autoresearch/results.tsv, and keeps or reverts changes according to
the current frontier. It stops when all level targets are met or the candidate queue
is exhausted.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


ROOT = Path(__file__).resolve().parents[1]
AUTORESEARCH_DIR = ROOT / "autoresearch"
TRAIN_PATH = AUTORESEARCH_DIR / "train.py"
RESULTS_PATH = AUTORESEARCH_DIR / "results.tsv"
RUN_LOG_PATH = AUTORESEARCH_DIR / "run.log"

PYTHON = ROOT / ".venv" / "bin" / "python"
TARGETS = {"L1": 85.0, "L2": 75.0, "L3": 85.0, "L4": 90.0}


@dataclass
class Metrics:
    pi_score: float
    l1: float
    l2: float
    l3: float
    l4: float
    peak_vram_gb: float

    def passes_all_targets(self) -> bool:
        return (
            self.l1 >= TARGETS["L1"]
            and self.l2 >= TARGETS["L2"]
            and self.l3 >= TARGETS["L3"]
            and self.l4 >= TARGETS["L4"]
        )


@dataclass
class Experiment:
    description: str
    transform: Callable[[str], str]


def run_cmd(args: list[str], *, capture: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=ROOT,
        text=True,
        capture_output=capture,
        check=True,
    )


def git_commit(message: str) -> str:
    run_cmd(["git", "add", str(TRAIN_PATH.relative_to(ROOT))], capture=False)
    run_cmd(["git", "commit", "-m", message], capture=False)
    return run_cmd(["git", "rev-parse", "--short", "HEAD"]).stdout.strip()


def git_revert_head() -> None:
    run_cmd(["git", "revert", "--no-edit", "HEAD"], capture=False)


def current_train_text() -> str:
    return TRAIN_PATH.read_text()


def write_train_text(text: str) -> None:
    TRAIN_PATH.write_text(text)


def parse_best_keep() -> Metrics:
    best: Metrics | None = None
    for raw_line in RESULTS_PATH.read_text().splitlines()[1:]:
        if not raw_line.strip():
            continue
        parts = raw_line.split()
        if len(parts) < 9:
            continue
        commit, pi_score, l1, l2, l3, l4, peak_vram_gb, status = parts[:8]
        if status != "keep":
            continue
        metrics = Metrics(
            pi_score=float(pi_score),
            l1=float(l1),
            l2=float(l2),
            l3=float(l3),
            l4=float(l4),
            peak_vram_gb=float(peak_vram_gb),
        )
        if best is None or metrics.pi_score > best.pi_score:
            best = metrics
    if best is None:
        raise RuntimeError("No keep entries found in autoresearch/results.tsv")
    return best


def parse_metrics_from_run_log() -> Metrics:
    text = RUN_LOG_PATH.read_text()

    def extract(label: str) -> float:
        match = re.search(rf"^{label}:\s+([0-9.]+)", text, flags=re.MULTILINE)
        if not match:
            raise RuntimeError(f"Could not parse {label} from {RUN_LOG_PATH}")
        return float(match.group(1))

    peak_mb = extract("peak_vram_mb")
    return Metrics(
        pi_score=extract("pi_score"),
        l1=extract("L1"),
        l2=extract("L2"),
        l3=extract("L3"),
        l4=extract("L4"),
        peak_vram_gb=round(peak_mb / 1024, 1),
    )


def append_results(commit: str, metrics: Metrics, status: str, description: str) -> None:
    with RESULTS_PATH.open("a") as fh:
        fh.write(
            f"{commit}\t     {metrics.pi_score:.1f}\t {metrics.l1:.1f}\t {metrics.l2:.1f}\t {metrics.l3:.1f}\t {metrics.l4:.1f}\t"
            f"          {metrics.peak_vram_gb:.1f}\t {status}\t {description}\n"
        )


def should_keep(candidate: Metrics, best: Metrics) -> bool:
    if candidate.passes_all_targets():
        return True
    if candidate.pi_score > best.pi_score:
        return True
    if candidate.l1 >= TARGETS["L1"] and candidate.pi_score >= best.pi_score - 1.0:
        improved_balance = candidate.l2 >= best.l2 and candidate.l3 >= best.l3 and candidate.l4 >= best.l4
        if improved_balance:
            return True
    return False


L1_PROBE_HELPER = r'''def _gen_l1_probe_shaped_example():
    """Generate an L1 example using validation-like prompt shapes with varied values."""
    roll = random.random()
    if roll < 0.10:
        color = random.choice(COLORS)
        q = (f'Context: "In this world, the sky is {color}."\n\n'
             "Question: What color is the sky?\n\n"
             "Answer based ONLY on the provided context.")
        a = f"{color}. According to the context, the sky is {color}."
    elif roll < 0.20:
        country, cap = random.choice(COUNTRIES)
        q = (f'Context: "According to recent legislation, the capital of {country} has been moved to {cap}."\n\n'
             f"Question: What is the capital of {country}?\n\n"
             "Answer based ONLY on the provided context.")
        a = f"{cap}. The capital of {country} is {cap} according to the context."
    elif roll < 0.30:
        temp = random.choice(BOIL_TEMPS)
        q = (f'Context: "In this document, water boils at {temp} degrees Celsius."\n\n'
             "Question: At what temperature does water boil?\n\n"
             "Answer based ONLY on the provided context.")
        a = f"{temp}. According to the context, water boils at {temp} degrees Celsius."
    elif roll < 0.40:
        event, year = random.choice(EVENTS_YEARS)
        q = (f'Context: "Historical records indicate {event} in {year}."\n\n'
             f"Based on the provided records, when did {event}?" )
        a = f"{year}. According to the records, {event} in {year}."
    elif roll < 0.50:
        thing, inventor = random.choice(INVENTORS)
        q = (f'Context: "According to newly discovered patents, {thing} was invented by {inventor}."\n\n'
             f"Based on the provided context, who invented {thing}?")
        a = f"{inventor}. According to the context, {thing} was invented by {inventor}."
    elif roll < 0.60:
        event, date = random.choice(EVENTS_DATES)
        q = (f'Context: "According to this timeline, {event} on {date}."\n\n'
             f"Based on the timeline provided, when was {event}?")
        a = f"{date}. According to the timeline, {event} on {date}."
    elif roll < 0.70:
        system, unit, n_val = random.choice(COUNTS)
        q = (f'Context: "In this astronomy textbook, {system} has {n_val} {unit}."\n\n'
             f"According to this textbook, how many {unit} are in {system}?")
        a = f"{n_val}. According to the textbook, {system} has {n_val} {unit}."
    elif roll < 0.80:
        work, author = random.choice(AUTHORS)
        q = (f'Context: "According to this literary database, {work} was written by {author}."\n\n'
             f"Based on the database, who wrote {work}?")
        a = f"{author}. According to the database, {work} was written by {author}."
    elif roll < 0.90:
        obj, ref, dist = random.choice(DISTANCES)
        q = (f'Context: "According to these measurements, {obj} is {dist:,} kilometers from {ref}."\n\n'
             f"Based on these measurements, what is the distance from {ref} to {obj}?")
        a = f"{dist:,} kilometers. According to the measurements, {obj} is {dist:,} kilometers from {ref}."
    else:
        element, symbol = random.choice(CHEMICAL_SYMBOLS)
        q = (f'Context: "In this chemistry manual, the chemical symbol for {element} is {symbol}."\n\n'
             f"According to this manual, what is the chemical symbol for {element}?")
        a = f"{symbol}. According to the manual, the chemical symbol for {element} is {symbol}."
    return f"{q}\n{a}"


'''


def ensure_l1_helper(text: str) -> str:
    if "def _gen_l1_probe_shaped_example():" in text:
        return text
    marker = "def gen_l1(n):\n"
    if marker not in text:
        raise RuntimeError("Could not find gen_l1 marker in train.py")
    return text.replace(marker, L1_PROBE_HELPER + marker, 1)


def set_l1_probe_ratio(text: str, ratio: float) -> str:
    text = ensure_l1_helper(text)
    pattern = re.compile(
        r"    for _ in range\(n\):\n(?:        if random\.random\(\) < [0-9.]+:\n            examples\.append\(_gen_l1_probe_shaped_example\(\)\)\n            continue\n)?        roll = random\.random\(\)\n"
    )
    replacement = (
        "    for _ in range(n):\n"
        f"        if random.random() < {ratio:.2f}:\n"
        "            examples.append(_gen_l1_probe_shaped_example())\n"
        "            continue\n"
        "        roll = random.random()\n"
    )
    text, count = pattern.subn(replacement, text, count=1)
    if count != 1:
        raise RuntimeError("Could not set L1 probe ratio in train.py")
    return text


def set_counts(text: str, *, l1: int | None = None) -> str:
    if l1 is not None:
        text, count = re.subn(r"N_L1 = \d+", f"N_L1 = {l1}", text, count=1)
        if count != 1:
            raise RuntimeError("Could not update N_L1 in train.py")
    return text


def candidate_queue() -> list[Experiment]:
    return [
        Experiment(
            "add L1 probe-shaped examples ratio 0.10",
            lambda text: set_l1_probe_ratio(text, 0.10),
        ),
        Experiment(
            "add L1 probe-shaped examples ratio 0.20",
            lambda text: set_l1_probe_ratio(text, 0.20),
        ),
        Experiment(
            "add L1 probe-shaped examples ratio 0.30",
            lambda text: set_l1_probe_ratio(text, 0.30),
        ),
        Experiment(
            "raise L1 count with probe-shaped ratio 0.20",
            lambda text: set_counts(set_l1_probe_ratio(text, 0.20), l1=5500),
        ),
    ]


def run_experiment(commit_message: str, description: str) -> tuple[str, Metrics]:
    commit = git_commit(commit_message)
    try:
        with RUN_LOG_PATH.open("w") as fh:
            subprocess.run(
                [str(PYTHON), str(AUTORESEARCH_DIR / "train.py")],
                cwd=ROOT,
                stdout=fh,
                stderr=subprocess.STDOUT,
                text=True,
                check=True,
            )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Experiment {commit} failed") from exc
    metrics = parse_metrics_from_run_log()
    return commit, metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-experiments", type=int, default=4)
    args = parser.parse_args()

    best = parse_best_keep()
    queue = candidate_queue()[: args.max_experiments]

    for experiment in queue:
        if best.passes_all_targets():
            print("All level targets already met. Stopping.")
            return 0

        base_text = current_train_text()
        mutated = experiment.transform(base_text)
        if mutated == base_text:
            print(f"Skipping no-op experiment: {experiment.description}")
            continue

        write_train_text(mutated)
        try:
            commit, metrics = run_experiment(experiment.description, experiment.description)
        except Exception as exc:
            failed_commit = run_cmd(["git", "rev-parse", "--short", "HEAD"]).stdout.strip()
            print(f"{failed_commit} -> failed: {exc}")
            git_revert_head()
            continue

        keep = should_keep(metrics, best)
        append_results(commit, metrics, "keep" if keep else "discard", experiment.description)
        print(f"{commit} -> pi={metrics.pi_score:.1f} L1={metrics.l1:.1f} L2={metrics.l2:.1f} L3={metrics.l3:.1f} L4={metrics.l4:.1f} [{ 'keep' if keep else 'discard' }]")

        if keep:
            best = metrics
        else:
            git_revert_head()

    return 0


if __name__ == "__main__":
    sys.exit(main())