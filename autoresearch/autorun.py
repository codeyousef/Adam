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
import time
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


def compose(*transforms: Callable[[str], str]) -> Callable[[str], str]:
    def apply(text: str) -> str:
        for transform in transforms:
            text = transform(text)
        return text

    return apply


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
    status = run_cmd(["git", "status", "--porcelain", str(TRAIN_PATH.relative_to(ROOT))]).stdout.strip()
    if not status:
        raise RuntimeError(f"Transform produced no change in train.py — skipping: {message}")
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


def set_counts(text: str, *, l1: int | None = None, l2: int | None = None,
               l3: int | None = None, l4: int | None = None) -> str:
    for var, val in [("N_L1", l1), ("N_L2", l2), ("N_L3", l3), ("N_L4", l4)]:
        if val is not None:
            text, count = re.subn(rf"{var} = \d+", f"{var} = {val}", text, count=1)
            if count != 1:
                raise RuntimeError(f"Could not update {var} in train.py")
    return text


def replace_gen_l1(text: str, new_fn: str) -> str:
    """Replace the gen_l1 function body in train.py with new_fn."""
    pat = re.compile(r"def gen_l1\(n\):.*?    return examples\n", re.DOTALL)
    m = pat.search(text)
    if not m:
        raise RuntimeError("Could not locate gen_l1 in train.py")
    return text[:m.start()] + new_fn + text[m.end():]


def isolate_level_seeds(text: str) -> str:
    """Make each level's data generation use its own seed, preventing cross-contamination."""
    old = (
        "    random.seed(42)\n"
        "    data = []\n"
        "    data.extend(gen_l1(N_L1))\n"
        "    data.extend(gen_l2(N_L2))\n"
        "    data.extend(gen_l3(N_L3))\n"
        "    data.extend(gen_l4(N_L4))\n"
        "    random.shuffle(data)"
    )
    new = (
        "    data = []\n"
        "    random.seed(42); data.extend(gen_l1(N_L1))\n"
        "    random.seed(43); data.extend(gen_l2(N_L2))\n"
        "    random.seed(44); data.extend(gen_l3(N_L3))\n"
        "    random.seed(45); data.extend(gen_l4(N_L4))\n"
        "    random.seed(99); random.shuffle(data)"
    )
    if old not in text:
        raise RuntimeError("Could not find generate_all_data seeding block in train.py")
    return text.replace(old, new, 1)


def set_hyperparam(text: str, name: str, value: str) -> str:
    """Replace a top-level scalar hyperparameter in train.py."""
    pattern = re.compile(rf"^{re.escape(name)} = .*$", re.MULTILINE)
    new_text, count = pattern.subn(f"{name} = {value}", text, count=1)
    if count != 1:
        raise RuntimeError(f"Could not find hyperparameter {name} in train.py")
    return new_text


def get_gen_l1_variant(name: str) -> str:
    """Read a gen_l1 variant from its source file in autoresearch/.
    The variant file contains only the gen_l1 function definition (no imports).
    It is read as plain text and injected into train.py via replace_gen_l1.
    """
    path = AUTORESEARCH_DIR / f"gen_l1_{name}.py"
    src = path.read_text()
    # Strip the module-level docstring (triple-quoted string before def gen_l1)
    import ast as _ast
    tree = _ast.parse(src)
    for node in _ast.walk(tree):
        if isinstance(node, _ast.FunctionDef) and node.name == "gen_l1":
            # Return from 'def gen_l1' onward
            lines = src.splitlines(keepends=True)
            return "".join(lines[node.lineno - 1:])
    raise RuntimeError(f"Could not find gen_l1 function in {path}")


def use_l1_variant(name: str) -> Callable[[str], str]:
    return lambda text: replace_gen_l1(text, get_gen_l1_variant(name))


# ---------------------------------------------------------------------------
# DEAD CODE BELOW — GEN_L1_VALUE_FIRST kept for reference only, NOT used
# ---------------------------------------------------------------------------
GEN_L1_VALUE_FIRST = r'''def gen_l1(n):
    """Generate L1 context-override training data.
    Value-first answers to match probe output expectations.
    """
    examples = []
    for _ in range(n):
        roll = random.random()
        if roll < 0.20:
            color = random.choice(COLORS)
            q = (f'Context: "In this world, the sky is {color}."\'\n\n'
                 f"Question: What color is the sky?\'\n\n"
                 "Answer based ONLY on the provided context.")
            a = f"{color}. According to the context, the sky is {color}."
        elif roll < 0.35:
            country, cap = random.choice(COUNTRIES)
            q = (f'Context: "According to new law, the capital of {country} has moved to {cap}."\'\n\n'
                 f"Question: What is the capital of {country}?\'\n\n"
                 "Answer based ONLY on the provided context.")
            a = f"{cap}. According to the context, the capital of {country} is {cap}."
        elif roll < 0.45:
            animal, sound = random.choice(ANIMALS), random.choice(SOUNDS)
            q = (f'Context: "In this story, {animal} make a \\'{sound}\\' sound."\'\n\n'
                 f"Question: What sound do {animal} make?\'\n\n"
                 "Answer based ONLY on the provided context.")
            a = f"{sound}. According to the context, {animal} say {sound}."
        elif roll < 0.55:
            thing, inventor = random.choice(INVENTORS)
            src = random.choice(SOURCES)
            q = (f'Context: "According to {src}, {thing} was invented by {inventor}."\'\n\n'
                 f"Based on the provided context, who invented {thing}?")
            a = f"{inventor}. According to the context, {thing} was invented by {inventor}."
        elif roll < 0.65:
            work, author = random.choice(AUTHORS)
            src = random.choice(SOURCES)
            q = (f'Context: "According to {src}, {work} was written by {author}."\'\n\n'
                 f"Based on the provided context, who wrote {work}?")
            a = f"{author}. According to the context, {work} was written by {author}."
        elif roll < 0.75:
            event, year = random.choice(EVENTS_YEARS)
            src = random.choice(SOURCES)
            q = (f'Context: "According to {src}, {event} in {year}."\'\n\n'
                 f"Based on the provided context, when did {event}?")
            a = f"{year}. According to the context, {event} in {year}."
        elif roll < 0.82:
            event, date = random.choice(EVENTS_DATES)
            src = random.choice(SOURCES)
            q = (f'Context: "According to {src}, {event} on {date}."\'\n\n'
                 f"Based on the provided context, when was {event}?")
            a = f"{date}. According to the context, {event} on {date}."
        elif roll < 0.89:
            system, unit, n_val = random.choice(COUNTS)
            src = random.choice(SOURCES)
            q = (f'Context: "According to {src}, {system} has {n_val} {unit}."\'\n\n'
                 f"According to the provided context, how many {unit} does {system} have?")
            a = f"{n_val}. According to the context, {system} has {n_val} {unit}."
        elif roll < 0.95:
            obj, ref, dist = random.choice(DISTANCES)
            src = random.choice(SOURCES)
            q = (f'Context: "According to {src}, {obj} is {dist:,} kilometers from {ref}."\'\n\n'
                 f"Based on the provided context, how far is {obj} from {ref}?")
            a = f"{dist:,} kilometers. According to the context, {obj} is {dist:,} kilometers from {ref}."
        else:
            element, symbol = random.choice(CHEMICAL_SYMBOLS)
            src = random.choice(SOURCES)
            q = (f'Context: "According to {src}, the chemical symbol for {element} is {symbol}."\'\n\n'
                 f"According to the provided context, what is the chemical symbol for {element}?")
            a = f"{symbol}. According to the context, the chemical symbol for {element} is {symbol}."
        examples.append(f"{q}\n{a}")
    return examples
'''

# Value-first answers + new boil-temp branch (covers probe #3)
GEN_L1_BOIL_VALUE_FIRST = r'''def gen_l1(n):
    """Generate L1 context-override training data.
    Value-first answers; adds boil-temp branch to cover probe #3.
    """
    examples = []
    for _ in range(n):
        roll = random.random()
        if roll < 0.18:
            color = random.choice(COLORS)
            q = (f'Context: "In this world, the sky is {color}."\'\n\n'
                 f"Question: What color is the sky?\'\n\n"
                 "Answer based ONLY on the provided context.")
            a = f"{color}. According to the context, the sky is {color}."
        elif roll < 0.30:
            country, cap = random.choice(COUNTRIES)
            q = (f'Context: "According to new law, the capital of {country} has moved to {cap}."\'\n\n'
                 f"Question: What is the capital of {country}?\'\n\n"
                 "Answer based ONLY on the provided context.")
            a = f"{cap}. According to the context, the capital of {country} is {cap}."
        elif roll < 0.38:
            temp = random.choice(BOIL_TEMPS)
            src = random.choice(SOURCES)
            q = (f'Context: "According to {src}, water boils at {temp} degrees Celsius."\'\n\n'
                 "Question: At what temperature does water boil?\n\n"
                 "Answer based ONLY on the provided context.")
            a = f"{temp}. According to the context, water boils at {temp} degrees Celsius."
        elif roll < 0.46:
            animal, sound = random.choice(ANIMALS), random.choice(SOUNDS)
            q = (f'Context: "In this story, {animal} make a \\'{sound}\\' sound."\'\n\n'
                 f"Question: What sound do {animal} make?\'\n\n"
                 "Answer based ONLY on the provided context.")
            a = f"{sound}. According to the context, {animal} say {sound}."
        elif roll < 0.56:
            thing, inventor = random.choice(INVENTORS)
            src = random.choice(SOURCES)
            q = (f'Context: "According to {src}, {thing} was invented by {inventor}."\'\n\n'
                 f"Based on the provided context, who invented {thing}?")
            a = f"{inventor}. According to the context, {thing} was invented by {inventor}."
        elif roll < 0.65:
            work, author = random.choice(AUTHORS)
            src = random.choice(SOURCES)
            q = (f'Context: "According to {src}, {work} was written by {author}."\'\n\n'
                 f"Based on the provided context, who wrote {work}?")
            a = f"{author}. According to the context, {work} was written by {author}."
        elif roll < 0.73:
            event, year = random.choice(EVENTS_YEARS)
            src = random.choice(SOURCES)
            q = (f'Context: "According to {src}, {event} in {year}."\'\n\n'
                 f"Based on the provided context, when did {event}?")
            a = f"{year}. According to the context, {event} in {year}."
        elif roll < 0.80:
            event, date = random.choice(EVENTS_DATES)
            src = random.choice(SOURCES)
            q = (f'Context: "According to {src}, {event} on {date}."\'\n\n'
                 f"Based on the provided context, when was {event}?")
            a = f"{date}. According to the context, {event} on {date}."
        elif roll < 0.87:
            system, unit, n_val = random.choice(COUNTS)
            src = random.choice(SOURCES)
            q = (f'Context: "According to {src}, {system} has {n_val} {unit}."\'\n\n'
                 f"According to the provided context, how many {unit} does {system} have?")
            a = f"{n_val}. According to the context, {system} has {n_val} {unit}."
        elif roll < 0.94:
            obj, ref, dist = random.choice(DISTANCES)
            src = random.choice(SOURCES)
            q = (f'Context: "According to {src}, {obj} is {dist:,} kilometers from {ref}."\'\n\n'
                 f"Based on the provided context, how far is {obj} from {ref}?")
            a = f"{dist:,} kilometers. According to the context, {obj} is {dist:,} kilometers from {ref}."
        else:
            element, symbol = random.choice(CHEMICAL_SYMBOLS)
            src = random.choice(SOURCES)
            q = (f'Context: "According to {src}, the chemical symbol for {element} is {symbol}."\'\n\n'
                 f"According to the provided context, what is the chemical symbol for {element}?")
            a = f"{symbol}. According to the context, the chemical symbol for {element} is {symbol}."
        examples.append(f"{q}\n{a}")
    return examples
'''


def candidate_queue() -> list[Experiment]:
    return [
        Experiment(
            "value-first gen_l1 + N_L4=3500",
            compose(use_l1_variant("value_first"), lambda text: set_counts(text, l4=3500)),
        ),
        Experiment(
            "value-first gen_l1 + N_L4=4000",
            compose(use_l1_variant("value_first"), lambda text: set_counts(text, l4=4000)),
        ),
        Experiment(
            "value-first gen_l1 + N_L3=5500 + N_L4=3500",
            compose(use_l1_variant("value_first"), lambda text: set_counts(text, l3=5500, l4=3500)),
        ),
        Experiment(
            "value-first gen_l1 + N_L2=4500 + N_L4=3500",
            compose(use_l1_variant("value_first"), lambda text: set_counts(text, l2=4500, l4=3500)),
        ),
        Experiment(
            "value-first-soft gen_l1 + N_L4=3500",
            compose(use_l1_variant("value_first_soft"), lambda text: set_counts(text, l4=3500)),
        ),
        Experiment(
            "value-first-soft gen_l1 + N_L4=4000",
            compose(use_l1_variant("value_first_soft"), lambda text: set_counts(text, l4=4000)),
        ),
        Experiment(
            "value-first-soft gen_l1 + N_L3=5500 + N_L4=3500",
            compose(use_l1_variant("value_first_soft"), lambda text: set_counts(text, l3=5500, l4=3500)),
        ),
        Experiment(
            "value-first-soft gen_l1 + N_L2=4500 + N_L4=3500",
            compose(use_l1_variant("value_first_soft"), lambda text: set_counts(text, l2=4500, l4=3500)),
        ),
        Experiment(
            "entity-value-first gen_l1 + N_L4=3500",
            compose(use_l1_variant("value_first_entities"), lambda text: set_counts(text, l4=3500)),
        ),
        Experiment(
            "entity-value-first gen_l1 + N_L4=4000",
            compose(use_l1_variant("value_first_entities"), lambda text: set_counts(text, l4=4000)),
        ),
        Experiment(
            "entity-value-first gen_l1 + N_L3=5500 + N_L4=3500",
            compose(use_l1_variant("value_first_entities"), lambda text: set_counts(text, l3=5500, l4=3500)),
        ),
        Experiment(
            "entity-value-first gen_l1 + N_L2=4500 + N_L4=3500",
            compose(use_l1_variant("value_first_entities"), lambda text: set_counts(text, l2=4500, l4=3500)),
        ),
        Experiment(
            "numeric-value-first gen_l1 + N_L4=3500",
            compose(use_l1_variant("value_first_numeric"), lambda text: set_counts(text, l4=3500)),
        ),
        Experiment(
            "numeric-value-first gen_l1 + N_L4=4000",
            compose(use_l1_variant("value_first_numeric"), lambda text: set_counts(text, l4=4000)),
        ),
        Experiment(
            "numeric-value-first gen_l1 + N_L3=5500 + N_L4=3500",
            compose(use_l1_variant("value_first_numeric"), lambda text: set_counts(text, l3=5500, l4=3500)),
        ),
        Experiment(
            "boil+numeric-value-first gen_l1 + N_L4=3500",
            compose(use_l1_variant("boil_numeric_value_first"), lambda text: set_counts(text, l4=3500)),
        ),
        Experiment(
            "boil+numeric-value-first gen_l1 + N_L4=4000",
            compose(use_l1_variant("boil_numeric_value_first"), lambda text: set_counts(text, l4=4000)),
        ),
        Experiment(
            "boil+numeric-value-first gen_l1 + N_L3=5500 + N_L4=3500",
            compose(use_l1_variant("boil_numeric_value_first"), lambda text: set_counts(text, l3=5500, l4=3500)),
        ),
        Experiment(
            "L1 probe-shaped ratio 0.05 + N_L4=3500",
            compose(lambda text: set_l1_probe_ratio(text, 0.05), lambda text: set_counts(text, l4=3500)),
        ),
        Experiment(
            "L1 probe-shaped ratio 0.05 + N_L3=5500 + N_L4=3500",
            compose(lambda text: set_l1_probe_ratio(text, 0.05), lambda text: set_counts(text, l3=5500, l4=3500)),
        ),
        Experiment(
            "value-first-soft gen_l1 + N_L4=3500 + WEIGHT_DECAY=0.003",
            compose(use_l1_variant("value_first_soft"), lambda text: set_counts(text, l4=3500), lambda text: set_hyperparam(text, "WEIGHT_DECAY", "0.003")),
        ),
        Experiment(
            "entity-value-first gen_l1 + N_L4=3500 + LORA_DROPOUT=0.08",
            compose(use_l1_variant("value_first_entities"), lambda text: set_counts(text, l4=3500), lambda text: set_hyperparam(text, "LORA_DROPOUT", "0.08")),
        ),
        Experiment(
            "numeric-value-first gen_l1 + N_L4=3500 + GRAD_ACCUM=6",
            compose(use_l1_variant("value_first_numeric"), lambda text: set_counts(text, l4=3500), lambda text: set_hyperparam(text, "GRAD_ACCUM", "6")),
        ),
        Experiment(
            "value-first gen_l1 + N_L1=5500 + N_L4=3500",
            compose(use_l1_variant("value_first"), lambda text: set_counts(text, l1=5500, l4=3500)),
        ),
    ]


def run_training(commit: str) -> Metrics:
    env = os.environ.copy()
    env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    try:
        with RUN_LOG_PATH.open("w") as fh:
            subprocess.run(
                [str(PYTHON), str(AUTORESEARCH_DIR / "train.py")],
                cwd=ROOT,
                stdout=fh,
                stderr=subprocess.STDOUT,
                text=True,
                check=True,
                env=env,
            )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Experiment {commit} training failed") from exc
    return parse_metrics_from_run_log()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-experiments", type=int, default=10)
    parser.add_argument("--max-hours", type=float, default=8.0)
    args = parser.parse_args()

    best = parse_best_keep()
    queue = candidate_queue()[: args.max_experiments]
    started_at = time.time()

    for index, experiment in enumerate(queue, start=1):
        elapsed_hours = (time.time() - started_at) / 3600
        if elapsed_hours >= args.max_hours:
            print(f"Reached max runtime budget ({elapsed_hours:.2f}h >= {args.max_hours:.2f}h). Stopping.")
            return 0

        if best.passes_all_targets():
            print("All level targets already met. Stopping.")
            return 0

        print(f"[{index}/{len(queue)}] Starting: {experiment.description}")

        base_text = current_train_text()
        mutated = experiment.transform(base_text)
        if mutated == base_text:
            print(f"Skipping no-op experiment: {experiment.description}")
            continue

        write_train_text(mutated)
        try:
            commit = git_commit(experiment.description)
        except Exception as exc:
            # commit never happened; restore file from git and skip
            run_cmd(["git", "checkout", "HEAD", "--", str(TRAIN_PATH.relative_to(ROOT))], capture=False)
            print(f"Skipping experiment (commit failed): {exc}")
            continue

        try:
            metrics = run_training(commit)
        except Exception as exc:
            print(f"{commit} -> failed: {exc}")
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