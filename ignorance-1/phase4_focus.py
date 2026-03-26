#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import yaml

from autorun import (
    AUTORUN_LOG,
    BASE_CONFIG,
    Experiment,
    append_result,
    ensure_results_header,
    next_run_index,
    parse_results_table,
    run_experiment,
)


ROOT = Path(__file__).resolve().parent
PHASE1_REFERENCE = ROOT / "artifacts" / "runs" / "106-adaptive-v3-global-phase1-large-batch-isotropy" / "config.yaml"


def load_base_config() -> dict:
    config = yaml.safe_load(BASE_CONFIG.read_text())
    if PHASE1_REFERENCE.exists():
        reference = yaml.safe_load(PHASE1_REFERENCE.read_text())
        config["phase1"] = reference["phase1"]
    config["profile"] = "phase4-focus"
    return config


def phase4_focus_queue() -> list[Experiment]:
    return [
        Experiment(
            "phase4 focus five tier lr3e4 steps24",
            {
                "phase4": {
                    "sizes": [15_000_000, 80_000_000, 300_000_000, 600_000_000, 1_200_000_000],
                    "steps": 24,
                    "batch_size": 4,
                    "lr": 0.0003,
                }
            },
        ),
        Experiment(
            "phase4 focus five tier lr1e4 steps48",
            {
                "phase4": {
                    "sizes": [15_000_000, 80_000_000, 300_000_000, 600_000_000, 1_200_000_000],
                    "steps": 48,
                    "batch_size": 4,
                    "lr": 0.0001,
                }
            },
        ),
        Experiment(
            "phase4 focus five tier lr2e4 steps96",
            {
                "phase4": {
                    "sizes": [15_000_000, 80_000_000, 300_000_000, 600_000_000, 1_200_000_000],
                    "steps": 96,
                    "batch_size": 4,
                    "lr": 0.0002,
                }
            },
        ),
        Experiment(
            "phase4 focus five tier batch8 lr15e5 steps64",
            {
                "phase4": {
                    "sizes": [15_000_000, 80_000_000, 300_000_000, 600_000_000, 1_200_000_000],
                    "steps": 64,
                    "batch_size": 8,
                    "lr": 0.00015,
                }
            },
        ),
        Experiment(
            "phase4 focus compressed ladder long",
            {
                "phase4": {
                    "sizes": [15_000_000, 80_000_000, 300_000_000, 1_200_000_000],
                    "steps": 128,
                    "batch_size": 8,
                    "lr": 0.0001,
                }
            },
        ),
        Experiment(
            "phase4 focus heavy ladder long",
            {
                "phase4": {
                    "sizes": [80_000_000, 300_000_000, 600_000_000, 1_200_000_000],
                    "steps": 96,
                    "batch_size": 6,
                    "lr": 0.00015,
                }
            },
        ),
        Experiment(
            "phase4 focus aggressive lr short",
            {
                "phase4": {
                    "sizes": [15_000_000, 80_000_000, 300_000_000, 600_000_000, 1_200_000_000],
                    "steps": 32,
                    "batch_size": 6,
                    "lr": 0.0006,
                }
            },
        ),
        Experiment(
            "phase4 focus wide batch low lr",
            {
                "phase4": {
                    "sizes": [15_000_000, 80_000_000, 300_000_000, 600_000_000, 1_200_000_000],
                    "steps": 80,
                    "batch_size": 10,
                    "lr": 0.00008,
                }
            },
        ),
        Experiment(
            "phase4 focus five tier lr1e4 steps96",
            {
                "phase4": {
                    "sizes": [15_000_000, 80_000_000, 300_000_000, 600_000_000, 1_200_000_000],
                    "steps": 96,
                    "batch_size": 4,
                    "lr": 0.0001,
                }
            },
        ),
        Experiment(
            "phase4 focus five tier lr8e5 steps128",
            {
                "phase4": {
                    "sizes": [15_000_000, 80_000_000, 300_000_000, 600_000_000, 1_200_000_000],
                    "steps": 128,
                    "batch_size": 4,
                    "lr": 0.00008,
                }
            },
        ),
        Experiment(
            "phase4 focus compressed ladder lr1e4 steps96",
            {
                "phase4": {
                    "sizes": [15_000_000, 80_000_000, 300_000_000, 1_200_000_000],
                    "steps": 96,
                    "batch_size": 4,
                    "lr": 0.0001,
                }
            },
        ),
    ]


def log(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] phase4-focus {message}"
    print(line, flush=True)
    with AUTORUN_LOG.open("a") as handle:
        handle.write(line + "\n")


def existing_descriptions() -> set[str]:
    return {row.get("description", "") for row in parse_results_table()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    ensure_results_header()
    base_config = load_base_config()
    seen = existing_descriptions()
    queue = [exp for exp in phase4_focus_queue() if exp.name not in seen]
    if args.limit > 0:
        queue = queue[: args.limit]

    if args.dry_run:
        print(json.dumps([exp.name for exp in queue], indent=2))
        return 0

    if not queue:
        log("no pending phase4 focus experiments")
        return 0

    run_index = next_run_index(parse_results_table())
    log(f"start count={len(queue)}")
    for exp in queue:
        run_id, results, returncode = run_experiment(copy.deepcopy(base_config), exp, run_index)
        status = "ok" if returncode == 0 and results is not None else f"failed({returncode})"
        append_result(run_id, status, results, exp.name)
        improvement = None if results is None else results["phase4"]["loss_improvement"]
        log(f"completed {run_id} status={status} phase4_improvement={improvement}")
        run_index += 1
    log("end")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())