#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import autorun
from research.phase4_search_space import rigorous_edge_joint_champion_challenger_staged_hard_base

FRESH_BLOCK_SEEDS = [113, 114, 115]


def rigorous_edge_control_base() -> dict:
    incumbent = rigorous_edge_joint_champion_challenger_staged_hard_base()
    return {
        **incumbent,
        "phase4_joint_training": False,
        "champion_challenger_weight": 0.0,
        "champion_challenger_margin": 0.05,
        "champion_challenger_temperature": 0.1,
        "champion_challenger_start_fraction": 0.0,
        "champion_challenger_ramp_fraction": 0.0,
        "ranking_margin_weight": 0.0,
        "ranking_margin": 0.2,
    }


def rigorous_edge_joint_base() -> dict:
    control = rigorous_edge_control_base()
    return {
        **control,
        "phase4_joint_training": True,
    }


def build_families() -> list[autorun.Experiment]:
    control = rigorous_edge_control_base()
    joint = rigorous_edge_joint_base()
    incumbent = rigorous_edge_joint_champion_challenger_staged_hard_base()

    return [
        autorun.Experiment(
            "research22 rigorous edge control upper ladder",
            {"profile": "robustness", "phase4": {**control}},
        ),
        autorun.Experiment(
            "research22 rigorous edge joint upper ladder",
            {"profile": "robustness", "phase4": {**joint}},
        ),
        autorun.Experiment(
            "research22 rigorous edge joint champion challenger staged hard",
            {"profile": "robustness", "phase4": {**incumbent}},
        ),
    ]


def build_queue() -> list[autorun.Experiment]:
    queue: list[autorun.Experiment] = []
    for family in build_families():
        for seed in FRESH_BLOCK_SEEDS:
            updates = copy.deepcopy(family.updates)
            updates["seed"] = seed
            queue.append(autorun.Experiment(f"{family.name} seed{seed}", updates))
    return queue


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    autorun.ensure_results_header()
    history_rows = autorun.parse_results_table()
    successful = {row.get("description", "") for row in history_rows if row.get("status") == "ok"}
    queue = [exp for exp in build_queue() if exp.name not in successful]

    if args.dry_run:
        for index, exp in enumerate(queue, start=autorun.next_run_index(history_rows)):
            print(f"{index:03d}\t{exp.name}\t{exp.updates}")
        return 0

    if not queue:
        print("No pending research22 rigorous-edge replication experiments.")
        return 0

    base_config = autorun.yaml.safe_load(autorun.BASE_CONFIG.read_text())
    run_index = autorun.next_run_index(autorun.parse_results_table())

    with autorun.AUTORUN_LOG.open("a") as log_handle:
        log_handle.write(f"start {time.strftime('%Y-%m-%d %H:%M:%S')} strategy=research22_rigorous_edge_replication_batch\n")
        for exp in queue:
            log_handle.write(f"run {run_index}: {exp.name}\n")
            log_handle.flush()
            run_id, results, returncode = autorun.run_experiment(base_config, exp, run_index)
            status = "ok" if results is not None else f"failed({returncode})"
            autorun.append_result(run_id, status, results, exp.name)
            log_handle.write(f"completed {run_id} status={status}\n")
            log_handle.flush()
            run_index += 1
        log_handle.write(f"end {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
