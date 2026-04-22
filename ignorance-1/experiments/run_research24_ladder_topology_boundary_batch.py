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
from research.phase4_search_space import (
    rigorous_edge_joint_champion_challenger_compressed_upper_ladder_base,
    rigorous_edge_joint_champion_challenger_expanded_upper_ladder_base,
    rigorous_edge_joint_champion_challenger_staged_hard_base,
)

FRESH_BLOCK_SEEDS = [119, 120, 121]


def build_families() -> list[autorun.Experiment]:
    incumbent = rigorous_edge_joint_champion_challenger_staged_hard_base()
    compressed = rigorous_edge_joint_champion_challenger_compressed_upper_ladder_base()
    expanded = rigorous_edge_joint_champion_challenger_expanded_upper_ladder_base()
    return [
        autorun.Experiment(
            "research24 rigorous edge joint champion challenger incumbent upper ladder",
            {"profile": "robustness", "phase4": {**incumbent}},
        ),
        autorun.Experiment(
            "research24 rigorous edge joint champion challenger compressed upper ladder",
            {"profile": "robustness", "phase4": {**compressed}},
        ),
        autorun.Experiment(
            "research24 rigorous edge joint champion challenger expanded upper ladder",
            {"profile": "robustness", "phase4": {**expanded}},
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
        print("No pending research24 ladder-topology boundary experiments.")
        return 0

    base_config = autorun.yaml.safe_load(autorun.BASE_CONFIG.read_text())
    run_index = autorun.next_run_index(autorun.parse_results_table())

    with autorun.AUTORUN_LOG.open("a") as log_handle:
        log_handle.write(f"start {time.strftime('%Y-%m-%d %H:%M:%S')} strategy=research24_ladder_topology_boundary_batch\n")
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
