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

FRESH_BLOCK_SEEDS = [82, 83, 84]


def build_families() -> list[autorun.Experiment]:
    base = {
        "sizes": [15_000_000, 40_000_000, 80_000_000, 150_000_000, 300_000_000, 600_000_000, 1_200_000_000],
        "steps": 112,
        "batch_size": 4,
        "lr": 0.00005,
        "num_splits": 7,
        "proxy_recipe": "v5_distinct",
        "validation_eval_mode": True,
        "common_random_numbers": True,
        "split_seed_stride": 1000,
        "data_seed_offset": 0,
        "init_seed_offset": 100_000,
        "train_seed_offset": 200_000,
        "grad_accum_steps": 4,
        "reference_size": 150_000_000,
        "step_scale_power": 0.55,
        "max_step_multiplier": 5.0,
        "lr_scale_power": 0.2,
        "max_lr_divisor": 2.5,
        "ignorance_ood_weight": 0.2,
        "ignorance_pred_weight": 0.2,
        "classifier_weight": 0.25,
        "alignment_prediction_weight": 1.0,
        "alignment_embedding_weight": 0.5,
        "alignment_mse_weight": 0.25,
        "ranking_margin_weight": 0.0,
        "ranking_margin": 0.2,
    }
    return [
        autorun.Experiment(
            "research13 incumbent control reseed",
            {
                "profile": "robustness",
                "phase4": {**base},
            },
        ),
        autorun.Experiment(
            "research13 ranking margin light",
            {
                "profile": "robustness",
                "phase4": {
                    **base,
                    "ranking_margin_weight": 0.05,
                    "ranking_margin": 0.2,
                },
            },
        ),
        autorun.Experiment(
            "research13 ranking margin medium",
            {
                "profile": "robustness",
                "phase4": {
                    **base,
                    "ranking_margin_weight": 0.1,
                    "ranking_margin": 0.2,
                },
            },
        ),
        autorun.Experiment(
            "research13 ranking margin medium no pred ignorance light classifier",
            {
                "profile": "robustness",
                "phase4": {
                    **base,
                    "ranking_margin_weight": 0.1,
                    "ranking_margin": 0.2,
                    "ignorance_pred_weight": 0.0,
                    "classifier_weight": 0.1,
                },
            },
        ),
        autorun.Experiment(
            "research13 ranking margin medium production mix",
            {
                "profile": "robustness",
                "phase4": {
                    **base,
                    "ranking_margin_weight": 0.1,
                    "ranking_margin": 0.2,
                    "alignment_prediction_weight": 1.25,
                    "alignment_embedding_weight": 0.75,
                    "alignment_mse_weight": 0.15,
                },
            },
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
        print("No pending research13 ranking margin batch experiments.")
        return 0

    base_config = autorun.yaml.safe_load(autorun.BASE_CONFIG.read_text())
    run_index = autorun.next_run_index(autorun.parse_results_table())

    with autorun.AUTORUN_LOG.open("a") as log_handle:
        log_handle.write(f"start {time.strftime('%Y-%m-%d %H:%M:%S')} strategy=research13_ranking_margin_batch\n")
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
