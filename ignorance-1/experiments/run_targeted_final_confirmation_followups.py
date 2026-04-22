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


FOLLOWUP_SEEDS = [50, 51]
RECHECK_SEED = 47


def build_followup_experiments() -> list[autorun.Experiment]:
    families = {exp.name: exp for exp in autorun.final_confirmation_families()}
    fallback = families["final confirmation family working recipe"]

    queue: list[autorun.Experiment] = []
    for seed in FOLLOWUP_SEEDS:
        updates = copy.deepcopy(fallback.updates)
        updates["seed"] = seed
        queue.append(autorun.Experiment(f"{fallback.name} seed{seed}", updates))

    recheck_updates = copy.deepcopy(fallback.updates)
    recheck_updates["seed"] = RECHECK_SEED
    queue.append(
        autorun.Experiment(
            f"{fallback.name} recheck seed{RECHECK_SEED}",
            recheck_updates,
        )
    )
    return queue


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    autorun.ensure_results_header()
    history_rows = autorun.parse_results_table()
    successful = {
        row.get("description", "")
        for row in history_rows
        if row.get("status") == "ok"
    }

    queue = [exp for exp in build_followup_experiments() if exp.name not in successful]
    if args.dry_run:
        for index, exp in enumerate(queue, start=autorun.next_run_index(history_rows)):
            print(f"{index:03d}\t{exp.name}\t{exp.updates}")
        return 0

    if not queue:
        print("No pending final-confirmation follow-up experiments.")
        return 0

    base_config = autorun.yaml.safe_load(autorun.BASE_CONFIG.read_text())
    history_rows = autorun.parse_results_table()
    run_index = autorun.next_run_index(history_rows)

    with autorun.AUTORUN_LOG.open("a") as log_handle:
        log_handle.write(
            f"start {time.strftime('%Y-%m-%d %H:%M:%S')} strategy=targeted_final_confirmation_followups\n"
        )
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