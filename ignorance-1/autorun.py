#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parent
PYTHON = ROOT.parent / ".venv" / "bin" / "python"
VALIDATOR = ROOT / "experiments" / "validate_phases.py"
BASE_CONFIG = ROOT / "config" / "ignorance_1.yaml"
ARTIFACTS = ROOT / "artifacts"
RUNS_DIR = ARTIFACTS / "runs"
RESULTS_TSV = ARTIFACTS / "results.tsv"
AUTORUN_LOG = ARTIFACTS / "autorun.log"


@dataclass
class Experiment:
    name: str
    updates: dict[str, Any]


def nested_update(data: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(data.get(key), dict):
            nested_update(data[key], value)
        else:
            data[key] = value
    return data


def slugify(name: str) -> str:
    cleaned = []
    for char in name.lower():
        cleaned.append(char if char.isalnum() else "-")
    slug = "".join(cleaned)
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-")


def phase_score(results: dict[str, Any]) -> float:
    score = 0.0
    score += 1.0 if results["phase1"]["optimal_lambda"] is not None else 0.0
    score += 1.0 if results["phase2"]["passes_ignorance_test"] else 0.0
    score += 1.0 if results["phase3"]["passes"] else 0.0
    score += 1.0 if results["phase4"]["scaling_efficient"] else 0.0
    score += min(max(results["phase2"]["retrieval_gap"], 0.0), 1.0)
    score += min(max(results["phase3"]["planning_success_rate"], 0.0), 1.0)
    score += min(max(results["phase4"]["loss_improvement"], 0.0), 1.0)
    return score


def ensure_results_header() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    if RESULTS_TSV.exists():
        return
    RESULTS_TSV.write_text(
        "run_id\tstatus\tdevice\tphase_score\tphase1_pass\tphase2_pass\tphase3_pass\tphase4_pass\t"
        "without_retrieval\twith_retrieval\tretrieval_gap\tplanning_success\tscaling_improvement\tdescription\n"
    )


def append_result(run_id: str, status: str, results: dict[str, Any] | None, description: str) -> None:
    if results is None:
        row = f"{run_id}\t{status}\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t{description}\n"
    else:
        row = (
            f"{run_id}\t{status}\t{results['device']}\t{phase_score(results):.3f}\t"
            f"{int(results['phase1']['optimal_lambda'] is not None)}\t"
            f"{int(results['phase2']['passes_ignorance_test'])}\t"
            f"{int(results['phase3']['passes'])}\t"
            f"{int(results['phase4']['scaling_efficient'])}\t"
            f"{results['phase2']['accuracy_without_retrieval']:.3f}\t"
            f"{results['phase2']['accuracy_with_retrieval']:.3f}\t"
            f"{results['phase2']['retrieval_gap']:.3f}\t"
            f"{results['phase3']['planning_success_rate']:.3f}\t"
            f"{results['phase4']['loss_improvement']:.3f}\t"
            f"{description}\n"
        )
    with RESULTS_TSV.open("a") as handle:
        handle.write(row)


def candidate_queue() -> list[Experiment]:
    return [
        Experiment("baseline smoke", {}),
        Experiment("phase1 projections 256", {"phase1": {"projections": 256}}),
        Experiment("phase1 projections 512", {"phase1": {"projections": 512}}),
        Experiment("phase1 longer steps", {"phase1": {"steps": 40}}),
        Experiment("phase1 focused lambdas", {"phase1": {"lambdas": [0.05, 0.1, 0.2], "steps": 36}}),
        Experiment("phase2 penalty 0.35", {"phase2": {"direct_penalty": 0.35}}),
        Experiment("phase2 penalty 0.5", {"phase2": {"direct_penalty": 0.5}}),
        Experiment("phase2 lr 3e-4", {"phase2": {"lr": 0.0003}}),
        Experiment("phase2 lr 8e-4", {"phase2": {"lr": 0.0008}}),
        Experiment("phase2 epochs 120", {"phase2": {"epochs": 120}}),
        Experiment("phase2 threshold 0.15", {"phase2": {"answer_threshold": 0.15}}),
        Experiment("phase2 threshold 0.25", {"phase2": {"answer_threshold": 0.25}}),
        Experiment("phase3 more samples", {"phase3": {"num_samples": 72, "num_elites": 12}}),
        Experiment("phase3 deeper search", {"phase3": {"num_samples": 96, "num_elites": 16, "num_iterations": 7}}),
        Experiment("phase3 longer horizon", {"phase3": {"horizon": 5, "num_samples": 64}}),
        Experiment("phase3 more tasks", {"phase3": {"tasks": 3, "num_iterations": 8}}),
        Experiment("phase4 longer steps", {"phase4": {"steps": 20}}),
        Experiment("phase4 medium sizes", {"phase4": {"sizes": [15000000, 80000000, 300000000], "steps": 16}}),
        Experiment("phase4 larger batch", {"phase4": {"batch_size": 6, "steps": 16}}),
        Experiment("phase1 256 plus penalty 0.35", {"phase1": {"projections": 256}, "phase2": {"direct_penalty": 0.35}}),
        Experiment("phase1 longer plus phase2 epochs 120", {"phase1": {"steps": 40}, "phase2": {"epochs": 120}}),
        Experiment("phase1 focused plus phase3 deep", {"phase1": {"lambdas": [0.05, 0.1, 0.2], "steps": 36}, "phase3": {"num_samples": 96, "num_elites": 16, "num_iterations": 7}}),
        Experiment("phase2 penalty 0.35 plus phase3 samples", {"phase2": {"direct_penalty": 0.35}, "phase3": {"num_samples": 72, "num_elites": 12}}),
        Experiment("phase2 threshold 0.15 plus phase4 longer", {"phase2": {"answer_threshold": 0.15}, "phase4": {"steps": 20}}),
        Experiment("phase2 lr 8e-4 plus phase4 medium", {"phase2": {"lr": 0.0008}, "phase4": {"sizes": [15000000, 80000000, 300000000], "steps": 16}}),
        Experiment("phase1 512 plus phase2 penalty 0.5", {"phase1": {"projections": 512}, "phase2": {"direct_penalty": 0.5}}),
        Experiment("phase3 horizon 5 plus deep search", {"phase3": {"horizon": 5, "num_samples": 96, "num_elites": 16, "num_iterations": 7}}),
        Experiment("phase1 long phase2 long phase4 long", {"phase1": {"steps": 40}, "phase2": {"epochs": 120}, "phase4": {"steps": 20}}),
        Experiment("phase1 focused phase2 threshold 0.15 phase3 deep", {"phase1": {"lambdas": [0.05, 0.1, 0.2], "steps": 36}, "phase2": {"answer_threshold": 0.15}, "phase3": {"num_samples": 96, "num_elites": 16, "num_iterations": 7}}),
        Experiment("phase1 256 phase2 penalty 0.35 phase4 medium", {"phase1": {"projections": 256}, "phase2": {"direct_penalty": 0.35}, "phase4": {"sizes": [15000000, 80000000, 300000000], "steps": 16}}),
        Experiment("phase1 512 phase2 epochs 120 phase3 deep", {"phase1": {"projections": 512}, "phase2": {"epochs": 120}, "phase3": {"num_samples": 96, "num_elites": 16, "num_iterations": 7}}),
    ]


def run_experiment(base_config: dict[str, Any], exp: Experiment, run_index: int) -> tuple[str, dict[str, Any] | None, int]:
    run_id = f"{run_index:03d}-{slugify(exp.name)}"
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    config = nested_update(copy.deepcopy(base_config), exp.updates)
    config_path = run_dir / "config.yaml"
    output_path = run_dir / "results.json"
    report_path = run_dir / "REPORT.md"
    log_path = run_dir / "run.log"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    cmd = [
        str(PYTHON),
        str(VALIDATOR),
        "--config",
        str(config_path),
        "--output",
        str(output_path),
        "--report",
        str(report_path),
    ]
    with log_path.open("w") as log_handle:
        proc = subprocess.run(cmd, cwd=ROOT, stdout=log_handle, stderr=subprocess.STDOUT, text=True)

    if proc.returncode != 0 or not output_path.exists():
        return run_id, None, proc.returncode
    return run_id, json.loads(output_path.read_text()), proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-hours", type=float, default=4.0)
    parser.add_argument("--max-experiments", type=int, default=999)
    args = parser.parse_args()

    ensure_results_header()
    base_config = yaml.safe_load(BASE_CONFIG.read_text())
    queue = candidate_queue()
    deadline = time.time() + args.max_hours * 3600.0

    with AUTORUN_LOG.open("a") as log_handle:
        log_handle.write(f"start {time.strftime('%Y-%m-%d %H:%M:%S')} max_hours={args.max_hours}\n")
        for run_index, exp in enumerate(queue[: args.max_experiments], start=1):
            if time.time() >= deadline:
                log_handle.write("deadline reached\n")
                break
            log_handle.write(f"run {run_index}: {exp.name}\n")
            log_handle.flush()
            run_id, results, returncode = run_experiment(base_config, exp, run_index)
            status = "ok" if results is not None else f"failed({returncode})"
            append_result(run_id, status, results, exp.name)
            log_handle.write(f"completed {run_id} status={status}\n")
            log_handle.flush()
        log_handle.write(f"end {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())