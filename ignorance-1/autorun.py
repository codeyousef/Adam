#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
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
ROBUSTNESS_SEEDS = [42, 43, 44, 45]
FINAL_CONFIRMATION_SEEDS = [42, 43, 44, 45, 46, 47, 48, 49]
FINAL_SPLIT_STRESS_SEEDS = [42, 43, 44, 45]
PRODUCTION_STRESS_SEEDS = [43, 47]
PRODUCTION_CONFIRMATION_SEEDS = [42, 43, 44, 45, 46, 47, 48, 49]
TRANSIENT_CUDA_ERROR_MARKERS = (
    "cuda error: unspecified launch failure",
    "cudaerrorlaunchfailure",
    "torch.acceleratorerror",
)


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


def safe_float(value: str | None, default: float = 0.0) -> float:
    try:
        return float(value or default)
    except (TypeError, ValueError):
        return default


def safe_int(value: str | None, default: int = 0) -> int:
    try:
        return int(value or default)
    except (TypeError, ValueError):
        return default


def parse_results_table() -> list[dict[str, str]]:
    if not RESULTS_TSV.exists():
        return []

    rows: list[dict[str, str]] = []
    with RESULTS_TSV.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            cleaned = {
                (key or "").strip(): (value or "").strip()
                for key, value in row.items()
            }
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


def load_run_config(run_id: str) -> dict[str, Any] | None:
    config_path = RUNS_DIR / run_id / "config.yaml"
    if not config_path.exists():
        return None
    return yaml.safe_load(config_path.read_text())


def load_run_results(run_id: str) -> dict[str, Any] | None:
    results_path = RUNS_DIR / run_id / "results.json"
    if not results_path.exists():
        return None
    return json.loads(results_path.read_text())


PHASE4_SIGNATURE_DEFAULTS: dict[str, Any] = {
    "sizes": [],
    "steps": 0,
    "batch_size": 0,
    "lr": 0.0,
    "num_splits": 3,
    "phase4_dataset": "benchmark_v1",
    "phase4_balance_families": False,
    "phase4_factorized_hard_negatives": False,
    "phase4_ood_mode": "default",
    "phase4_prompt_template": "default",
    "proxy_recipe": "v4",
    "reference_size": None,
    "step_scale_power": 0.0,
    "max_step_multiplier": 1.0,
    "lr_scale_power": 0.0,
    "max_lr_divisor": 1.0,
    "scheduler": "constant",
    "warmup_fraction": 0.0,
    "min_lr_ratio": 1.0,
    "grad_accum_steps": 1,
    "ema_target_decay": 0.0,
    "proxy_disable_batchnorm": False,
    "common_random_numbers": False,
    "split_seed_stride": 1000,
    "data_seed_offset": 0,
    "init_seed_offset": 100_000,
    "train_seed_offset": 200_000,
    "validation_eval_mode": True,
    "ignorance_ood_weight": 0.2,
    "ignorance_pred_weight": 0.2,
    "classifier_weight": 0.25,
    "alignment_prediction_weight": 1.0,
    "alignment_embedding_weight": 0.5,
    "alignment_mse_weight": 0.25,
    "ranking_margin_weight": 0.0,
    "ranking_margin": 0.2,
    "ranking_focal_gamma": 0.0,
    "ranking_start_fraction": 0.0,
    "ranking_ramp_fraction": 0.0,
    "ranking_largest_only": False,
    "use_retrieval_head": False,
    "retrieval_head_dim": 0,
    "retrieval_head_hidden_dim": 0,
    "use_retrieval_facets": False,
    "retrieval_num_facets": 0,
    "retrieval_facet_dim": 0,
    "retrieval_facet_hidden_dim": 0,
    "retrieval_facet_separate_query_code": False,
    "retrieval_facet_score_mode": "hard_maxsim",
    "retrieval_facet_softmax_temperature": 0.1,
    "retrieval_facet_loss_weight": 0.0,
    "phase4_joint_training": False,
    "champion_challenger_weight": 0.0,
    "champion_challenger_margin": 0.05,
    "champion_challenger_temperature": 0.1,
    "champion_challenger_start_fraction": 0.0,
    "champion_challenger_ramp_fraction": 0.0,
}


def phase4_signature(phase4_updates: dict[str, Any]) -> str:
    normalized = copy.deepcopy(PHASE4_SIGNATURE_DEFAULTS)
    for key, value in phase4_updates.items():
        if key in normalized:
            normalized[key] = value
    normalized["sizes"] = [int(size) for size in normalized["sizes"]]
    return json.dumps(normalized, sort_keys=True)


def existing_phase4_signatures_by_seed(history_rows: list[dict[str, str]]) -> set[tuple[int, str]]:
    signatures: set[tuple[int, str]] = set()
    for row in history_rows:
        if row.get("status") != "ok":
            continue
        run_id = row.get("run_id", "")
        if not run_id:
            continue
        results = load_run_results(run_id)
        if results is None:
            continue
        phase4_result = results.get("phase4", {})
        if phase4_result.get("validation_eval_mode") is not True:
            continue
        config = load_run_config(run_id)
        if config is None:
            continue
        seed = config.get("seed")
        phase4_config = config.get("phase4")
        if seed is None or phase4_config is None:
            continue
        signatures.add((int(seed), phase4_signature(phase4_config)))
    return signatures


def dedupe_sorted(values: list[float]) -> list[float]:
    deduped: list[float] = []
    for value in sorted(values):
        if not deduped or abs(deduped[-1] - value) > 1e-9:
            deduped.append(round(value, 6))
    return deduped


def broaden_lambdas(current: list[float]) -> list[float]:
    expanded = list(current)
    if current:
        lowest = min(current)
        highest = max(current)
        expanded.extend([max(lowest / 2.0, 0.005), min(highest * 1.5, 0.4)])
    expanded.extend([0.005, 0.02, 0.3])
    return dedupe_sorted(expanded)


def widen_sizes(current: list[int]) -> list[int]:
    expanded = list(current)
    if current:
        expanded.append(max(current) * 2)
    expanded.extend([1_200_000_000])
    return sorted(set(expanded))


def compatible_phase1_embed_dim(target: int) -> int:
    aligned = ((target + 5) // 6) * 6
    return max(192, aligned)


def compatible_head_count(embed_dim: int, preferred: int) -> int:
    for candidate in [preferred, 12, 10, 8, 6, 4, 3, 2, 1]:
        if candidate > 0 and embed_dim % candidate == 0:
            return candidate
    return 1


def adaptive_depth(description: str) -> int:
    return description.count("adaptive ")


def robust_family_name(description: str) -> str | None:
    parts = description.rsplit(" seed", 1)
    if len(parts) != 2:
        return None
    suffix = parts[1].strip()
    if suffix.isdigit():
        return parts[0].strip()
    return None


def robust_seed_from_description(description: str) -> int | None:
    parts = description.rsplit(" seed", 1)
    if len(parts) != 2:
        return None
    suffix = parts[1].strip()
    return int(suffix) if suffix.isdigit() else None


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


def is_transient_cuda_failure(log_path: Path) -> bool:
    if not log_path.exists():
        return False
    log_text = log_path.read_text(encoding="utf-8", errors="ignore").lower()
    return any(marker in log_text for marker in TRANSIENT_CUDA_ERROR_MARKERS)


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


def adaptive_experiments_for_row(row: dict[str, str], config: dict[str, Any]) -> list[Experiment]:
    phase1 = config["phase1"]
    phase2 = config["phase2"]
    phase3 = config["phase3"]
    phase4 = config["phase4"]
    description = row["description"]
    stronger_embed_dim = compatible_phase1_embed_dim(
        min(max(int(phase1["embed_dim"]) + 72, 288), 384)
    )
    stronger_encoder_heads = compatible_head_count(stronger_embed_dim, int(phase1.get("encoder_heads", 6)))
    stronger_predictor_heads = compatible_head_count(stronger_embed_dim, int(phase1.get("predictor_heads", 8)))

    stronger_phase1 = {
        "phase1": {
            "embed_dim": stronger_embed_dim,
            "encoder_layers": min(max(int(phase1.get("encoder_layers", 4)) + 2, 6), 10),
            "encoder_heads": stronger_encoder_heads,
            "predictor_layers": min(max(int(phase1.get("predictor_layers", 4)) + 2, 6), 12),
            "predictor_heads": stronger_predictor_heads,
            "projections": min(max(int(phase1["projections"]) * 2, 256), 1024),
            "steps": min(max(int(phase1["steps"]) + 40, 64), 160),
            "lambdas": broaden_lambdas(list(phase1["lambdas"])),
            "lr": min(float(phase1["lr"]) * 1.35, 0.0008),
        }
    }
    deeper_phase3 = {
        "phase3": {
            "horizon": min(int(phase3["horizon"]) + 1, 6),
            "num_samples": min(max(int(phase3["num_samples"]) + 24, 72), 128),
            "num_elites": min(max(int(phase3["num_elites"]) + 4, 12), 24),
            "num_iterations": min(max(int(phase3["num_iterations"]) + 2, 7), 10),
        }
    }
    stronger_phase4 = {
        "phase4": {
            "sizes": widen_sizes(list(phase4["sizes"])),
            "steps": min(max(int(phase4["steps"]) + 16, 28), 56),
            "batch_size": min(max(int(phase4["batch_size"]) + 2, 6), 8),
            "lr": min(float(phase4["lr"]) * 1.2, 0.0006),
        }
    }

    return [
        Experiment(f"adaptive {description} phase1 rescue", stronger_phase1),
        Experiment(
            f"adaptive {description} phase1+phase4 rescue",
            nested_update(copy.deepcopy(stronger_phase1), copy.deepcopy(stronger_phase4)),
        ),
        Experiment(
            f"adaptive {description} phase1+phase3 refinement",
            nested_update(copy.deepcopy(stronger_phase1), copy.deepcopy(deeper_phase3)),
        ),
        Experiment(
            f"adaptive {description} full refinement",
            nested_update(
                nested_update(copy.deepcopy(stronger_phase1), copy.deepcopy(deeper_phase3)),
                copy.deepcopy(stronger_phase4),
            ),
        ),
        Experiment(
            f"adaptive {description} retrieval tighten",
            {
                "phase2": {
                    "epochs": min(max(int(phase2["epochs"]) + 40, 120), 180),
                    "answer_threshold": max(float(phase2["answer_threshold"]) - 0.05, 0.1),
                    "direct_penalty": min(float(phase2["direct_penalty"]) + 0.1, 0.6),
                },
                "phase3": copy.deepcopy(deeper_phase3["phase3"]),
            },
        ),
    ]


def global_adaptive_experiments() -> list[Experiment]:
    return [
        Experiment(
            "adaptive v2 global phase1 architecture sweep",
            {
                "phase1": {
                    "embed_dim": compatible_phase1_embed_dim(336),
                    "encoder_layers": 8,
                    "encoder_heads": compatible_head_count(compatible_phase1_embed_dim(336), 8),
                    "predictor_layers": 10,
                    "predictor_heads": compatible_head_count(compatible_phase1_embed_dim(336), 12),
                    "projections": 2048,
                    "steps": 160,
                    "lambdas": [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4],
                    "lr": 0.0006,
                }
            },
        ),
        Experiment(
            "adaptive v2 global phase1 phase4 ladder",
            {
                "phase1": {
                    "embed_dim": compatible_phase1_embed_dim(336),
                    "encoder_layers": 8,
                    "encoder_heads": compatible_head_count(compatible_phase1_embed_dim(336), 8),
                    "predictor_layers": 10,
                    "predictor_heads": compatible_head_count(compatible_phase1_embed_dim(336), 12),
                    "projections": 2048,
                    "steps": 160,
                    "lambdas": [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3],
                    "lr": 0.0006,
                },
                "phase4": {
                    "sizes": [15_000_000, 80_000_000, 300_000_000, 600_000_000, 1_200_000_000],
                    "steps": 56,
                    "batch_size": 6,
                    "lr": 0.0004,
                },
            },
        ),
        Experiment(
            "adaptive v2 global phase4 scaling ladder",
            {
                "phase4": {
                    "sizes": [15_000_000, 80_000_000, 300_000_000, 600_000_000, 1_200_000_000],
                    "steps": 56,
                    "batch_size": 6,
                    "lr": 0.0004,
                }
            },
        ),
        Experiment(
            "adaptive v2 global phase1 max isotropy",
            {
                "phase1": {
                    "embed_dim": 384,
                    "encoder_layers": 10,
                    "encoder_heads": 12,
                    "predictor_layers": 12,
                    "predictor_heads": 12,
                    "projections": 2048,
                    "steps": 160,
                    "lambdas": [0.0025, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4],
                    "lr": 0.0007,
                }
            },
        ),
        Experiment(
            "adaptive v2 global full readiness probe",
            {
                "phase1": {
                    "embed_dim": 384,
                    "encoder_layers": 10,
                    "encoder_heads": 12,
                    "predictor_layers": 12,
                    "predictor_heads": 12,
                    "projections": 2048,
                    "steps": 160,
                    "lambdas": [0.0025, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4],
                    "lr": 0.0007,
                },
                "phase2": {
                    "epochs": 160,
                    "answer_threshold": 0.15,
                    "direct_penalty": 0.35,
                },
                "phase3": {
                    "horizon": 5,
                    "num_samples": 96,
                    "num_elites": 16,
                    "num_iterations": 8,
                },
                "phase4": {
                    "sizes": [15_000_000, 80_000_000, 300_000_000, 600_000_000, 1_200_000_000],
                    "steps": 56,
                    "batch_size": 6,
                    "lr": 0.0004,
                },
            },
        ),
        Experiment(
            "adaptive v3 global phase1 large batch isotropy",
            {
                "phase1": {
                    "embed_dim": 384,
                    "encoder_layers": 10,
                    "encoder_heads": 12,
                    "predictor_layers": 12,
                    "predictor_heads": 12,
                    "projections": 4096,
                    "batch_size": 32,
                    "steps": 192,
                    "lambdas": [0.001, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2],
                    "lr": 0.0005,
                }
            },
        ),
        Experiment(
            "adaptive v3 global phase4 long scaling ladder",
            {
                "phase4": {
                    "sizes": [15_000_000, 80_000_000, 300_000_000, 600_000_000, 1_200_000_000],
                    "steps": 96,
                    "batch_size": 8,
                    "lr": 0.00025,
                }
            },
        ),
        Experiment(
            "adaptive v3 global long readiness probe",
            {
                "phase1": {
                    "embed_dim": 384,
                    "encoder_layers": 10,
                    "encoder_heads": 12,
                    "predictor_layers": 12,
                    "predictor_heads": 12,
                    "projections": 4096,
                    "batch_size": 32,
                    "steps": 192,
                    "lambdas": [0.001, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2],
                    "lr": 0.0005,
                },
                "phase2": {
                    "epochs": 160,
                    "answer_threshold": 0.15,
                    "direct_penalty": 0.35,
                },
                "phase3": {
                    "horizon": 5,
                    "num_samples": 96,
                    "num_elites": 16,
                    "num_iterations": 8,
                },
                "phase4": {
                    "sizes": [15_000_000, 80_000_000, 300_000_000, 600_000_000, 1_200_000_000],
                    "steps": 96,
                    "batch_size": 8,
                    "lr": 0.00025,
                },
            },
        ),
    ]


def phase4_robustness_families() -> list[Experiment]:
    return [
        Experiment(
            "phase4 robustness v4 family lower lr full ladder",
            {
                "profile": "robustness",
                "phase4": {
                    "sizes": [15_000_000, 80_000_000, 300_000_000, 600_000_000, 1_200_000_000],
                    "steps": 64,
                    "batch_size": 4,
                    "lr": 0.00008,
                    "num_splits": 5,
                },
            },
        ),
        Experiment(
            "phase4 robustness v4 family lower lr medium long",
            {
                "profile": "robustness",
                "phase4": {
                    "sizes": [15_000_000, 80_000_000, 300_000_000, 600_000_000, 1_200_000_000],
                    "steps": 96,
                    "batch_size": 4,
                    "lr": 0.00008,
                    "num_splits": 5,
                },
            },
        ),
        Experiment(
            "phase4 robustness v4 family very low lr full ladder",
            {
                "profile": "robustness",
                "phase4": {
                    "sizes": [15_000_000, 80_000_000, 300_000_000, 600_000_000, 1_200_000_000],
                    "steps": 80,
                    "batch_size": 4,
                    "lr": 0.00006,
                    "num_splits": 5,
                },
            },
        ),
        Experiment(
            "phase4 robustness v4 family very low lr medium long",
            {
                "profile": "robustness",
                "phase4": {
                    "sizes": [15_000_000, 80_000_000, 300_000_000, 600_000_000, 1_200_000_000],
                    "steps": 112,
                    "batch_size": 4,
                    "lr": 0.00006,
                    "num_splits": 5,
                },
            },
        ),
        Experiment(
            "phase4 robustness v4 family low lr dense ladder",
            {
                "profile": "robustness",
                "phase4": {
                    "sizes": [15_000_000, 40_000_000, 80_000_000, 150_000_000, 300_000_000, 600_000_000, 1_200_000_000],
                    "steps": 96,
                    "batch_size": 4,
                    "lr": 0.00008,
                    "num_splits": 5,
                },
            },
        ),
        Experiment(
            "phase4 robustness v4 family ultra low lr dense ladder",
            {
                "profile": "robustness",
                "phase4": {
                    "sizes": [15_000_000, 40_000_000, 80_000_000, 150_000_000, 300_000_000, 600_000_000, 1_200_000_000],
                    "steps": 128,
                    "batch_size": 4,
                    "lr": 0.00005,
                    "num_splits": 5,
                },
            },
        ),
        Experiment(
            "phase4 robustness v4 family low lr larger batch",
            {
                "profile": "robustness",
                "phase4": {
                    "sizes": [15_000_000, 80_000_000, 300_000_000, 600_000_000, 1_200_000_000],
                    "steps": 96,
                    "batch_size": 6,
                    "lr": 0.00008,
                    "num_splits": 5,
                },
            },
        ),
        Experiment(
            "phase4 robustness v4 family very low lr larger batch",
            {
                "profile": "robustness",
                "phase4": {
                    "sizes": [15_000_000, 80_000_000, 300_000_000, 600_000_000, 1_200_000_000],
                    "steps": 128,
                    "batch_size": 6,
                    "lr": 0.00006,
                    "num_splits": 5,
                },
            },
        ),
        Experiment(
            "phase4 robustness v4 family very low lr heavy ladder",
            {
                "profile": "robustness",
                "phase4": {
                    "sizes": [80_000_000, 150_000_000, 300_000_000, 600_000_000, 1_200_000_000],
                    "steps": 128,
                    "batch_size": 4,
                    "lr": 0.00005,
                    "num_splits": 5,
                },
            },
        ),
        Experiment(
            "phase4 robustness v4 family ultra low lr upper heavy",
            {
                "profile": "robustness",
                "phase4": {
                    "sizes": [80_000_000, 300_000_000, 600_000_000, 1_200_000_000],
                    "steps": 160,
                    "batch_size": 4,
                    "lr": 0.00004,
                    "num_splits": 5,
                },
            },
        ),
    ]


def phase4_v5_recovery_families() -> list[Experiment]:
    return [
        Experiment(
            "phase4 robustness v5 family full distinct ladder",
            {
                "profile": "robustness",
                "phase4": {
                    "sizes": [15_000_000, 40_000_000, 80_000_000, 150_000_000, 300_000_000, 600_000_000, 1_200_000_000],
                    "steps": 160,
                    "batch_size": 4,
                    "lr": 0.00005,
                    "num_splits": 5,
                    "proxy_recipe": "v5_distinct",
                },
            },
        ),
        Experiment(
            "phase4 robustness v5 family upper heavy distinct",
            {
                "profile": "robustness",
                "phase4": {
                    "sizes": [80_000_000, 150_000_000, 300_000_000, 600_000_000, 1_200_000_000],
                    "steps": 160,
                    "batch_size": 4,
                    "lr": 0.00005,
                    "num_splits": 5,
                    "proxy_recipe": "v5_distinct",
                },
            },
        ),
        Experiment(
            "phase4 robustness v5 family upper heavy longer",
            {
                "profile": "robustness",
                "phase4": {
                    "sizes": [80_000_000, 150_000_000, 300_000_000, 600_000_000, 1_200_000_000],
                    "steps": 224,
                    "batch_size": 4,
                    "lr": 0.00004,
                    "num_splits": 5,
                    "proxy_recipe": "v5_distinct",
                },
            },
        ),
        Experiment(
            "phase4 robustness v5 family upper heavy larger batch",
            {
                "profile": "robustness",
                "phase4": {
                    "sizes": [80_000_000, 150_000_000, 300_000_000, 600_000_000, 1_200_000_000],
                    "steps": 192,
                    "batch_size": 6,
                    "lr": 0.00005,
                    "num_splits": 5,
                    "proxy_recipe": "v5_distinct",
                },
            },
        ),
        Experiment(
            "phase4 robustness v5 family upper only long",
            {
                "profile": "robustness",
                "phase4": {
                    "sizes": [150_000_000, 300_000_000, 600_000_000, 1_200_000_000],
                    "steps": 224,
                    "batch_size": 4,
                    "lr": 0.00004,
                    "num_splits": 5,
                    "proxy_recipe": "v5_distinct",
                },
            },
        ),
        Experiment(
            "phase4 robustness v5 family upper only ultra low lr",
            {
                "profile": "robustness",
                "phase4": {
                    "sizes": [150_000_000, 300_000_000, 600_000_000, 1_200_000_000],
                    "steps": 256,
                    "batch_size": 4,
                    "lr": 0.00003,
                    "num_splits": 5,
                    "proxy_recipe": "v5_distinct",
                },
            },
        ),
    ]


def phase4_v6_breadth_families() -> list[Experiment]:
    return [
        Experiment(
            "phase4 robustness v6 family full distinct step scaled",
            {
                "profile": "robustness",
                "phase4": {
                    "sizes": [15_000_000, 40_000_000, 80_000_000, 150_000_000, 300_000_000, 600_000_000, 1_200_000_000],
                    "steps": 96,
                    "batch_size": 4,
                    "lr": 0.00005,
                    "num_splits": 5,
                    "proxy_recipe": "v5_distinct",
                    "step_scale_power": 0.5,
                    "max_step_multiplier": 4.0,
                },
            },
        ),
        Experiment(
            "phase4 robustness v6 family full distinct step and lr scaled",
            {
                "profile": "robustness",
                "phase4": {
                    "sizes": [15_000_000, 40_000_000, 80_000_000, 150_000_000, 300_000_000, 600_000_000, 1_200_000_000],
                    "steps": 96,
                    "batch_size": 4,
                    "lr": 0.00006,
                    "num_splits": 5,
                    "proxy_recipe": "v5_distinct",
                    "step_scale_power": 0.5,
                    "max_step_multiplier": 4.0,
                    "lr_scale_power": 0.3,
                    "max_lr_divisor": 3.0,
                },
            },
        ),
        Experiment(
            "phase4 robustness v6 family upper heavy step scaled",
            {
                "profile": "robustness",
                "phase4": {
                    "sizes": [80_000_000, 150_000_000, 300_000_000, 600_000_000, 1_200_000_000],
                    "steps": 128,
                    "batch_size": 4,
                    "lr": 0.00005,
                    "num_splits": 5,
                    "proxy_recipe": "v5_distinct",
                    "step_scale_power": 0.35,
                    "max_step_multiplier": 3.0,
                },
            },
        ),
        Experiment(
            "phase4 robustness v6 family upper heavy step and lr scaled",
            {
                "profile": "robustness",
                "phase4": {
                    "sizes": [80_000_000, 150_000_000, 300_000_000, 600_000_000, 1_200_000_000],
                    "steps": 128,
                    "batch_size": 4,
                    "lr": 0.00006,
                    "num_splits": 5,
                    "proxy_recipe": "v5_distinct",
                    "step_scale_power": 0.35,
                    "max_step_multiplier": 3.0,
                    "lr_scale_power": 0.35,
                    "max_lr_divisor": 3.0,
                },
            },
        ),
        Experiment(
            "phase4 robustness v6 family upper only long high splits",
            {
                "profile": "robustness",
                "phase4": {
                    "sizes": [150_000_000, 300_000_000, 600_000_000, 1_200_000_000],
                    "steps": 160,
                    "batch_size": 4,
                    "lr": 0.00004,
                    "num_splits": 7,
                    "proxy_recipe": "v5_distinct",
                    "step_scale_power": 0.35,
                    "max_step_multiplier": 3.0,
                    "lr_scale_power": 0.3,
                    "max_lr_divisor": 3.0,
                },
            },
        ),
        Experiment(
            "phase4 robustness v6 family top triad ultra low lr",
            {
                "profile": "robustness",
                "phase4": {
                    "sizes": [300_000_000, 600_000_000, 1_200_000_000],
                    "steps": 192,
                    "batch_size": 4,
                    "lr": 0.00003,
                    "num_splits": 7,
                    "proxy_recipe": "v5_distinct",
                    "step_scale_power": 0.5,
                    "max_step_multiplier": 4.0,
                    "lr_scale_power": 0.4,
                    "max_lr_divisor": 4.0,
                },
            },
        ),
    ]


def final_confirmation_families() -> list[Experiment]:
    return [
        Experiment(
            "final confirmation family working recipe",
            {
                "profile": "robustness",
                "phase4": {
                    "sizes": [15_000_000, 40_000_000, 80_000_000, 150_000_000, 300_000_000, 600_000_000, 1_200_000_000],
                    "steps": 96,
                    "batch_size": 4,
                    "lr": 0.00005,
                    "num_splits": 5,
                    "proxy_recipe": "v5_distinct",
                    "step_scale_power": 0.5,
                    "max_step_multiplier": 4.0,
                },
            },
        ),
        Experiment(
            "final confirmation family working recipe high splits",
            {
                "profile": "robustness",
                "phase4": {
                    "sizes": [15_000_000, 40_000_000, 80_000_000, 150_000_000, 300_000_000, 600_000_000, 1_200_000_000],
                    "steps": 96,
                    "batch_size": 4,
                    "lr": 0.00005,
                    "num_splits": 7,
                    "proxy_recipe": "v5_distinct",
                    "step_scale_power": 0.5,
                    "max_step_multiplier": 4.0,
                },
            },
        ),
    ]


def production_readiness_families() -> list[Experiment]:
    return [
        Experiment(
            "production readiness family long high splits",
            {
                "profile": "robustness",
                "phase4": {
                    "sizes": [15_000_000, 40_000_000, 80_000_000, 150_000_000, 300_000_000, 600_000_000, 1_200_000_000],
                    "steps": 112,
                    "batch_size": 4,
                    "lr": 0.00005,
                    "num_splits": 7,
                    "proxy_recipe": "v5_distinct",
                    "step_scale_power": 0.55,
                    "max_step_multiplier": 5.0,
                },
            },
        ),
        Experiment(
            "production readiness family long high splits lr scaled",
            {
                "profile": "robustness",
                "phase4": {
                    "sizes": [15_000_000, 40_000_000, 80_000_000, 150_000_000, 300_000_000, 600_000_000, 1_200_000_000],
                    "steps": 112,
                    "batch_size": 4,
                    "lr": 0.00005,
                    "num_splits": 7,
                    "proxy_recipe": "v5_distinct",
                    "step_scale_power": 0.55,
                    "max_step_multiplier": 5.0,
                    "lr_scale_power": 0.2,
                    "max_lr_divisor": 2.5,
                },
            },
        ),
        Experiment(
            "production readiness family extra high splits",
            {
                "profile": "robustness",
                "phase4": {
                    "sizes": [15_000_000, 40_000_000, 80_000_000, 150_000_000, 300_000_000, 600_000_000, 1_200_000_000],
                    "steps": 112,
                    "batch_size": 4,
                    "lr": 0.00005,
                    "num_splits": 9,
                    "proxy_recipe": "v5_distinct",
                    "step_scale_power": 0.5,
                    "max_step_multiplier": 4.0,
                },
            },
        ),
        Experiment(
            "production readiness family extra high splits lr scaled",
            {
                "profile": "robustness",
                "phase4": {
                    "sizes": [15_000_000, 40_000_000, 80_000_000, 150_000_000, 300_000_000, 600_000_000, 1_200_000_000],
                    "steps": 128,
                    "batch_size": 4,
                    "lr": 0.00005,
                    "num_splits": 9,
                    "proxy_recipe": "v5_distinct",
                    "step_scale_power": 0.6,
                    "max_step_multiplier": 5.0,
                    "lr_scale_power": 0.25,
                    "max_lr_divisor": 3.0,
                },
            },
        ),
    ]


def robustness_queue(
    history_rows: list[dict[str, str]],
    families: list[Experiment] | None = None,
    seeds: list[int] | None = None,
) -> list[Experiment]:
    if families is None:
        families = phase4_robustness_families()
    if seeds is None:
        seeds = ROBUSTNESS_SEEDS
    successful_descriptions = {
        row.get("description", "")
        for row in history_rows
        if row.get("status") == "ok"
    }
    family_rows: dict[str, list[dict[str, str]]] = {}
    for row in history_rows:
        description = row.get("description", "")
        family = robust_family_name(description)
        if family is None:
            continue
        family_rows.setdefault(family, []).append(row)

    passing_family_found = False
    for family, rows in family_rows.items():
        seen_seeds = {robust_seed_from_description(row.get("description", "")) for row in rows}
        if None in seen_seeds:
            seen_seeds.discard(None)
        all_done = all(seed in seen_seeds for seed in seeds)
        all_pass = all(row.get("phase4_pass") == "1" for row in rows if row.get("status") == "ok")
        if all_done and all_pass and len(rows) >= len(seeds):
            passing_family_found = True
            break

    if passing_family_found:
        return []

    queue: list[Experiment] = []
    completed_signatures = existing_phase4_signatures_by_seed(history_rows)
    scheduled_signatures: set[tuple[int, str]] = set()
    for family_exp in families:
        family_name = family_exp.name
        family_signature = phase4_signature(family_exp.updates.get("phase4", {}))
        existing = family_rows.get(family_name, [])
        existing_seeds = {
            robust_seed_from_description(row.get("description", ""))
            for row in existing
            if row.get("status") == "ok"
        }
        existing_seeds.discard(None)
        for seed in seeds:
            description = f"{family_name} seed{seed}"
            if description in successful_descriptions:
                continue
            signature_key = (seed, family_signature)
            if signature_key in completed_signatures or signature_key in scheduled_signatures:
                continue
            updates = copy.deepcopy(family_exp.updates)
            updates["seed"] = seed
            queue.append(Experiment(description, updates))
            scheduled_signatures.add(signature_key)
        if queue:
            return queue
    return queue


def final_confirmation_queue(history_rows: list[dict[str, str]]) -> list[Experiment]:
    family_seeds: dict[str, list[int]] = {
        "final confirmation family working recipe": FINAL_CONFIRMATION_SEEDS,
        "final confirmation family working recipe high splits": FINAL_SPLIT_STRESS_SEEDS,
    }
    families = final_confirmation_families()
    successful_descriptions = {
        row.get("description", "")
        for row in history_rows
        if row.get("status") == "ok"
    }

    queue: list[Experiment] = []
    completed_signatures = existing_phase4_signatures_by_seed(history_rows)
    scheduled_signatures: set[tuple[int, str]] = set()
    for family in families:
        family_signature = phase4_signature(family.updates.get("phase4", {}))
        seeds = family_seeds[family.name]
        for seed in seeds:
            description = f"{family.name} seed{seed}"
            if description in successful_descriptions:
                continue
            signature_key = (seed, family_signature)
            if signature_key in completed_signatures or signature_key in scheduled_signatures:
                continue
            updates = copy.deepcopy(family.updates)
            updates["seed"] = seed
            queue.append(Experiment(description, updates))
            scheduled_signatures.add(signature_key)
    return queue


def production_readiness_queue(history_rows: list[dict[str, str]]) -> list[Experiment]:
    families = production_readiness_families()
    rows_by_description = {
        row.get("description", ""): row
        for row in history_rows
        if row.get("status") == "ok"
    }
    completed_signatures = existing_phase4_signatures_by_seed(history_rows)

    for family in families:
        family_signature = phase4_signature(family.updates.get("phase4", {}))
        stress_missing: list[Experiment] = []
        stress_rows: list[dict[str, str]] = []
        for seed in PRODUCTION_STRESS_SEEDS:
            description = f"{family.name} seed{seed}"
            row = rows_by_description.get(description)
            if row is None:
                if (seed, family_signature) in completed_signatures:
                    continue
                updates = copy.deepcopy(family.updates)
                updates["seed"] = seed
                stress_missing.append(Experiment(description, updates))
            else:
                stress_rows.append(row)
        if stress_missing:
            return stress_missing

        stress_pass = all(
            row.get("phase1_pass") == "1"
            and row.get("phase2_pass") == "1"
            and row.get("phase3_pass") == "1"
            and row.get("phase4_pass") == "1"
            for row in stress_rows
        )
        if not stress_pass:
            continue

        confirmation_missing: list[Experiment] = []
        confirmation_rows: list[dict[str, str]] = []
        for seed in PRODUCTION_CONFIRMATION_SEEDS:
            description = f"{family.name} seed{seed}"
            row = rows_by_description.get(description)
            if row is None:
                if (seed, family_signature) in completed_signatures:
                    continue
                updates = copy.deepcopy(family.updates)
                updates["seed"] = seed
                confirmation_missing.append(Experiment(description, updates))
            else:
                confirmation_rows.append(row)
        if confirmation_missing:
            return confirmation_missing

        confirmation_pass = all(
            row.get("phase1_pass") == "1"
            and row.get("phase2_pass") == "1"
            and row.get("phase3_pass") == "1"
            and row.get("phase4_pass") == "1"
            for row in confirmation_rows
        )
        if confirmation_pass and len(confirmation_rows) >= len(PRODUCTION_CONFIRMATION_SEEDS):
            return []

    return []


def production_readiness_full_queue(history_rows: list[dict[str, str]]) -> list[Experiment]:
    families = production_readiness_families()
    rows_by_description = {
        row.get("description", ""): row
        for row in history_rows
        if row.get("status") == "ok"
    }
    completed_signatures = existing_phase4_signatures_by_seed(history_rows)
    scheduled_signatures: set[tuple[int, str]] = set()

    queue: list[Experiment] = []
    for family in families:
        family_signature = phase4_signature(family.updates.get("phase4", {}))
        stress_missing: list[Experiment] = []
        stress_rows: list[dict[str, str]] = []
        for seed in PRODUCTION_STRESS_SEEDS:
            description = f"{family.name} seed{seed}"
            row = rows_by_description.get(description)
            if row is None:
                signature_key = (seed, family_signature)
                if signature_key in completed_signatures or signature_key in scheduled_signatures:
                    continue
                updates = copy.deepcopy(family.updates)
                updates["seed"] = seed
                stress_missing.append(Experiment(description, updates))
                scheduled_signatures.add(signature_key)
            else:
                stress_rows.append(row)
        if stress_missing:
            queue.extend(stress_missing)
            continue

        stress_pass = all(
            row.get("phase1_pass") == "1"
            and row.get("phase2_pass") == "1"
            and row.get("phase3_pass") == "1"
            and row.get("phase4_pass") == "1"
            for row in stress_rows
        )
        if not stress_pass:
            continue

        for seed in PRODUCTION_CONFIRMATION_SEEDS:
            description = f"{family.name} seed{seed}"
            if description in rows_by_description:
                continue
            signature_key = (seed, family_signature)
            if signature_key in completed_signatures or signature_key in scheduled_signatures:
                continue
            updates = copy.deepcopy(family.updates)
            updates["seed"] = seed
            queue.append(Experiment(description, updates))
            scheduled_signatures.add(signature_key)
    return queue


def adaptive_queue(history_rows: list[dict[str, str]], top_k: int) -> list[Experiment]:
    if not history_rows:
        return []

    successful_descriptions = {
        row.get("description", "")
        for row in history_rows
        if row.get("status") == "ok"
    }
    failure_counts: dict[str, int] = {}
    for row in history_rows:
        if row.get("status") == "ok":
            continue
        description = row.get("description", "")
        failure_counts[description] = failure_counts.get(description, 0) + 1

    ok_rows = [
        row
        for row in history_rows
        if row.get("status") == "ok" and adaptive_depth(row.get("description", "")) <= 1
    ]
    ranked_rows = sorted(ok_rows, key=lambda row: safe_float(row.get("phase_score")), reverse=True)
    queue: list[Experiment] = []

    for row in ranked_rows[:top_k]:
        config = load_run_config(row["run_id"])
        if config is None:
            continue
        for exp in adaptive_experiments_for_row(row, config):
            if exp.name in successful_descriptions:
                continue
            if failure_counts.get(exp.name, 0) >= 2:
                continue
            successful_descriptions.add(exp.name)
            queue.append(exp)

    for exp in global_adaptive_experiments():
        if exp.name in successful_descriptions:
            continue
        if failure_counts.get(exp.name, 0) >= 2:
            continue
        successful_descriptions.add(exp.name)
        queue.append(exp)

    return queue


def build_queue(strategy: str, history_rows: list[dict[str, str]], top_k: int) -> list[Experiment]:
    seed_queue = candidate_queue()
    if strategy == "seed":
        return seed_queue
    if strategy == "robustness":
        return robustness_queue(history_rows)
    if strategy == "robustness_v5":
        return robustness_queue(history_rows, phase4_v5_recovery_families())
    if strategy == "robustness_v6":
        return robustness_queue(history_rows, phase4_v6_breadth_families())
    if strategy == "final_confirmation":
        return final_confirmation_queue(history_rows)
    if strategy == "production_readiness":
        return production_readiness_queue(history_rows)
    if strategy == "production_readiness_full":
        return production_readiness_full_queue(history_rows)
    if strategy == "adaptive":
        if not history_rows:
            return seed_queue
        return adaptive_queue(history_rows, top_k)
    return seed_queue + adaptive_queue(history_rows, top_k)


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
    env = os.environ.copy()
    attempts = 2
    for attempt in range(1, attempts + 1):
        if output_path.exists():
            output_path.unlink()
        if report_path.exists():
            report_path.unlink()
        with log_path.open("a" if attempt > 1 else "w") as log_handle:
            if attempt > 1:
                log_handle.write(f"\n=== retry attempt {attempt} after transient CUDA failure ===\n")
            proc = subprocess.run(
                cmd,
                cwd=ROOT,
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )

        if proc.returncode == 0 and output_path.exists():
            return run_id, json.loads(output_path.read_text()), proc.returncode
        if attempt >= attempts or not is_transient_cuda_failure(log_path):
            return run_id, None, proc.returncode
        time.sleep(5)

    return run_id, None, 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-hours", type=float, default=4.0)
    parser.add_argument("--max-experiments", type=int, default=999)
    parser.add_argument("--strategy", choices=["seed", "adaptive", "hybrid", "robustness", "robustness_v5", "robustness_v6", "final_confirmation", "production_readiness", "production_readiness_full"], default="adaptive")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    ensure_results_header()
    history_rows = parse_results_table()
    queue = build_queue(args.strategy, history_rows, args.top_k)

    if args.dry_run:
        for index, exp in enumerate(queue[: args.max_experiments], start=next_run_index(history_rows)):
            print(f"{index:03d}\t{exp.name}\t{json.dumps(exp.updates, sort_keys=True)}")
        return 0

    base_config = yaml.safe_load(BASE_CONFIG.read_text())
    deadline = time.time() + args.max_hours * 3600.0
    start_index = next_run_index(history_rows)
    completed_runs = 0

    with AUTORUN_LOG.open("a") as log_handle:
        log_handle.write(
            f"start {time.strftime('%Y-%m-%d %H:%M:%S')} max_hours={args.max_hours} strategy={args.strategy}\n"
        )
        run_index = start_index
        while completed_runs < args.max_experiments:
            if time.time() >= deadline:
                log_handle.write("deadline reached\n")
                break
            history_rows = parse_results_table()
            queue = build_queue(args.strategy, history_rows, args.top_k)
            if not queue:
                log_handle.write("no pending experiments\n")
                break
            exp = queue[0]
            log_handle.write(f"run {run_index}: {exp.name}\n")
            log_handle.flush()
            run_id, results, returncode = run_experiment(base_config, exp, run_index)
            status = "ok" if results is not None else f"failed({returncode})"
            append_result(run_id, status, results, exp.name)
            log_handle.write(f"completed {run_id} status={status}\n")
            log_handle.flush()
            completed_runs += 1
            run_index += 1
        log_handle.write(f"end {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())