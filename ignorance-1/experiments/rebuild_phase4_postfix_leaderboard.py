from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.phase4 import run_phase4
from src.utils.config import load_config
from src.utils.data import set_seed


FAMILY_REGISTRY: dict[str, dict[str, Any]] = {
    "phase4 robustness v6 family full distinct step scaled": {
        "aliases": ["final confirmation family working recipe"],
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
    "production readiness family long high splits lr scaled": {
        "aliases": [],
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
    "postfix long high splits lr scaled softer step scale": {
        "aliases": [],
        "phase4": {
            "sizes": [15_000_000, 40_000_000, 80_000_000, 150_000_000, 300_000_000, 600_000_000, 1_200_000_000],
            "steps": 112,
            "batch_size": 4,
            "lr": 0.00005,
            "num_splits": 7,
            "proxy_recipe": "v5_distinct",
            "step_scale_power": 0.45,
            "max_step_multiplier": 5.0,
            "lr_scale_power": 0.2,
            "max_lr_divisor": 2.5,
        },
    },
    "postfix long high splits lr scaled softer step scale 0.40": {
        "aliases": [],
        "phase4": {
            "sizes": [15_000_000, 40_000_000, 80_000_000, 150_000_000, 300_000_000, 600_000_000, 1_200_000_000],
            "steps": 112,
            "batch_size": 4,
            "lr": 0.00005,
            "num_splits": 7,
            "proxy_recipe": "v5_distinct",
            "step_scale_power": 0.4,
            "max_step_multiplier": 5.0,
            "lr_scale_power": 0.2,
            "max_lr_divisor": 2.5,
        },
    },
    "postfix long high splits lr scaled softer step scale 0.425": {
        "aliases": [],
        "phase4": {
            "sizes": [15_000_000, 40_000_000, 80_000_000, 150_000_000, 300_000_000, 600_000_000, 1_200_000_000],
            "steps": 112,
            "batch_size": 4,
            "lr": 0.00005,
            "num_splits": 7,
            "proxy_recipe": "v5_distinct",
            "step_scale_power": 0.425,
            "max_step_multiplier": 5.0,
            "lr_scale_power": 0.2,
            "max_lr_divisor": 2.5,
        },
    },
    "postfix long high splits softer step weaker lr scaling": {
        "aliases": [],
        "phase4": {
            "sizes": [15_000_000, 40_000_000, 80_000_000, 150_000_000, 300_000_000, 600_000_000, 1_200_000_000],
            "steps": 112,
            "batch_size": 4,
            "lr": 0.00005,
            "num_splits": 7,
            "proxy_recipe": "v5_distinct",
            "step_scale_power": 0.45,
            "max_step_multiplier": 5.0,
            "lr_scale_power": 0.15,
            "max_lr_divisor": 2.0,
        },
    },
    "postfix long high splits softer step more top steps": {
        "aliases": [],
        "phase4": {
            "sizes": [15_000_000, 40_000_000, 80_000_000, 150_000_000, 300_000_000, 600_000_000, 1_200_000_000],
            "steps": 112,
            "batch_size": 4,
            "lr": 0.00005,
            "num_splits": 7,
            "proxy_recipe": "v5_distinct",
            "step_scale_power": 0.45,
            "max_step_multiplier": 6.0,
            "lr_scale_power": 0.2,
            "max_lr_divisor": 2.5,
        },
    },
    "postfix long high splits lr scaled lower lr": {
        "aliases": [],
        "phase4": {
            "sizes": [15_000_000, 40_000_000, 80_000_000, 150_000_000, 300_000_000, 600_000_000, 1_200_000_000],
            "steps": 112,
            "batch_size": 4,
            "lr": 0.00004,
            "num_splits": 7,
            "proxy_recipe": "v5_distinct",
            "step_scale_power": 0.55,
            "max_step_multiplier": 5.0,
            "lr_scale_power": 0.2,
            "max_lr_divisor": 2.5,
        },
    },
    "postfix long high splits lr scaled lower lr longer": {
        "aliases": [],
        "phase4": {
            "sizes": [15_000_000, 40_000_000, 80_000_000, 150_000_000, 300_000_000, 600_000_000, 1_200_000_000],
            "steps": 128,
            "batch_size": 4,
            "lr": 0.00004,
            "num_splits": 7,
            "proxy_recipe": "v5_distinct",
            "step_scale_power": 0.55,
            "max_step_multiplier": 5.0,
            "lr_scale_power": 0.2,
            "max_lr_divisor": 2.5,
        },
    },
    "postfix dense ladder lower lr": {
        "aliases": [],
        "phase4": {
            "sizes": [15_000_000, 30_000_000, 60_000_000, 135_000_000, 270_000_000, 540_000_000, 1_200_000_000],
            "steps": 96,
            "batch_size": 4,
            "lr": 0.00004,
            "num_splits": 5,
            "proxy_recipe": "v5_distinct",
            "reference_size": 135_000_000,
            "step_scale_power": 0.0,
            "max_step_multiplier": 1.0,
        },
    },
}


FAMILY_ALIASES = {
    alias: canonical_name
    for canonical_name, entry in FAMILY_REGISTRY.items()
    for alias in entry.get("aliases", [])
}


def _select_device(requested_device: str) -> tuple[str, str | None]:
    if requested_device != "cuda":
        return requested_device, None
    if not torch.cuda.is_available():
        return "cpu", "torch.cuda.is_available() is false"
    free_bytes, _ = torch.cuda.mem_get_info()
    free_gb = free_bytes / 1e9
    if free_gb < 6.0:
        return "cpu", f"only {free_gb:.2f} GB free VRAM available (< 6.00 GB threshold)"
    return "cuda", None


def _apply_phase4_updates(phase4_config, updates: dict[str, object]) -> None:
    for key, value in updates.items():
        setattr(phase4_config, key, value)


def _details_for(result: dict[str, object], size: int) -> dict[str, object]:
    details = result["details"]
    if size in details:
        return details[size]
    return details[str(size)]


def _summarize(result: dict[str, object]) -> dict[str, object]:
    largest_size = int(result["largest_size"])
    competitor_size = int(result["competitor_size"])
    best_size = int(result["best_size"])
    largest_details = _details_for(result, largest_size)
    competitor_details = _details_for(result, competitor_size)
    return {
        "scaling_efficient": bool(result["scaling_efficient"]),
        "largest_wins": bool(result["largest_wins"]),
        "largest_size": largest_size,
        "best_size": best_size,
        "competitor_size": competitor_size,
        "monotonic_fraction": float(result["monotonic_fraction"]),
        "pairwise_win_rate": float(result["pairwise_win_rate"]),
        "pairwise_margin_std": float(result["pairwise_margin_std"]),
        "confidence_margin": float(result["confidence_margin"]),
        "largest_margin_ratio": float(result["largest_margin_ratio"]),
        "largest_val_loss": float(largest_details["val_loss"]),
        "largest_val_loss_std": float(largest_details["val_loss_std"]),
        "competitor_val_loss": float(competitor_details["val_loss"]),
        "competitor_val_loss_std": float(competitor_details["val_loss_std"]),
        "validation_eval_mode": bool(result.get("validation_eval_mode", True)),
    }


def _mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def _min_run(runs: list[dict[str, object]], metric: str) -> dict[str, object]:
    return min(runs, key=lambda run: float(run["summary"][metric]))


def _rank_families(runs: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for run in runs:
        grouped.setdefault(str(run["family"]), []).append(run)

    leaderboard: list[dict[str, object]] = []
    for family_name, family_runs in grouped.items():
        summaries = [run["summary"] for run in family_runs]
        worst_confidence = _min_run(family_runs, "confidence_margin")
        worst_pairwise = _min_run(family_runs, "pairwise_win_rate")
        leaderboard.append(
            {
                "family": family_name,
                "runs": len(family_runs),
                "pass_count": sum(1 for summary in summaries if summary["scaling_efficient"]),
                "largest_win_count": sum(1 for summary in summaries if summary["largest_wins"]),
                "mean_pairwise_win_rate": _mean([float(summary["pairwise_win_rate"]) for summary in summaries]),
                "min_pairwise_win_rate": min(float(summary["pairwise_win_rate"]) for summary in summaries),
                "mean_confidence_margin": _mean([float(summary["confidence_margin"]) for summary in summaries]),
                "min_confidence_margin": min(float(summary["confidence_margin"]) for summary in summaries),
                "mean_largest_margin_ratio": _mean([float(summary["largest_margin_ratio"]) for summary in summaries]),
                "min_largest_margin_ratio": min(float(summary["largest_margin_ratio"]) for summary in summaries),
                "mean_pairwise_margin_std": _mean([float(summary["pairwise_margin_std"]) for summary in summaries]),
                "worst_confidence_seed": int(worst_confidence["seed"]),
                "worst_confidence_margin": float(worst_confidence["summary"]["confidence_margin"]),
                "worst_pairwise_seed": int(worst_pairwise["seed"]),
                "worst_pairwise_win_rate": float(worst_pairwise["summary"]["pairwise_win_rate"]),
            }
        )

    leaderboard.sort(
        key=lambda entry: (
            -int(entry["pass_count"]),
            -float(entry["largest_win_count"]),
            -float(entry["mean_pairwise_win_rate"]),
            -float(entry["mean_confidence_margin"]),
            -float(entry["mean_largest_margin_ratio"]),
            -float(entry["min_confidence_margin"]),
        )
    )
    for rank, entry in enumerate(leaderboard, start=1):
        entry["rank"] = rank
    return leaderboard


def _write_summary_tsv(path: Path, leaderboard: list[dict[str, object]]) -> None:
    fieldnames = [
        "rank",
        "family",
        "runs",
        "pass_count",
        "largest_win_count",
        "mean_pairwise_win_rate",
        "min_pairwise_win_rate",
        "mean_confidence_margin",
        "min_confidence_margin",
        "mean_largest_margin_ratio",
        "min_largest_margin_ratio",
        "mean_pairwise_margin_std",
        "worst_confidence_seed",
        "worst_confidence_margin",
        "worst_pairwise_seed",
        "worst_pairwise_win_rate",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in leaderboard:
            writer.writerow({name: row[name] for name in fieldnames})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "config" / "ignorance_1.yaml"))
    parser.add_argument("--output", default=str(ROOT / "artifacts" / "postfix_phase4_leaderboard.json"))
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--allow-cpu-fallback", action="store_true")
    parser.add_argument("--seed", dest="seeds", action="append", type=int)
    parser.add_argument("--family", dest="families", action="append")
    args = parser.parse_args()

    config = load_config(args.config)
    requested_device = args.device or config.device
    selected_device, fallback_reason = _select_device(requested_device)
    if requested_device == "cuda" and selected_device != "cuda" and not args.allow_cpu_fallback:
        print(
            f"[postfix-leaderboard] refusing CPU fallback for requested cuda: {fallback_reason}",
            flush=True,
        )
        return 2

    seeds = args.seeds or [47, 50, 51]
    requested_families = args.families or list(FAMILY_REGISTRY.keys())
    family_names: list[str] = []
    for family_name in requested_families:
        canonical_name = FAMILY_ALIASES.get(family_name, family_name)
        if canonical_name not in FAMILY_REGISTRY:
            raise ValueError(f"Unknown family: {family_name}")
        if canonical_name not in family_names:
            family_names.append(canonical_name)
    print(
        f"[postfix-leaderboard] requested_device={requested_device} selected_device={selected_device} seeds={seeds} families={family_names}",
        flush=True,
    )
    if fallback_reason is not None:
        print(f"[postfix-leaderboard] fallback_reason={fallback_reason}", flush=True)

    runs: list[dict[str, object]] = []
    for family_name in family_names:
        family_entry = FAMILY_REGISTRY[family_name]
        family_updates = family_entry["phase4"]
        for seed in seeds:
            run_config = load_config(args.config)
            run_config.seed = seed
            _apply_phase4_updates(run_config.phase4, family_updates)
            run_config.phase4.validation_eval_mode = True
            run_config.phase4.common_random_numbers = False
            set_seed(seed)
            print(f"[postfix-leaderboard] family={family_name} seed={seed} starting", flush=True)
            result = run_phase4(
                copy.deepcopy(run_config.phase4),
                selected_device,
                seed=seed,
                metadata={
                    "profile": run_config.profile,
                    "seed": seed,
                    "family": family_name,
                    "config_path": str(Path(args.config).resolve()),
                    "leaderboard_mode": "postfix_eval_validation",
                },
            )
            summary = _summarize(result)
            run_record = {
                "family": family_name,
                "aliases": family_entry.get("aliases", []),
                "seed": seed,
                "summary": summary,
            }
            runs.append(run_record)
            print(
                "[postfix-leaderboard] "
                f"family={family_name} seed={seed} pass={summary['scaling_efficient']} "
                f"largest_wins={summary['largest_wins']} best={summary['best_size']} "
                f"win_rate={summary['pairwise_win_rate']:.3f} confidence_margin={summary['confidence_margin']:.4f}",
                flush=True,
            )

    leaderboard = _rank_families(runs)
    output = {
        "device": selected_device,
        "seeds": seeds,
        "families": family_names,
        "aliases": {family: FAMILY_REGISTRY[family].get("aliases", []) for family in family_names},
        "runs": runs,
        "leaderboard": leaderboard,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    tsv_path = output_path.with_suffix(".tsv")
    _write_summary_tsv(tsv_path, leaderboard)
    print(f"[postfix-leaderboard] wrote output={output_path}", flush=True)
    print(f"[postfix-leaderboard] wrote summary_tsv={tsv_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())