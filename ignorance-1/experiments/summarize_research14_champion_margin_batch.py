#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "artifacts" / "runs"
SEEDS = [85, 86, 87]
FAMILIES = [
    "research14 incumbent control reseed",
    "research14 staged champion margin hard",
    "research14 staged champion margin smooth focal",
    "research14 immediate champion margin hard",
]


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def load_result(family: str, seed: int) -> dict:
    slug = f"{family} seed{seed}".lower().replace(" ", "-")
    candidates = sorted(RUNS_DIR.glob(f"*-{slug}/results.json"))
    if not candidates:
        raise FileNotFoundError(f"missing results for {family} seed {seed}")
    return json.loads(candidates[-1].read_text())


def summarize_family(family: str) -> dict:
    runs = []
    for seed in SEEDS:
        result = load_result(family, seed)
        p4 = result["phase4"]
        runs.append(
            {
                "seed": seed,
                "largest_wins": bool(p4["largest_wins"]),
                "pairwise_win_rate": float(p4["pairwise_win_rate"]),
                "pairwise_margin_std": float(p4["pairwise_margin_std"]),
                "confidence_margin": float(p4["confidence_margin"]),
                "largest_margin_ratio": float(p4["largest_margin_ratio"]),
                "monotonic_fraction": float(p4["monotonic_fraction"]),
                "best_size": int(p4["best_size"]),
                "competitor_size": int(p4["competitor_size"]),
                "phase4_pass": bool(p4["scaling_efficient"]),
                "ranking_margin_weight": float(p4.get("ranking_margin_weight", 0.0)),
                "ranking_margin": float(p4.get("ranking_margin", 0.2)),
                "ranking_focal_gamma": float(p4.get("ranking_focal_gamma", 0.0)),
                "ranking_start_fraction": float(p4.get("ranking_start_fraction", 0.0)),
                "ranking_ramp_fraction": float(p4.get("ranking_ramp_fraction", 0.0)),
                "ranking_largest_only": bool(p4.get("ranking_largest_only", False)),
            }
        )
    return {
        "family": family,
        "seeds": SEEDS,
        "largest_wins": sum(1 for run in runs if run["largest_wins"]),
        "mean_pairwise_win_rate": mean([run["pairwise_win_rate"] for run in runs]),
        "min_pairwise_win_rate": min(run["pairwise_win_rate"] for run in runs),
        "mean_pairwise_margin_std": mean([run["pairwise_margin_std"] for run in runs]),
        "mean_confidence_margin": mean([run["confidence_margin"] for run in runs]),
        "max_confidence_margin": max(run["confidence_margin"] for run in runs),
        "mean_largest_margin_ratio": mean([run["largest_margin_ratio"] for run in runs]),
        "mean_monotonic_fraction": mean([run["monotonic_fraction"] for run in runs]),
        "phase4_pass_count": sum(1 for run in runs if run["phase4_pass"]),
        "runs": runs,
    }


def main() -> int:
    output = [summarize_family(family) for family in FAMILIES]
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
