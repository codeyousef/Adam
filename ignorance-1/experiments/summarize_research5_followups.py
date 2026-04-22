#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "artifacts" / "runs"
FAMILY_SEEDS = {
    "research5 trapezoid wsd cooldown": [55, 56, 57],
    "research5 high inertia ema target replication": [58, 59, 60],
}


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def load_result(family: str, seed: int) -> dict:
    slug = f"{family} seed{seed}".lower().replace(" ", "-")
    candidates = sorted(RUNS_DIR.glob(f"*-{slug}/results.json"))
    if not candidates:
        raise FileNotFoundError(f"missing results for {family} seed {seed}")
    return json.loads(candidates[-1].read_text())


def summarize_family(family: str, seeds: list[int]) -> dict:
    runs = []
    for seed in seeds:
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
            }
        )
    return {
        "family": family,
        "seeds": seeds,
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
    output = [summarize_family(family, seeds) for family, seeds in FAMILY_SEEDS.items()]
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
