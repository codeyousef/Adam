from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


RECENT_FAMILIES: dict[str, list[int]] = {
    "research5 cosine warmup cooldown": [55, 56, 57],
    "research5 proxy batchnorm disabled": [55, 56, 57],
    "research5 high inertia ema target": [55, 56, 57],
    "research5 trapezoid wsd cooldown": [55, 56, 57],
    "research5 high inertia ema target replication": [58, 59, 60],
    "research6 baseline incumbent": [61, 62, 63],
    "research6 protocol common randomness": [61, 62, 63],
    "research6 reference anchor 150m": [61, 62, 63],
    "research6 reference anchor 600m": [61, 62, 63],
    "research6 effective batch accum4": [61, 62, 63],
    "research7 baseline incumbent": [64, 65, 66],
    "research7 common randomness accum4": [64, 65, 66],
    "research7 common randomness accum4 anchor150m": [64, 65, 66],
    "research8 objective incumbent crn accum4 anchor150m": [67, 68, 69],
    "research8 objective lighter ignorance": [67, 68, 69],
    "research8 objective no pred ignorance": [67, 68, 69],
    "research8 objective lighter classifier": [67, 68, 69],
    "research9 alignment incumbent control": [70, 71, 72],
    "research9 alignment production mix": [70, 71, 72],
    "research9 alignment prediction heavy": [70, 71, 72],
    "research9 alignment embedding heavy": [70, 71, 72],
    "research10 embedding heavy control": [73, 74, 75],
    "research10 embedding heavy lighter classifier": [73, 74, 75],
    "research10 embedding heavy no pred ignorance": [73, 74, 75],
    "research10 embedding heavy lower mse": [73, 74, 75],
    "research11 baseline incumbent control": [76, 77, 78],
    "research11 compute fair incumbent control": [76, 77, 78],
    "research11 compute fair simple core": [76, 77, 78],
    "research11 compute fair simple core embedding heavy": [76, 77, 78],
    "research17 benchmark control upper ladder": [94, 95, 96],
    "research17 semantic contrast benchmark upper ladder": [94, 95, 96],
    "research17 semantic contrast benchmark joint upper ladder": [94, 95, 96],
    "research18 benchmark control upper ladder": [97, 98, 99],
    "research18 semantic contrast upper ladder": [97, 98, 99],
    "research18 semantic contrast balanced upper ladder": [97, 98, 99],
    "research18 semantic contrast joint upper ladder": [97, 98, 99],
    "research18 semantic contrast balanced joint upper ladder": [97, 98, 99],
    "research18 semantic contrast balanced upper ladder ranking light": [97, 98, 99],
    "research18 semantic contrast balanced joint upper ladder ranking light": [97, 98, 99],
    "research18 semantic contrast balanced joint upper ladder champion challenger staged hard": [97, 98, 99],
    "research18 semantic contrast balanced joint upper ladder champion challenger staged smooth": [97, 98, 99],
    "research18 semantic contrast balanced joint upper ladder champion challenger immediate hard": [97, 98, 99],
    "research18 semantic contrast balanced joint upper ladder champion challenger plus ranking": [97, 98, 99],
    "research19 benchmark control upper ladder": [100, 101, 102],
    "research19 semantic contrast upper ladder replicate": [100, 101, 102],
    "research19 semantic contrast upper ladder ranking light": [100, 101, 102],
    "research19 semantic contrast upper ladder ranking medium": [100, 101, 102],
    "research19 semantic contrast upper ladder champion challenger staged smooth": [100, 101, 102],
    "research19 semantic contrast upper ladder champion challenger staged smooth plus ranking": [100, 101, 102],
}


ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "artifacts" / "runs"


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def load_family_run(family: str, seed: int) -> dict[str, Any]:
    slug = f"{family} seed{seed}".lower().replace(" ", "-")
    matches = sorted(RUNS_DIR.glob(f"*-{slug}/results.json"))
    if not matches:
        raise FileNotFoundError(f"Missing run for {family} seed {seed}")
    artifact = json.loads(matches[-1].read_text())
    return artifact["phase4"]


def summarize_family_block(runs: list[dict[str, Any]]) -> dict[str, Any]:
    largest_wins = sum(1 for run in runs if bool(run["largest_wins"]))
    mean_pairwise_win_rate = _mean([float(run["pairwise_win_rate"]) for run in runs])
    min_pairwise_win_rate = min(float(run["pairwise_win_rate"]) for run in runs)
    mean_pairwise_margin_std = _mean([float(run["pairwise_margin_std"]) for run in runs])
    mean_confidence_margin = _mean([float(run["confidence_margin"]) for run in runs])
    max_confidence_margin = max(float(run["confidence_margin"]) for run in runs)
    mean_largest_margin_ratio = _mean([float(run["largest_margin_ratio"]) for run in runs])
    mean_monotonic_fraction = _mean([float(run["monotonic_fraction"]) for run in runs])
    best_sizes = [int(run["best_size"]) for run in runs]
    mode_best_size, mode_best_count = Counter(best_sizes).most_common(1)[0]
    top_rung_consistency = largest_wins >= 2 and mode_best_size == max(best_sizes)
    blocked_top_rung_pass = (
        largest_wins >= 2
        and mean_pairwise_win_rate >= 0.56
        and min_pairwise_win_rate >= 0.42
        and mean_monotonic_fraction >= 0.83
        and max_confidence_margin >= -0.10
    )
    variance_aware_pass = (
        blocked_top_rung_pass
        and mean_pairwise_margin_std <= 0.12
        and mean_confidence_margin >= -0.15
        and mean_largest_margin_ratio >= 0.0
    )
    return {
        "runs": len(runs),
        "largest_wins": largest_wins,
        "mean_pairwise_win_rate": mean_pairwise_win_rate,
        "min_pairwise_win_rate": min_pairwise_win_rate,
        "mean_pairwise_margin_std": mean_pairwise_margin_std,
        "mean_confidence_margin": mean_confidence_margin,
        "max_confidence_margin": max_confidence_margin,
        "mean_largest_margin_ratio": mean_largest_margin_ratio,
        "mean_monotonic_fraction": mean_monotonic_fraction,
        "mode_best_size": mode_best_size,
        "mode_best_count": mode_best_count,
        "top_rung_consistency": top_rung_consistency,
        "blocked_top_rung_pass": blocked_top_rung_pass,
        "variance_aware_pass": variance_aware_pass,
    }


def review_families(families: dict[str, list[int]]) -> list[dict[str, Any]]:
    reviews = []
    for family, seeds in families.items():
        runs = [load_family_run(family, seed) for seed in seeds]
        reviews.append(
            {
                "family": family,
                "seeds": seeds,
                "protocol_review": summarize_family_block(runs),
            }
        )
    reviews.sort(
        key=lambda item: (
            -int(item["protocol_review"]["variance_aware_pass"]),
            -int(item["protocol_review"]["blocked_top_rung_pass"]),
            -float(item["protocol_review"]["largest_wins"]),
            -float(item["protocol_review"]["mean_pairwise_win_rate"]),
            -float(item["protocol_review"]["max_confidence_margin"]),
        )
    )
    return reviews


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", action="append", help="Optional family name to review; may be repeated")
    args = parser.parse_args()

    families = RECENT_FAMILIES
    if args.family:
        families = {name: RECENT_FAMILIES[name] for name in args.family}

    print(json.dumps(review_families(families), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
