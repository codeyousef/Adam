from __future__ import annotations

import json
from pathlib import Path
from typing import Any

FAMILIES = [
    "research25 rigorous edge joint champion challenger incumbent upper ladder",
    "research25 rigorous edge joint champion challenger longer upper ladder",
    "research25 rigorous edge joint champion challenger high split upper ladder",
]
SEEDS = [122, 123, 124]

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


def summarize_family(family: str) -> dict[str, Any]:
    runs = [load_family_run(family, seed) for seed in SEEDS]
    return {
        "family": family,
        "seeds": SEEDS,
        "largest_wins": sum(1 for run in runs if bool(run["largest_wins"])),
        "mean_pairwise_win_rate": _mean([float(run["pairwise_win_rate"]) for run in runs]),
        "mean_confidence_margin": _mean([float(run["confidence_margin"]) for run in runs]),
        "mean_worst_pairwise_margin": _mean([float(run["worst_pairwise_margin"]) for run in runs]),
        "phase4_pass_count": sum(1 for run in runs if bool(run["scaling_efficient"]) or bool(run.get("recalibrated_scaling_efficient", False))),
        "mean_steps": _mean([float(run.get("steps", 0.0)) for run in runs]),
        "mean_num_splits": _mean([float(run.get("num_splits", 0.0)) for run in runs]),
        "joint_training_sizes": runs[-1].get("joint_training_sizes", []),
    }


def main() -> int:
    print(json.dumps([summarize_family(family) for family in FAMILIES], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
