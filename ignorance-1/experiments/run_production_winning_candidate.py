#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autorun import nested_update
from research.phase4_search_space import rigorous_edge_joint_champion_challenger_staged_hard_base
from src.utils.config import load_config


def build_production_config() -> dict:
    config_path = ROOT / "config" / "ignorance_1.yaml"
    config = load_config(config_path)
    production = {
        "seed": config.seed,
        "device": config.device,
        "profile": "production_winning_candidate",
        "phase1": config.phase1.__dict__.copy(),
        "phase2": config.phase2.__dict__.copy(),
        "phase3": config.phase3.__dict__.copy(),
        "phase4": config.phase4.__dict__.copy(),
    }
    nested_update(
        production,
        {
            "profile": "production_winning_candidate",
            "phase4": rigorous_edge_joint_champion_challenger_staged_hard_base(),
        },
    )
    return production


def validate_recipe(cfg: dict) -> None:
    phase4 = cfg["phase4"]
    expected = rigorous_edge_joint_champion_challenger_staged_hard_base()
    mismatches: list[str] = []
    for key, value in expected.items():
        if phase4.get(key) != value:
            mismatches.append(f"phase4.{key}: expected {value!r}, got {phase4.get(key)!r}")
    if mismatches:
        raise ValueError("Winning Phase 4 recipe mismatch:\n" + "\n".join(mismatches))
    if phase4.get("sizes") != [300_000_000, 600_000_000, 1_200_000_000]:
        raise ValueError(f"Unexpected production sizes: {phase4.get('sizes')!r}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--print-config", action="store_true")
    parser.add_argument(
        "--write-config",
        default=str(ROOT / "artifacts" / "production_winning_candidate_config.json"),
    )
    args = parser.parse_args()

    production = build_production_config()
    validate_recipe(production)

    output = {
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "config_path": str((ROOT / "config" / "ignorance_1.yaml").resolve()),
        "production_config": production,
        "notes": [
            "This locks the production candidate to the research22-25 winning Phase 4 recipe.",
            "Validation of all phases still depends on actually running experiments/validate_phases.py on the hardware.",
            "The current repo entrypoint is still the validation pipeline, not a separate full-train production launcher.",
        ],
    }

    write_path = Path(args.write_config)
    write_path.parent.mkdir(parents=True, exist_ok=True)
    write_path.write_text(json.dumps(output, indent=2) + "\n")

    if args.print_config:
        print(json.dumps(output, indent=2))
    else:
        print(json.dumps({
            "write_config": str(write_path),
            "profile": production["profile"],
            "phase4_sizes": production["phase4"]["sizes"],
            "phase4_dataset": production["phase4"]["phase4_dataset"],
            "phase4_joint_training": production["phase4"]["phase4_joint_training"],
            "champion_challenger_weight": production["phase4"]["champion_challenger_weight"],
        }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
