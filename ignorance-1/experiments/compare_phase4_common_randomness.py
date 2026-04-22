from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.phase4 import run_phase4
from src.utils.config import load_config
from src.utils.data import set_seed


PHASE4_FAMILY = {
    "sizes": [15_000_000, 40_000_000, 80_000_000, 150_000_000, 300_000_000, 600_000_000, 1_200_000_000],
    "steps": 96,
    "batch_size": 4,
    "lr": 0.00005,
    "num_splits": 5,
    "proxy_recipe": "v5_distinct",
    "step_scale_power": 0.5,
    "max_step_multiplier": 4.0,
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
    largest_details = _details_for(result, largest_size)
    competitor_details = _details_for(result, competitor_size)
    return {
        "scaling_efficient": bool(result["scaling_efficient"]),
        "largest_size": largest_size,
        "best_size": int(result["best_size"]),
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
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "config" / "ignorance_1.yaml"))
    parser.add_argument("--output", default=str(ROOT / "artifacts" / "common_randomness_comparison.json"))
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--allow-cpu-fallback", action="store_true")
    parser.add_argument("--seed", dest="seeds", action="append", type=int)
    args = parser.parse_args()

    config = load_config(args.config)
    requested_device = args.device or config.device
    selected_device, fallback_reason = _select_device(requested_device)
    if requested_device == "cuda" and selected_device != "cuda" and not args.allow_cpu_fallback:
        print(
            f"[common-rng] refusing CPU fallback for requested cuda: {fallback_reason}",
            flush=True,
        )
        return 2

    seeds = args.seeds or [47, 50]
    print(
        f"[common-rng] requested_device={requested_device} selected_device={selected_device} seeds={seeds}",
        flush=True,
    )
    if fallback_reason is not None:
        print(f"[common-rng] fallback_reason={fallback_reason}", flush=True)

    runs: list[dict[str, object]] = []
    by_seed: dict[int, dict[str, dict[str, object]]] = {}
    for seed in seeds:
        for mode_name, mode_updates in (
            ("baseline", {}),
            (
                "common_random_numbers",
                {
                    "common_random_numbers": True,
                    "split_seed_stride": 1000,
                    "data_seed_offset": 0,
                    "init_seed_offset": 100_000,
                    "train_seed_offset": 200_000,
                },
            ),
        ):
            run_config = load_config(args.config)
            run_config.seed = seed
            _apply_phase4_updates(run_config.phase4, PHASE4_FAMILY)
            _apply_phase4_updates(run_config.phase4, mode_updates)
            set_seed(seed)
            print(f"[common-rng] seed={seed} mode={mode_name} starting", flush=True)
            result = run_phase4(
                copy.deepcopy(run_config.phase4),
                selected_device,
                seed=seed,
                metadata={
                    "profile": run_config.profile,
                    "seed": seed,
                    "comparison_mode": mode_name,
                    "config_path": str(Path(args.config).resolve()),
                },
            )
            summary = _summarize(result)
            record = {
                "seed": seed,
                "mode": mode_name,
                "summary": summary,
            }
            runs.append(record)
            by_seed.setdefault(seed, {})[mode_name] = summary
            print(
                "[common-rng] "
                f"seed={seed} mode={mode_name} pass={summary['scaling_efficient']} "
                f"win_rate={summary['pairwise_win_rate']:.3f} "
                f"margin_std={summary['pairwise_margin_std']:.4f} "
                f"confidence_margin={summary['confidence_margin']:.4f}",
                flush=True,
            )

    comparisons: list[dict[str, object]] = []
    for seed in seeds:
        baseline = by_seed[seed]["baseline"]
        common = by_seed[seed]["common_random_numbers"]
        comparisons.append(
            {
                "seed": seed,
                "pairwise_margin_std_delta": common["pairwise_margin_std"] - baseline["pairwise_margin_std"],
                "largest_val_loss_std_delta": common["largest_val_loss_std"] - baseline["largest_val_loss_std"],
                "competitor_val_loss_std_delta": common["competitor_val_loss_std"] - baseline["competitor_val_loss_std"],
                "pairwise_win_rate_delta": common["pairwise_win_rate"] - baseline["pairwise_win_rate"],
                "confidence_margin_delta": common["confidence_margin"] - baseline["confidence_margin"],
                "largest_margin_ratio_delta": common["largest_margin_ratio"] - baseline["largest_margin_ratio"],
                "baseline": baseline,
                "common_random_numbers": common,
            }
        )

    output = {
        "family": "final confirmation family working recipe",
        "phase4_updates": PHASE4_FAMILY,
        "device": selected_device,
        "seeds": seeds,
        "runs": runs,
        "comparisons": comparisons,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    print(f"[common-rng] wrote output={output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())