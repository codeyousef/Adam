from __future__ import annotations

import argparse
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "config" / "medium_validation.yaml"))
    parser.add_argument("--output", default=str(ROOT / "artifacts" / "phase4_results.json"))
    parser.add_argument("--checkpoint", default=str(ROOT / "artifacts" / "phase4_progress.json"))
    parser.add_argument("--log", default=str(ROOT / "artifacts" / "phase4.log"))
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--allow-cpu-fallback", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    requested_device = args.device or config.device
    selected_device, fallback_reason = _select_device(requested_device)

    if requested_device == "cuda" and selected_device != "cuda":
        print(
            f"[phase4-runner] refusing CPU fallback for requested cuda: {fallback_reason}",
            flush=True,
        )
        if not args.allow_cpu_fallback:
            return 2

    print(
        f"[phase4-runner] profile={config.profile} requested_device={requested_device} selected_device={selected_device}",
        flush=True,
    )
    if fallback_reason is not None:
        print(f"[phase4-runner] fallback_reason={fallback_reason}", flush=True)
    print(
        f"[phase4-runner] checkpoint={args.checkpoint} log={args.log} output={args.output}",
        flush=True,
    )

    set_seed(config.seed)
    try:
        result = run_phase4(
            config.phase4,
            selected_device,
            seed=config.seed,
            checkpoint_path=args.checkpoint,
            log_path=args.log,
            metadata={
                "profile": config.profile,
                "seed": config.seed,
                "requested_device": requested_device,
                "selected_device": selected_device,
                "config_path": str(Path(args.config).resolve()),
            },
        )
    except KeyboardInterrupt:
        print("[phase4-runner] interrupted; partial progress remains in checkpoint/log files", flush=True)
        return 130

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"[phase4-runner] wrote output={output_path}", flush=True)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())