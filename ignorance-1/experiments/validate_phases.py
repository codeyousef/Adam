from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.phase1 import run_phase1
from src.training.phase2 import run_phase2
from src.training.phase3 import run_phase3
from src.training.phase4 import run_phase4
from src.utils.config import load_config
from src.utils.data import set_seed


def _log(message: str) -> None:
    print(message, flush=True)


def _write_report(report_path: Path, results: dict) -> None:
    lines = [
        "# IGNORANCE-1 Report",
        "",
        f"Generated: {datetime.utcnow().isoformat()}Z",
        "",
        "## Run Mode",
        "",
        f"- Profile: {results['profile']}",
        f"- Device: {results['device']}",
        f"- Requested device: {results['requested_device']}",
        f"- Full 14-day validation completed: no",
        "",
        "## Summary",
        "",
        f"- Phase 1 pass: {results['phase1']['optimal_lambda'] is not None}",
        f"- Phase 2 pass: {results['phase2']['passes_ignorance_test']}",
        f"- Phase 3 pass: {results['phase3']['passes']}",
        f"- Phase 4 pass: {results['phase4']['scaling_efficient']}",
        "",
        "## Phase Details",
        "",
        "```json",
        json.dumps(results, indent=2),
        "```",
        "",
        "## Notes",
        "",
        "- This run is a smoke-scale implementation of the JEPA workflow in a new isolated folder.",
        "- Phase 4 uses proxy model sizes to test the pipeline and VRAM behavior quickly.",
        "- Retrieval uses FAISS if available, otherwise a cosine-similarity torch backend.",
    ]
    report_path.write_text("\n".join(lines) + "\n")


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
    parser.add_argument("--config", default=str(ROOT / "config" / "ignorance_1.yaml"))
    parser.add_argument("--output", default=str(ROOT / "artifacts" / "results.json"))
    parser.add_argument("--report", default=str(ROOT / "REPORT.md"))
    parser.add_argument("--allow-cpu-fallback", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    selected_device, fallback_reason = _select_device(config.device)

    if config.device == "cuda" and selected_device != "cuda":
        _log(f"[validate] refusing CPU fallback for requested cuda: {fallback_reason}")
        if not args.allow_cpu_fallback:
            return 2

    _log(f"[validate] profile={config.profile} requested_device={config.device} selected_device={selected_device}")
    if fallback_reason is not None:
        _log(f"[validate] fallback_reason={fallback_reason}")

    set_seed(config.seed)
    _log("[validate] starting phase1")
    phase1 = run_phase1(config.phase1, selected_device)
    _log(f"[validate] finished phase1 optimal_lambda={phase1['optimal_lambda']}")
    set_seed(config.seed)
    _log("[validate] starting phase2")
    phase2 = run_phase2(config.phase2, phase1, selected_device)
    _log(
        "[validate] finished phase2 "
        f"without={phase2['accuracy_without_retrieval']:.3f} "
        f"with={phase2['accuracy_with_retrieval']:.3f} "
        f"pass={phase2['passes_ignorance_test']}"
    )
    set_seed(config.seed)
    _log("[validate] starting phase3")
    phase3 = run_phase3(config.phase3, selected_device)
    _log(
        "[validate] finished phase3 "
        f"success={phase3['planning_success_rate']:.3f} pass={phase3['passes']}"
    )
    set_seed(config.seed)
    _log("[validate] starting phase4")
    phase4 = run_phase4(config.phase4, selected_device, seed=config.seed)
    _log(
        "[validate] finished phase4 "
        f"best_size={phase4['best_size']} pass={phase4['scaling_efficient']}"
    )

    results = {
        "profile": config.profile,
        "requested_device": config.device,
        "device": selected_device,
        "phase1": phase1,
        "phase2": phase2,
        "phase3": phase3,
        "phase4": phase4,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2) + "\n")
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    _write_report(report_path, results)
    _log(f"[validate] wrote output={output_path} report={report_path}")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())