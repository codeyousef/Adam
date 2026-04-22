from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.jepa import JEPAModel, approximate_model_params
from src.training.phase4 import _proxy_config


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this probe")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    cfg = _proxy_config(7_000_000_000, "v6_overnight")
    approx_params = approximate_model_params(cfg)
    print(f"approx_params={approx_params}")
    print(f"config={cfg}")

    model = JEPAModel(cfg).to("cuda").to(torch.bfloat16)
    peak_alloc_gb = torch.cuda.max_memory_allocated() / 1e9
    print("instantiated=true")
    print(f"peak_alloc_gb={peak_alloc_gb:.2f}")
    del model
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
