from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Phase1Config:
    embed_dim: int
    encoder_layers: int
    encoder_heads: int
    predictor_layers: int
    predictor_heads: int
    lambdas: list[float]
    projections: int
    batch_size: int
    steps: int
    seq_len: int
    vocab_size: int
    patch_size: int
    lr: float


@dataclass
class Phase2Config:
    batch_size: int
    epochs: int
    lr: float
    retrieval_k: int
    answer_threshold: float
    direct_penalty: float


@dataclass
class Phase3Config:
    horizon: int
    num_samples: int
    num_elites: int
    num_iterations: int
    tasks: int


@dataclass
class Phase4Config:
    sizes: list[int]
    steps: int
    batch_size: int
    lr: float
    max_vram_gb: float
    num_splits: int = 3
    proxy_recipe: str = "v4"
    step_scale_power: float = 0.0
    max_step_multiplier: float = 1.0
    lr_scale_power: float = 0.0
    max_lr_divisor: float = 1.0


@dataclass
class RunConfig:
    seed: int
    device: str
    profile: str
    phase1: Phase1Config
    phase2: Phase2Config
    phase3: Phase3Config
    phase4: Phase4Config


def _section(data: dict[str, Any], key: str, cls: type[Any]) -> Any:
    return cls(**data[key])


def load_config(path: str | Path) -> RunConfig:
    with Path(path).open() as handle:
        data = yaml.safe_load(handle)
    return RunConfig(
        seed=data["seed"],
        device=data["device"],
        profile=data.get("profile", "smoke"),
        phase1=_section(data, "phase1", Phase1Config),
        phase2=_section(data, "phase2", Phase2Config),
        phase3=_section(data, "phase3", Phase3Config),
        phase4=_section(data, "phase4", Phase4Config),
    )