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
    production_mode: bool = False
    production_steps: int | None = None
    production_phase4_repeats: int | None = None
    num_splits: int = 3
    phase4_dataset: str = "benchmark_v1"
    phase4_balance_families: bool = False
    phase4_factorized_hard_negatives: bool = False
    phase4_ood_mode: str = "default"
    phase4_prompt_template: str = "default"
    proxy_recipe: str = "v4"
    reference_size: int | None = None
    step_scale_power: float = 0.0
    max_step_multiplier: float = 1.0
    lr_scale_power: float = 0.0
    max_lr_divisor: float = 1.0
    scheduler: str = "constant"
    warmup_fraction: float = 0.0
    min_lr_ratio: float = 1.0
    wsd_decay_start_fraction: float = 0.8
    wsd_cooldown_shape: str = "linear"
    grad_accum_steps: int = 1
    ema_target_decay: float = 0.0
    proxy_disable_batchnorm: bool = False
    common_random_numbers: bool = False
    split_seed_stride: int = 1000
    data_seed_offset: int = 0
    init_seed_offset: int = 100_000
    train_seed_offset: int = 200_000
    validation_eval_mode: bool = True
    ignorance_ood_weight: float = 0.2
    ignorance_pred_weight: float = 0.2
    classifier_weight: float = 0.25
    classifier_query_weight: float = 1.0
    classifier_prediction_weight: float = 0.0
    alignment_prediction_weight: float = 1.0
    alignment_embedding_weight: float = 0.5
    alignment_mse_weight: float = 0.25
    alignment_symmetric: bool = True
    alignment_decoupled: bool = False
    retrieval_margin_prediction_weight: float = 1.0
    retrieval_margin_embedding_weight: float = 0.5
    ranking_margin_weight: float = 0.0
    ranking_margin: float = 0.2
    ranking_focal_gamma: float = 0.0
    ranking_prediction_weight: float = 1.0
    ranking_embedding_weight: float = 1.0
    ranking_start_fraction: float = 0.0
    ranking_ramp_fraction: float = 0.0
    ranking_largest_only: bool = False
    support_slate_localization_weight: float = 0.0
    support_slate_prediction_weight: float = 0.0
    support_slate_same_family_weight: float = 1.0
    support_slate_cross_family_weight: float = 1.0
    support_slate_temperature: float = 0.07
    support_slate_margin_weight: float = 0.0
    support_slate_margin: float = 0.0
    support_slate_cross_family_negatives: int = 0
    sigreg_weight: float = 0.5
    epistemic_boundary_weight: float = 0.0
    epistemic_margin: float = 0.2
    epistemic_query_weight: float = 1.0
    epistemic_prediction_weight: float = 1.0
    freeze_backbone: bool = False
    reset_query_head_on_resume: bool = False
    use_retrieval_head: bool = False
    retrieval_head_dim: int = 0
    retrieval_head_hidden_dim: int = 0
    use_retrieval_facets: bool = False
    retrieval_num_facets: int = 0
    retrieval_facet_dim: int = 0
    retrieval_facet_hidden_dim: int = 0
    retrieval_facet_separate_query_code: bool = False
    retrieval_facet_score_mode: str = "hard_maxsim"
    retrieval_facet_softmax_temperature: float = 0.1
    retrieval_facet_loss_weight: float = 0.0
    phase4_joint_training: bool = False
    champion_challenger_weight: float = 0.0
    champion_challenger_margin: float = 0.05
    champion_challenger_temperature: float = 0.1
    champion_challenger_start_fraction: float = 0.0
    champion_challenger_ramp_fraction: float = 0.0


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