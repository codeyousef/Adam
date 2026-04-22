from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Phase4Candidate:
    name: str
    hypothesis_id: str
    intervention_type: str
    rationale: str
    expected_effect: str
    phase4_updates: dict[str, Any]


INCUMBENT_CANDIDATE_NAME = "autoresearch rigorous edge joint champion challenger incumbent"


FULL_LADDER_SIZES = [15_000_000, 40_000_000, 80_000_000, 150_000_000, 300_000_000, 600_000_000, 1_200_000_000]
UPPER_LADDER_SIZES = [300_000_000, 600_000_000, 1_200_000_000]
BROADER_LADDER_SIZES = [150_000_000, 300_000_000, 600_000_000, 1_200_000_000]
COMPRESSED_UPPER_LADDER_SIZES = [600_000_000, 1_200_000_000]
EXPANDED_UPPER_LADDER_SIZES = [300_000_000, 600_000_000, 900_000_000, 1_200_000_000]


def semantic_contrast_base() -> dict[str, Any]:
    return {
        "sizes": [300_000_000, 600_000_000, 1_200_000_000],
        "steps": 112,
        "batch_size": 4,
        "lr": 0.00005,
        "num_splits": 7,
        "proxy_recipe": "v5_distinct",
        "validation_eval_mode": True,
        "common_random_numbers": True,
        "split_seed_stride": 1000,
        "data_seed_offset": 0,
        "init_seed_offset": 100_000,
        "train_seed_offset": 200_000,
        "grad_accum_steps": 4,
        "reference_size": 300_000_000,
        "step_scale_power": 0.55,
        "max_step_multiplier": 5.0,
        "lr_scale_power": 0.2,
        "max_lr_divisor": 2.5,
        "ignorance_ood_weight": 0.2,
        "ignorance_pred_weight": 0.2,
        "classifier_weight": 0.25,
        "alignment_prediction_weight": 1.0,
        "alignment_embedding_weight": 0.5,
        "alignment_mse_weight": 0.25,
        "ranking_margin_weight": 0.0,
        "ranking_margin": 0.2,
        "ranking_focal_gamma": 0.0,
        "ranking_start_fraction": 0.0,
        "ranking_ramp_fraction": 0.0,
        "ranking_largest_only": False,
        "epistemic_boundary_weight": 0.0,
        "epistemic_margin": 0.2,
        "phase4_dataset": "semantic_contrast_v1",
        "phase4_balance_families": False,
        "phase4_joint_training": False,
        "champion_challenger_weight": 0.0,
        "champion_challenger_margin": 0.05,
        "champion_challenger_temperature": 0.1,
        "champion_challenger_start_fraction": 0.0,
        "champion_challenger_ramp_fraction": 0.0,
    }


def benchmark_control_base() -> dict[str, Any]:
    return {**semantic_contrast_base(), "phase4_dataset": "benchmark_v1"}


def balanced_ranking_light_base() -> dict[str, Any]:
    return {
        **semantic_contrast_base(),
        "phase4_balance_families": True,
        "ranking_margin_weight": 0.05,
        "ranking_margin": 0.2,
    }


def behavioral_constraints_base() -> dict[str, Any]:
    return {
        **semantic_contrast_base(),
        "phase4_dataset": "behavioral_constraints_v2",
        "phase4_balance_families": True,
    }


def behavioral_constraints_rigorous_base() -> dict[str, Any]:
    return {
        **behavioral_constraints_base(),
        "phase4_dataset": "behavioral_constraints_v2_rigorous",
    }


def behavioral_constraints_adversarial_base() -> dict[str, Any]:
    return {
        **behavioral_constraints_base(),
        "phase4_dataset": "behavioral_constraints_v2_adversarial",
    }


def rigorous_edge_control_base() -> dict[str, Any]:
    return {
        **behavioral_constraints_rigorous_base(),
        "sizes": [*UPPER_LADDER_SIZES],
        "reference_size": 300_000_000,
        "phase4_joint_training": False,
        "champion_challenger_weight": 0.0,
        "champion_challenger_margin": 0.05,
        "champion_challenger_temperature": 0.1,
        "champion_challenger_start_fraction": 0.0,
        "champion_challenger_ramp_fraction": 0.0,
        "ranking_margin_weight": 0.0,
        "ranking_margin": 0.2,
    }


def rigorous_edge_joint_base() -> dict[str, Any]:
    return {
        **rigorous_edge_control_base(),
        "phase4_joint_training": True,
    }


def rigorous_edge_joint_champion_challenger_staged_hard_base() -> dict[str, Any]:
    return {
        **rigorous_edge_joint_base(),
        "champion_challenger_weight": 0.5,
        "champion_challenger_margin": 0.05,
        "champion_challenger_temperature": 0.1,
        "champion_challenger_start_fraction": 0.3,
        "champion_challenger_ramp_fraction": 0.2,
    }


def rigorous_edge_joint_champion_challenger_broader_ladder_base() -> dict[str, Any]:
    return {
        **rigorous_edge_joint_champion_challenger_staged_hard_base(),
        "sizes": [*BROADER_LADDER_SIZES],
        "reference_size": 300_000_000,
    }


def rigorous_edge_joint_champion_challenger_full_ladder_base() -> dict[str, Any]:
    return {
        **rigorous_edge_joint_champion_challenger_staged_hard_base(),
        "sizes": [*FULL_LADDER_SIZES],
        "reference_size": 300_000_000,
    }


def rigorous_edge_joint_champion_challenger_compressed_upper_ladder_base() -> dict[str, Any]:
    return {
        **rigorous_edge_joint_champion_challenger_staged_hard_base(),
        "sizes": [*COMPRESSED_UPPER_LADDER_SIZES],
        "reference_size": 600_000_000,
    }


def rigorous_edge_joint_champion_challenger_expanded_upper_ladder_base() -> dict[str, Any]:
    return {
        **rigorous_edge_joint_champion_challenger_staged_hard_base(),
        "sizes": [*EXPANDED_UPPER_LADDER_SIZES],
        "reference_size": 300_000_000,
    }


def rigorous_edge_joint_champion_challenger_longer_base() -> dict[str, Any]:
    return {
        **rigorous_edge_joint_champion_challenger_staged_hard_base(),
        "steps": 160,
    }


def rigorous_edge_joint_champion_challenger_high_split_base() -> dict[str, Any]:
    return {
        **rigorous_edge_joint_champion_challenger_staged_hard_base(),
        "num_splits": 11,
    }


def semantic_contrast_minimal_pairs_base() -> dict[str, Any]:
    return {
        **semantic_contrast_base(),
        "phase4_dataset": "semantic_contrast_minimal_pairs_v1",
        "phase4_balance_families": True,
    }


def factorized_hard_negatives_base() -> dict[str, Any]:
    return {
        **semantic_contrast_base(),
        "phase4_factorized_hard_negatives": True,
    }


def answerability_split_base() -> dict[str, Any]:
    return {
        **semantic_contrast_base(),
        "phase4_ood_mode": "answerability_split_v1",
    }


def evaluator_style_base() -> dict[str, Any]:
    return {
        **semantic_contrast_base(),
        "phase4_prompt_template": "evaluator_v1",
    }


def candidate_library() -> list[Phase4Candidate]:
    compressed = rigorous_edge_joint_champion_challenger_compressed_upper_ladder_base()
    longer = rigorous_edge_joint_champion_challenger_longer_base()
    high_split = rigorous_edge_joint_champion_challenger_high_split_base()
    incumbent = rigorous_edge_joint_champion_challenger_staged_hard_base()
    return [
        Phase4Candidate(
            name=INCUMBENT_CANDIDATE_NAME,
            hypothesis_id="H1",
            intervention_type="benchmark_strengthening",
            rationale="Use the best replicated rigorous-edge joint champion-challenger recipe as the live incumbent baseline for future autoresearch comparisons.",
            expected_effect="Should preserve the strongest currently known combination of benchmark strength, joint training, and ladder-level separation pressure.",
            phase4_updates={**incumbent},
        ),
        Phase4Candidate(
            name="autoresearch rigorous edge joint champion challenger compressed upper ladder",
            hypothesis_id="H3",
            intervention_type="allocation_change",
            rationale="Research24 showed the 3-point upper ladder is the real winner, but a compressed 2-rung ladder remained viable enough to matter as a compute-saving fallback.",
            expected_effect="If the middle rung is helpful but not strictly necessary, the compressed ladder should remain usable as a fallback while staying below the incumbent on confidence and pairwise fidelity.",
            phase4_updates={**compressed},
        ),
        Phase4Candidate(
            name="autoresearch rigorous edge joint champion challenger longer scout",
            hypothesis_id="H3",
            intervention_type="allocation_change",
            rationale="Probe whether the already-winning regime is still optimization-limited rather than fundamentally capped.",
            expected_effect="If budget is still binding, more steps should improve confidence and worst-pairwise margins without changing the winning mechanism.",
            phase4_updates={**longer},
        ),
        Phase4Candidate(
            name="autoresearch rigorous edge joint champion challenger high split scout",
            hypothesis_id="H4",
            intervention_type="evaluation_stress",
            rationale="Probe whether the remaining uncertainty is mostly evaluation variance rather than a flaw in the winner itself.",
            expected_effect="If variance is the main remaining issue, higher split count should stabilize confidence metrics more than it changes the means.",
            phase4_updates={**high_split},
        ),
    ]
