from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_WINNER_CONFIG = ROOT / "config" / "production_winning_2_7b.yaml"
STRICT_INCUMBENT_NAME = "v4 light rank-reg control"
STRICT_SCOUT_SIZE = 30_000_000


@dataclass(frozen=True)
class StrictEvalCandidate:
    name: str
    hypothesis_id: str
    intervention_type: str
    rationale: str
    expected_effect: str
    phase4_updates: dict[str, Any]
    eval_overrides: dict[str, Any] | None = None
    base_model_path: str | None = None
    warm_start_model_path: str | None = None
    eval_repeats: int = 1


def strict_v363_candidate_library() -> list[StrictEvalCandidate]:
    # v363 already ran and produced results. This stub exists to satisfy
    # runner imports. The actual candidate definitions are in the artifact configs.
    return []


def strict_v364_candidate_library() -> list[StrictEvalCandidate]:
    # v364 is defined inline in run_strict_eval_autoresearch.py
    # to avoid circular dependency issues after search_space corruption.
    return []


def build_strict_eval_base_config() -> dict[str, Any]:
    import yaml
    winner = yaml.safe_load((PRODUCTION_WINNER_CONFIG).read_text())
    config = copy.deepcopy(winner)
    config["profile"] = "strict_eval_autoresearch_v4_base"
    phase4 = config["phase4"]
    phase4["sizes"] = [2_700_000_000]
    phase4["production_mode"] = True
    phase4["production_steps"] = winner["phase4"]["production_steps"]
    phase4["production_phase4_repeats"] = winner["phase4"]["production_phase4_repeats"]
    phase4["retrieval_margin_weight"] = 0.10
    phase4["retrieval_margin"] = 0.20
    phase4["spread_weight"] = 0.01
    phase4["query_spread_weight"] = 0.02
    phase4["pred_spread_weight"] = 0.02
    phase4["rank_reg_weight"] = 0.02
    phase4["rank_reg_eps"] = 1.0e-4
    return config


def build_strict_eval_scout_config(seed: int = 0, phase4_updates: dict[str, Any] | None = None) -> dict[str, Any]:
    config = build_strict_eval_base_config()
    phase4 = config["phase4"]
    phase4["seed"] = seed
    phase4["sizes"] = [STRICT_SCOUT_SIZE]
    phase4["production_mode"] = False
    if phase4_updates:
        phase4.update(copy.deepcopy(phase4_updates))
    return config


def strict_answer_score(summary: dict[str, Any]) -> float:
    strict_status = str(summary.get("strict_status", ""))
    strict_pass_bonus = 10.0 if "PASS" in strict_status else 0.0
    avg_known = float(summary.get("avg_known_similarity", 0.0))
    exact = float(summary.get("avg_known_exact_similarity", 0.0))
    paraphrase = float(summary.get("avg_known_paraphrase_similarity", 0.0))
    synthesis = float(summary.get("synthesis_similarity", 0.0))
    margin = float(summary.get("avg_known_margin", 0.0))
    ignorance_gap = float(summary.get("ignorance_gap", 0.0))
    ood_conf = float(summary.get("avg_ood_confidence", 1.0))
    code_diag = summary.get("code_diagnostics", {}) or {}
    query_diag = summary.get("query_diagnostics", {}) or {}
    code_offdiag = float(code_diag.get("avg_offdiag_similarity", 1.0))
    query_offdiag = float(query_diag.get("avg_offdiag_similarity", 1.0))
    code_rank = float(code_diag.get("participation_ratio_fraction", 0.0))
    query_rank = float(query_diag.get("participation_ratio_fraction", 0.0))
    failures = len(summary.get("strict_failures", []) or [])
    zero_known_penalty = 30.0 if avg_known <= 0.0 and exact <= 0.0 and paraphrase <= 0.0 and synthesis <= 0.0 else 0.0
    objective_bonus = 0.0
    objective_supported_direct_rate = summary.get("objective_supported_direct_rate")
    if objective_supported_direct_rate is not None:
        objective_bonus += 6.0 * float(objective_supported_direct_rate)
    objective_supported_wrong_chunk_rate = summary.get("objective_supported_wrong_chunk_rate")
    if objective_supported_wrong_chunk_rate is not None:
        objective_bonus -= 6.0 * float(objective_supported_wrong_chunk_rate)
    objective_in_domain_unsupported_abstention_rate = summary.get("objective_in_domain_unsupported_abstention_rate")
    if objective_in_domain_unsupported_abstention_rate is not None:
        objective_bonus += 6.0 * float(objective_in_domain_unsupported_abstention_rate)
    objective_confidence_gap = summary.get("objective_confidence_gap")
    if objective_confidence_gap is not None:
        objective_bonus += 4.0 * float(objective_confidence_gap)
    return (
        strict_pass_bonus
        + 6.0 * avg_known
        + 8.0 * exact
        + 8.0 * paraphrase
        + 5.0 * synthesis
        + 5.0 * margin
        + 4.0 * ignorance_gap
        + 8.0 * code_rank
        + 10.0 * query_rank
        - 4.0 * code_offdiag
        - 5.0 * query_offdiag
        - 4.0 * ood_conf
        - 0.5 * failures
        - zero_known_penalty
        + objective_bonus
    )


def strict_candidate_library() -> list[StrictEvalCandidate]:
    base_phase4 = build_strict_eval_scout_config(seed=0)["phase4"]
    return [
        StrictEvalCandidate(
            name=STRICT_INCUMBENT_NAME,
            hypothesis_id="S4",
            intervention_type="query_multiview_mechanism",
            rationale="Lock the best surviving v3 branch as the control before testing explicit query multiview.",
            expected_effect="Provides the stable margin-preserving baseline for evaluating whether multiview helps query geometry and paraphrase.",
            phase4_updates=copy.deepcopy(base_phase4),
        ),
    ]
