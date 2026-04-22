from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from research.phase4_search_space import (
    INCUMBENT_CANDIDATE_NAME,
    Phase4Candidate,
    candidate_library,
    rigorous_edge_joint_champion_challenger_staged_hard_base,
)

ROOT = Path(__file__).resolve().parents[1]
HYPOTHESIS_PATH = ROOT / "research" / "phase4_hypotheses.yaml"


@dataclass(frozen=True)
class CandidateScore:
    candidate: Phase4Candidate
    score: float
    reasons: list[str]


def load_hypotheses(path: Path = HYPOTHESIS_PATH) -> list[dict[str, Any]]:
    data = yaml.safe_load(path.read_text())
    return list(data.get("hypotheses", []))


def phase4_fingerprint(phase4_updates: dict[str, Any]) -> str:
    return json.dumps(phase4_updates, sort_keys=True)


def answer_score_from_phase4_result(result: dict[str, Any]) -> float:
    mean_confidence_margin = float(result.get("confidence_margin", 0.0))
    pairwise_win_rate = float(result.get("pairwise_win_rate", 0.0))
    largest_wins = 1.0 if bool(result.get("largest_wins", False)) else 0.0
    monotonic_fraction = float(result.get("monotonic_fraction", 0.0))
    pairwise_margin_std = float(result.get("pairwise_margin_std", 0.0))
    epistemic_gap = float(result.get("epistemic_gap", 0.0))
    epistemic_gap_margin = float(result.get("epistemic_gap_margin", 0.0))
    epistemic_gap_monotonic_fraction = float(result.get("epistemic_gap_monotonic_fraction", 1.0))
    mode_penalty = 0.0 if int(result.get("best_size", 0)) == int(result.get("largest_size", 0)) else 0.25
    monotonic_penalty = max(0.0, 0.9 - monotonic_fraction)
    variance_penalty = max(0.0, pairwise_margin_std - 0.12)
    epistemic_monotonic_penalty = max(0.0, 0.9 - epistemic_gap_monotonic_fraction)
    return (
        4.0 * mean_confidence_margin
        + 2.0 * pairwise_win_rate
        + 1.0 * largest_wins
        + 1.5 * epistemic_gap
        + 2.0 * epistemic_gap_margin
        - mode_penalty
        - monotonic_penalty
        - variance_penalty
        - 0.5 * epistemic_monotonic_penalty
    )


def infer_branch_status(evidence_rows: list[dict[str, Any]]) -> dict[str, str]:
    status: dict[str, str] = {}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in evidence_rows:
        grouped.setdefault(str(row.get("candidate_name", "")), []).append(row)
    for name, rows in grouped.items():
        if any(row.get("decision") == "kill" for row in rows):
            status[name] = "killed"
        elif any(row.get("decision") == "promote" for row in rows):
            status[name] = "promoted"
        elif any(row.get("decision") == "hold" for row in rows):
            status[name] = "held"
        else:
            status[name] = "active"
    return status


def hypothesis_activity(evidence_rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    activity: dict[str, dict[str, int]] = {}
    for row in evidence_rows:
        hypothesis_id = str(row.get("hypothesis_id", ""))
        if not hypothesis_id:
            continue
        bucket = activity.setdefault(hypothesis_id, {"scout_events": 0, "kills": 0, "holds": 0, "promotes": 0})
        if row.get("stage") == "scout":
            bucket["scout_events"] += 1
        decision = str(row.get("decision", ""))
        if decision == "kill":
            bucket["kills"] += 1
        elif decision == "hold":
            bucket["holds"] += 1
        elif decision == "promote":
            bucket["promotes"] += 1
    return activity


def score_candidates(history_rows: list[dict[str, str]], evidence_rows: list[dict[str, Any]]) -> list[CandidateScore]:
    hypotheses = {item["id"]: item for item in load_hypotheses()}
    branch_status = infer_branch_status(evidence_rows)
    activity = hypothesis_activity(evidence_rows)
    prior_runs = {row.get("description", "") for row in history_rows if row.get("status") == "ok"}
    incumbent_fingerprint = phase4_fingerprint(rigorous_edge_joint_champion_challenger_staged_hard_base())
    incumbent_baselined = any(row.get("candidate_name") == INCUMBENT_CANDIDATE_NAME for row in evidence_rows)
    scored: list[CandidateScore] = []
    for candidate in candidate_library():
        reasons: list[str] = []
        branch_state = branch_status.get(candidate.name)
        if branch_state == "killed":
            continue
        score = 0.0
        hypothesis = hypotheses.get(candidate.hypothesis_id)
        if hypothesis is not None:
            score += float(hypothesis.get("confidence", 0.0)) * 10.0
            reasons.append(f"hypothesis {candidate.hypothesis_id} confidence={hypothesis.get('confidence', 0.0):.2f}")
        hypothesis_counts = activity.get(candidate.hypothesis_id, {"scout_events": 0, "kills": 0, "holds": 0, "promotes": 0})
        if hypothesis_counts["scout_events"] == 0:
            score += 3.0
            reasons.append("unexplored hypothesis bonus")
        else:
            score -= float(hypothesis_counts["scout_events"])
            reasons.append(f"hypothesis already tested {hypothesis_counts['scout_events']} scout time(s)")
        if candidate.intervention_type == "benchmark_strengthening":
            score += 1.0
            reasons.append("benchmark branch remains useful only until anchored")
            dataset = candidate.phase4_updates.get("phase4_dataset")
            if dataset in {
                "behavioral_constraints_v2",
                "behavioral_constraints_v2_rigorous",
                "behavioral_constraints_v2_adversarial",
                "semantic_contrast_minimal_pairs_v1",
            }:
                score += 3.0
                reasons.append("behavioral/minimal-pair benchmark redesign gets post-research19 priority")
                if dataset == "behavioral_constraints_v2":
                    score += 1.0
                    reasons.append("base behavioral constraints v2 is the primary redesign anchor")
                elif dataset == "behavioral_constraints_v2_rigorous":
                    score += 0.25
                    reasons.append("rigorous edge variant deepens invariant stress")
                elif dataset == "behavioral_constraints_v2_adversarial":
                    score += 0.1
                    reasons.append("adversarial negatives deepen near-miss discrimination")
                elif dataset == "semantic_contrast_minimal_pairs_v1":
                    score += 0.15
                    reasons.append("minimal-pair variant probes single-clause semantic sensitivity")
            if candidate.phase4_updates.get("phase4_factorized_hard_negatives"):
                score += 2.0
                reasons.append("factorized hard negatives add structural benchmark novelty")
            if candidate.phase4_updates.get("phase4_ood_mode") == "answerability_split_v1":
                score += 2.0
                reasons.append("answerability split targets know-vs-should-not-know directly")
        elif candidate.intervention_type == "separation_objective":
            score += 1.0
            reasons.append("objective branch gets exploration priority")
            if candidate.phase4_updates.get("phase4_balance_families"):
                score += 1.5
                reasons.append("balanced semantic branch gets structural novelty bonus")
        elif candidate.intervention_type in {"allocation_change", "evaluation_stress"}:
            score += 0.5
            reasons.append("non-objective structural probe")
            if candidate.phase4_updates.get("phase4_prompt_template") == "evaluator_v1":
                score += 2.0
                reasons.append("evaluator-style protocol adds untried protocol stress branch")
        if (
            incumbent_baselined
            and candidate.phase4_updates.get("phase4_dataset") == "behavioral_constraints_v2_rigorous"
            and candidate.phase4_updates.get("phase4_joint_training")
            and float(candidate.phase4_updates.get("champion_challenger_weight", 0.0)) > 0.0
        ):
            score += 2.5
            reasons.append("winner-neighborhood bonus after incumbent replication")
            sizes = list(candidate.phase4_updates.get("sizes", []))
            if len(sizes) < 3:
                score += 4.5
                reasons.append("compressed upper-ladder fallback is the only topology variant left standing after research24 and should outrank variance probes")
            if len(sizes) > 3:
                score += 1.5
                reasons.append("broader ladder generalization is the first post-win structural question")
            if int(candidate.phase4_updates.get("steps", 0)) > 112:
                score += 0.5
                reasons.append("longer winner follow-up remains relevant after generalization")
            if int(candidate.phase4_updates.get("num_splits", 0)) > 7:
                score -= 0.25
                reasons.append("variance-only follow-up is lower priority than ladder generalization")
        if hypothesis_counts["kills"] >= 1 and candidate.hypothesis_id == "H1":
            score -= 2.5
            reasons.append("H1 already produced a losing scout; de-prioritize re-anchors")
        if branch_state == "held":
            score -= 2.0
            reasons.append("candidate already matched incumbent without improving it")
        prior_scout_count = sum(1 for row in evidence_rows if row.get("candidate_name") == candidate.name and row.get("stage") == "scout")
        score -= prior_scout_count * 1.5
        if prior_scout_count:
            reasons.append(f"{prior_scout_count} prior scout(s)")
        if incumbent_baselined and phase4_fingerprint(candidate.phase4_updates) == incumbent_fingerprint:
            score -= 4.0
            reasons.append("equivalent to incumbent fingerprint")
        if any(desc.startswith(candidate.name) for desc in prior_runs):
            score -= 0.5
            reasons.append("similar run already exists in artifacts")
        scored.append(CandidateScore(candidate=candidate, score=score, reasons=reasons))
    scored.sort(key=lambda item: item.score, reverse=True)
    return scored


def choose_next_candidate(history_rows: list[dict[str, str]], evidence_rows: list[dict[str, Any]]) -> CandidateScore | None:
    scored = score_candidates(history_rows, evidence_rows)
    return scored[0] if scored else None


def judge_scout(incumbent_phase4_or_score: float | dict[str, Any], scout_phase4: dict[str, Any]) -> tuple[str, float]:
    incumbent_score = (
        answer_score_from_phase4_result(incumbent_phase4_or_score)
        if isinstance(incumbent_phase4_or_score, dict)
        else float(incumbent_phase4_or_score)
    )
    scout_score = answer_score_from_phase4_result(scout_phase4)
    largest_wins = bool(scout_phase4.get("largest_wins", False))
    confidence_margin = float(scout_phase4.get("confidence_margin", 0.0))
    pairwise_win_rate = float(scout_phase4.get("pairwise_win_rate", 0.0))
    if scout_score >= incumbent_score + 0.10 and largest_wins and confidence_margin >= -0.25 and pairwise_win_rate >= 0.55:
        return "promote", scout_score
    if scout_score >= incumbent_score - 0.05:
        return "hold", scout_score
    return "kill", scout_score


def dump_candidate_scores(history_rows: list[dict[str, str]], evidence_rows: list[dict[str, Any]]) -> str:
    return json.dumps(
        [
            {
                "name": item.candidate.name,
                "hypothesis_id": item.candidate.hypothesis_id,
                "score": item.score,
                "reasons": item.reasons,
            }
            for item in score_candidates(history_rows, evidence_rows)
        ],
        indent=2,
    )
