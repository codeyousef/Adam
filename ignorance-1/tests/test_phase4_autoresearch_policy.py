from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from research.phase4_decision_policy import (
    answer_score_from_phase4_result,
    judge_scout,
    load_hypotheses,
    phase4_fingerprint,
    score_candidates,
)
from research.phase4_search_space import (
    COMPRESSED_UPPER_LADDER_SIZES,
    INCUMBENT_CANDIDATE_NAME,
    rigorous_edge_joint_champion_challenger_compressed_upper_ladder_base,
    rigorous_edge_joint_champion_challenger_staged_hard_base,
)
from research.phase4_evidence import append_evidence, load_evidence


class Phase4AutoresearchPolicyTests(unittest.TestCase):
    def test_answer_score_prefers_better_confidence_and_top_rung_win(self) -> None:
        weak = {
            "confidence_margin": -0.02,
            "pairwise_win_rate": 0.4,
            "largest_wins": False,
            "monotonic_fraction": 0.5,
            "pairwise_margin_std": 0.3,
            "worst_pairwise_margin_ratio": -0.14,
            "best_size": 600_000_000,
            "largest_size": 1_200_000_000,
        }
        strong = {
            "confidence_margin": 0.05,
            "pairwise_win_rate": 0.8,
            "largest_wins": True,
            "monotonic_fraction": 1.0,
            "pairwise_margin_std": 0.05,
            "worst_pairwise_margin_ratio": 0.02,
            "best_size": 1_200_000_000,
            "largest_size": 1_200_000_000,
        }
        self.assertGreater(answer_score_from_phase4_result(strong), answer_score_from_phase4_result(weak))

    def test_answer_score_penalizes_worse_worst_pairwise_margin_and_variance(self) -> None:
        cleaner = {
            "confidence_margin": -0.10,
            "pairwise_win_rate": 0.6,
            "largest_wins": True,
            "monotonic_fraction": 1.0,
            "pairwise_margin_std": 0.10,
            "worst_pairwise_margin_ratio": -0.05,
            "best_size": 1_200_000_000,
            "largest_size": 1_200_000_000,
        }
        noisier = {
            **cleaner,
            "pairwise_margin_std": 0.35,
            "worst_pairwise_margin_ratio": -0.18,
        }
        self.assertGreater(answer_score_from_phase4_result(cleaner), answer_score_from_phase4_result(noisier))

    def test_answer_score_rewards_epistemic_gap_improvement(self) -> None:
        baseline = {
            "confidence_margin": 0.01,
            "pairwise_win_rate": 0.7,
            "largest_wins": True,
            "monotonic_fraction": 1.0,
            "pairwise_margin_std": 0.05,
            "best_size": 1_200_000_000,
            "largest_size": 1_200_000_000,
            "epistemic_gap": 0.10,
            "epistemic_gap_margin": 0.01,
            "epistemic_gap_monotonic_fraction": 0.5,
        }
        stronger_boundary = {
            **baseline,
            "epistemic_gap": 0.35,
            "epistemic_gap_margin": 0.18,
            "epistemic_gap_monotonic_fraction": 1.0,
        }
        self.assertGreater(answer_score_from_phase4_result(stronger_boundary), answer_score_from_phase4_result(baseline))

    def test_answer_score_does_not_penalize_missing_epistemic_metrics(self) -> None:
        legacy = {
            "confidence_margin": -0.02,
            "pairwise_win_rate": 0.5,
            "largest_wins": False,
            "monotonic_fraction": 1.0,
            "pairwise_margin_std": 0.05,
            "best_size": 1_200_000_000,
            "largest_size": 1_200_000_000,
        }
        explicit_neutral = {
            **legacy,
            "epistemic_gap": 0.0,
            "epistemic_gap_margin": 0.0,
            "epistemic_gap_monotonic_fraction": 1.0,
        }
        self.assertEqual(answer_score_from_phase4_result(legacy), answer_score_from_phase4_result(explicit_neutral))

    def test_judge_scout_promotes_clear_improvement(self) -> None:
        incumbent = {
            "confidence_margin": -0.35,
            "pairwise_win_rate": 0.45,
            "largest_wins": False,
            "monotonic_fraction": 1.0,
            "pairwise_margin_std": 0.22,
            "worst_pairwise_margin_ratio": -0.14,
            "best_size": 600_000_000,
            "largest_size": 1_200_000_000,
        }
        scout = {
            "confidence_margin": -0.20,
            "pairwise_win_rate": 0.58,
            "largest_wins": True,
            "monotonic_fraction": 1.0,
            "pairwise_margin_std": 0.15,
            "worst_pairwise_margin_ratio": -0.08,
            "best_size": 1_200_000_000,
            "largest_size": 1_200_000_000,
        }
        decision, score = judge_scout(incumbent, scout)
        self.assertEqual(decision, "promote")
        self.assertGreater(score, answer_score_from_phase4_result(incumbent))

    def test_judge_scout_rejects_winner_flip_without_confidence_improvement(self) -> None:
        incumbent = {
            "confidence_margin": -0.4090,
            "pairwise_win_rate": 0.7143,
            "largest_wins": False,
            "monotonic_fraction": 1.0,
            "pairwise_margin_std": 0.2361,
            "worst_pairwise_margin_ratio": -0.1027,
            "best_size": 300_000_000,
            "largest_size": 1_200_000_000,
        }
        scout = {
            "confidence_margin": -0.4141,
            "pairwise_win_rate": 0.5714,
            "largest_wins": True,
            "monotonic_fraction": 1.0,
            "pairwise_margin_std": 0.3653,
            "worst_pairwise_margin_ratio": -0.1167,
            "best_size": 1_200_000_000,
            "largest_size": 1_200_000_000,
        }
        decision, _ = judge_scout(incumbent, scout)
        self.assertNotEqual(decision, "promote")

    def test_candidate_scoring_prioritizes_active_higher_confidence_hypotheses(self) -> None:
        scored = score_candidates([], [])
        self.assertGreater(len(scored), 0)
        self.assertEqual(scored[0].candidate.name, INCUMBENT_CANDIDATE_NAME)

    def test_candidate_scoring_surfaces_compressed_upper_ladder_followup_after_incumbent_baseline(self) -> None:
        evidence_rows = [
            {
                "candidate_name": INCUMBENT_CANDIDATE_NAME,
                "stage": "scout",
                "decision": "baseline",
                "hypothesis_id": "H1",
            },
        ]
        scored = score_candidates([], evidence_rows)
        names = [item.candidate.name for item in scored[:3]]
        self.assertIn("autoresearch rigorous edge joint champion challenger compressed upper ladder", names)
        self.assertNotIn(INCUMBENT_CANDIDATE_NAME, names)

    def test_candidate_scoring_deprioritizes_incumbent_equivalent_after_baseline(self) -> None:
        evidence_rows = [
            {
                "candidate_name": INCUMBENT_CANDIDATE_NAME,
                "stage": "scout",
                "decision": "baseline",
                "hypothesis_id": "H1",
            },
        ]
        scored = score_candidates([], evidence_rows)
        incumbent_entry = next(item for item in scored if item.candidate.name == INCUMBENT_CANDIDATE_NAME)
        compressed_entry = next(
            item for item in scored if item.candidate.name == "autoresearch rigorous edge joint champion challenger compressed upper ladder"
        )
        self.assertLess(incumbent_entry.score, compressed_entry.score)

    def test_candidate_scoring_prefers_compressed_fallback_after_topology_generalization_failures(self) -> None:
        base_fp = phase4_fingerprint(rigorous_edge_joint_champion_challenger_staged_hard_base())
        history_rows = [
            {"status": "ok", "description": "autoresearch rigorous edge joint champion challenger compressed upper ladder seed106"},
            {"status": "ok", "description": "autoresearch rigorous edge joint champion challenger longer scout seed106"},
        ]
        evidence_rows = [
            {"candidate_name": INCUMBENT_CANDIDATE_NAME, "stage": "scout", "decision": "baseline", "hypothesis_id": "H1"},
            {
                "candidate_name": "autoresearch rigorous edge joint champion challenger longer scout",
                "stage": "scout",
                "decision": "hold",
                "hypothesis_id": "H3",
            },
        ]
        scored = score_candidates(history_rows, evidence_rows)
        self.assertGreater(len(scored), 0)
        self.assertEqual(scored[0].candidate.name, "autoresearch rigorous edge joint champion challenger compressed upper ladder")
        self.assertNotEqual(phase4_fingerprint(scored[0].candidate.phase4_updates), base_fp)

    def test_compressed_upper_ladder_fingerprint_differs_from_upper_ladder_incumbent(self) -> None:
        incumbent = phase4_fingerprint(rigorous_edge_joint_champion_challenger_staged_hard_base())
        compressed = phase4_fingerprint(rigorous_edge_joint_champion_challenger_compressed_upper_ladder_base())
        self.assertNotEqual(incumbent, compressed)
        self.assertEqual(rigorous_edge_joint_champion_challenger_compressed_upper_ladder_base()["sizes"], COMPRESSED_UPPER_LADDER_SIZES)

    def test_phase4_fingerprint_matches_equivalent_dicts(self) -> None:
        left = {"a": 1, "b": 2}
        right = {"b": 2, "a": 1}
        self.assertEqual(phase4_fingerprint(left), phase4_fingerprint(right))

    def test_evidence_log_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "evidence.jsonl"
            append_evidence({"candidate_name": "x", "decision": "hold"}, path=path)
            rows = load_evidence(path=path)
            self.assertEqual(rows, [{"candidate_name": "x", "decision": "hold"}])

    def test_hypotheses_file_loads(self) -> None:
        hypotheses = load_hypotheses()
        self.assertGreaterEqual(len(hypotheses), 4)


if __name__ == "__main__":
    unittest.main()
