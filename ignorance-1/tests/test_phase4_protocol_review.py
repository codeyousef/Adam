from __future__ import annotations

import unittest

from experiments.review_phase4_gate_protocol import summarize_family_block


class Phase4ProtocolReviewTests(unittest.TestCase):
    def test_block_summary_accepts_consistent_top_rung_with_variance_margin(self) -> None:
        summary = summarize_family_block(
            [
                {
                    "largest_wins": True,
                    "pairwise_win_rate": 0.71,
                    "pairwise_margin_std": 0.09,
                    "confidence_margin": -0.08,
                    "largest_margin_ratio": 0.03,
                    "monotonic_fraction": 1.0,
                    "best_size": 1_200_000_000,
                    "competitor_size": 600_000_000,
                },
                {
                    "largest_wins": True,
                    "pairwise_win_rate": 0.67,
                    "pairwise_margin_std": 0.11,
                    "confidence_margin": -0.03,
                    "largest_margin_ratio": 0.025,
                    "monotonic_fraction": 1.0,
                    "best_size": 1_200_000_000,
                    "competitor_size": 600_000_000,
                },
                {
                    "largest_wins": True,
                    "pairwise_win_rate": 0.72,
                    "pairwise_margin_std": 0.10,
                    "confidence_margin": 0.01,
                    "largest_margin_ratio": 0.028,
                    "monotonic_fraction": 1.0,
                    "best_size": 1_200_000_000,
                    "competitor_size": 600_000_000,
                },
            ]
        )
        self.assertTrue(summary["blocked_top_rung_pass"])
        self.assertTrue(summary["variance_aware_pass"])

    def test_block_summary_rejects_mid_rung_collapse_despite_clean_monotonicity(self) -> None:
        summary = summarize_family_block(
            [
                {
                    "largest_wins": False,
                    "pairwise_win_rate": 0.57,
                    "pairwise_margin_std": 0.08,
                    "confidence_margin": -0.11,
                    "largest_margin_ratio": -0.01,
                    "monotonic_fraction": 1.0,
                    "best_size": 150_000_000,
                    "competitor_size": 150_000_000,
                },
                {
                    "largest_wins": False,
                    "pairwise_win_rate": 0.57,
                    "pairwise_margin_std": 0.07,
                    "confidence_margin": -0.10,
                    "largest_margin_ratio": -0.02,
                    "monotonic_fraction": 1.0,
                    "best_size": 300_000_000,
                    "competitor_size": 300_000_000,
                },
                {
                    "largest_wins": False,
                    "pairwise_win_rate": 0.57,
                    "pairwise_margin_std": 0.06,
                    "confidence_margin": -0.09,
                    "largest_margin_ratio": -0.01,
                    "monotonic_fraction": 1.0,
                    "best_size": 600_000_000,
                    "competitor_size": 600_000_000,
                },
            ]
        )
        self.assertFalse(summary["blocked_top_rung_pass"])
        self.assertFalse(summary["variance_aware_pass"])


if __name__ == "__main__":
    unittest.main()
