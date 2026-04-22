from __future__ import annotations

import unittest

from research.phase4_search_space import (
    COMPRESSED_UPPER_LADDER_SIZES,
    INCUMBENT_CANDIDATE_NAME,
    UPPER_LADDER_SIZES,
    candidate_library,
)


class Phase4CandidateGenerationTests(unittest.TestCase):
    def test_candidate_library_contains_winner_followup_candidates(self) -> None:
        candidates = candidate_library()
        names = [candidate.name for candidate in candidates]
        self.assertIn(INCUMBENT_CANDIDATE_NAME, names)
        self.assertIn("autoresearch rigorous edge joint champion challenger compressed upper ladder", names)
        self.assertIn("autoresearch rigorous edge joint champion challenger longer scout", names)
        self.assertIn("autoresearch rigorous edge joint champion challenger high split scout", names)
        self.assertNotIn("autoresearch rigorous edge joint champion challenger broader ladder", names)
        self.assertNotIn("autoresearch rigorous edge joint champion challenger full ladder", names)
        intervention_types = {candidate.intervention_type for candidate in candidates}
        self.assertEqual(
            intervention_types,
            {"benchmark_strengthening", "allocation_change", "evaluation_stress"},
        )

    def test_compressed_upper_ladder_candidate_is_the_only_topology_followup_after_research24(self) -> None:
        candidates = {candidate.name: candidate for candidate in candidate_library()}
        incumbent = candidates[INCUMBENT_CANDIDATE_NAME]
        compressed = candidates["autoresearch rigorous edge joint champion challenger compressed upper ladder"]

        self.assertEqual(incumbent.phase4_updates["sizes"], UPPER_LADDER_SIZES)
        self.assertEqual(compressed.phase4_updates["sizes"], COMPRESSED_UPPER_LADDER_SIZES)
        self.assertTrue(incumbent.phase4_updates["phase4_joint_training"])
        self.assertTrue(compressed.phase4_updates["phase4_joint_training"])
        self.assertEqual(compressed.hypothesis_id, "H3")

    def test_winner_followup_candidates_keep_rigorous_edge_dataset_and_winner_objective(self) -> None:
        candidates = {candidate.name: candidate for candidate in candidate_library()}
        compressed = candidates["autoresearch rigorous edge joint champion challenger compressed upper ladder"]
        longer = candidates["autoresearch rigorous edge joint champion challenger longer scout"]
        high_split = candidates["autoresearch rigorous edge joint champion challenger high split scout"]

        for candidate in (compressed, longer, high_split):
            self.assertEqual(candidate.phase4_updates["phase4_dataset"], "behavioral_constraints_v2_rigorous")
            self.assertAlmostEqual(candidate.phase4_updates["champion_challenger_weight"], 0.5)
            self.assertAlmostEqual(candidate.phase4_updates["champion_challenger_start_fraction"], 0.3)
            self.assertAlmostEqual(candidate.phase4_updates["champion_challenger_ramp_fraction"], 0.2)

        self.assertEqual(longer.phase4_updates["steps"], 160)
        self.assertEqual(high_split.phase4_updates["num_splits"], 11)


if __name__ == "__main__":
    unittest.main()
