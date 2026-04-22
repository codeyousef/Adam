from __future__ import annotations

import importlib
import unittest


class Research21BatchTests(unittest.TestCase):
    def test_research21_runner_defines_expected_families_and_seeds(self) -> None:
        module = importlib.import_module("experiments.run_research21_rigorous_edge_followup_batch")
        self.assertEqual(module.FRESH_BLOCK_SEEDS, [110, 111, 112])
        family_names = [family.name for family in module.build_families()]
        self.assertEqual(
            family_names,
            [
                "research21 rigorous edge control upper ladder",
                "research21 rigorous edge joint upper ladder",
                "research21 rigorous edge joint champion challenger staged hard",
                "research21 rigorous edge joint champion challenger plus ranking",
            ],
        )

        families = {family.name: family.updates["phase4"] for family in module.build_families()}
        self.assertEqual(families["research21 rigorous edge control upper ladder"]["phase4_dataset"], "behavioral_constraints_v2_rigorous")
        self.assertFalse(families["research21 rigorous edge control upper ladder"]["phase4_joint_training"])
        self.assertTrue(families["research21 rigorous edge joint upper ladder"]["phase4_joint_training"])
        self.assertAlmostEqual(
            families["research21 rigorous edge joint champion challenger staged hard"]["champion_challenger_weight"],
            0.5,
        )
        self.assertAlmostEqual(
            families["research21 rigorous edge joint champion challenger plus ranking"]["ranking_margin_weight"],
            0.05,
        )


if __name__ == "__main__":
    unittest.main()
