from __future__ import annotations

import importlib
import unittest


class Research25BatchTests(unittest.TestCase):
    def test_research25_runner_defines_expected_families_and_seeds(self) -> None:
        module = importlib.import_module("experiments.run_research25_budget_validation_batch")
        self.assertEqual(module.FRESH_BLOCK_SEEDS, [122, 123, 124])
        family_names = [family.name for family in module.build_families()]
        self.assertEqual(
            family_names,
            [
                "research25 rigorous edge joint champion challenger incumbent upper ladder",
                "research25 rigorous edge joint champion challenger longer upper ladder",
                "research25 rigorous edge joint champion challenger high split upper ladder",
            ],
        )

        families = {family.name: family.updates["phase4"] for family in module.build_families()}
        incumbent = families["research25 rigorous edge joint champion challenger incumbent upper ladder"]
        longer = families["research25 rigorous edge joint champion challenger longer upper ladder"]
        high_split = families["research25 rigorous edge joint champion challenger high split upper ladder"]

        self.assertEqual(incumbent["phase4_dataset"], "behavioral_constraints_v2_rigorous")
        self.assertEqual(incumbent["sizes"], [300_000_000, 600_000_000, 1_200_000_000])
        self.assertEqual(longer["steps"], 160)
        self.assertEqual(high_split["num_splits"], 11)
        self.assertTrue(incumbent["phase4_joint_training"])
        self.assertTrue(longer["phase4_joint_training"])
        self.assertTrue(high_split["phase4_joint_training"])
        self.assertAlmostEqual(incumbent["champion_challenger_weight"], 0.5)
        self.assertAlmostEqual(longer["champion_challenger_start_fraction"], 0.3)
        self.assertAlmostEqual(high_split["champion_challenger_temperature"], 0.1)


if __name__ == "__main__":
    unittest.main()
