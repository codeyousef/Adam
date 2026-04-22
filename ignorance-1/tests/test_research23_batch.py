from __future__ import annotations

import importlib
import unittest


class Research23BatchTests(unittest.TestCase):
    def test_research23_runner_defines_expected_families_and_seeds(self) -> None:
        module = importlib.import_module("experiments.run_research23_winner_generalization_batch")
        self.assertEqual(module.FRESH_BLOCK_SEEDS, [116, 117, 118])
        family_names = [family.name for family in module.build_families()]
        self.assertEqual(
            family_names,
            [
                "research23 rigorous edge joint champion challenger incumbent upper ladder",
                "research23 rigorous edge joint champion challenger broader ladder",
                "research23 rigorous edge joint champion challenger longer upper ladder",
            ],
        )

        families = {family.name: family.updates["phase4"] for family in module.build_families()}
        incumbent = families["research23 rigorous edge joint champion challenger incumbent upper ladder"]
        broader = families["research23 rigorous edge joint champion challenger broader ladder"]
        longer = families["research23 rigorous edge joint champion challenger longer upper ladder"]

        self.assertEqual(incumbent["phase4_dataset"], "behavioral_constraints_v2_rigorous")
        self.assertEqual(incumbent["sizes"], [300_000_000, 600_000_000, 1_200_000_000])
        self.assertEqual(broader["sizes"], [150_000_000, 300_000_000, 600_000_000, 1_200_000_000])
        self.assertEqual(longer["steps"], 160)
        self.assertTrue(incumbent["phase4_joint_training"])
        self.assertTrue(broader["phase4_joint_training"])
        self.assertTrue(longer["phase4_joint_training"])
        self.assertAlmostEqual(broader["champion_challenger_weight"], 0.5)
        self.assertAlmostEqual(longer["champion_challenger_start_fraction"], 0.3)


if __name__ == "__main__":
    unittest.main()
