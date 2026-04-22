from __future__ import annotations

import importlib
import unittest


class Research24BatchTests(unittest.TestCase):
    def test_research24_runner_defines_expected_families_and_seeds(self) -> None:
        module = importlib.import_module("experiments.run_research24_ladder_topology_boundary_batch")
        self.assertEqual(module.FRESH_BLOCK_SEEDS, [119, 120, 121])
        family_names = [family.name for family in module.build_families()]
        self.assertEqual(
            family_names,
            [
                "research24 rigorous edge joint champion challenger incumbent upper ladder",
                "research24 rigorous edge joint champion challenger compressed upper ladder",
                "research24 rigorous edge joint champion challenger expanded upper ladder",
            ],
        )

        families = {family.name: family.updates["phase4"] for family in module.build_families()}
        incumbent = families["research24 rigorous edge joint champion challenger incumbent upper ladder"]
        compressed = families["research24 rigorous edge joint champion challenger compressed upper ladder"]
        expanded = families["research24 rigorous edge joint champion challenger expanded upper ladder"]

        self.assertEqual(incumbent["phase4_dataset"], "behavioral_constraints_v2_rigorous")
        self.assertEqual(incumbent["sizes"], [300_000_000, 600_000_000, 1_200_000_000])
        self.assertEqual(compressed["sizes"], [600_000_000, 1_200_000_000])
        self.assertEqual(expanded["sizes"], [300_000_000, 600_000_000, 900_000_000, 1_200_000_000])
        self.assertEqual(compressed["reference_size"], 600_000_000)
        self.assertEqual(expanded["reference_size"], 300_000_000)
        self.assertTrue(incumbent["phase4_joint_training"])
        self.assertTrue(compressed["phase4_joint_training"])
        self.assertTrue(expanded["phase4_joint_training"])
        self.assertAlmostEqual(incumbent["champion_challenger_weight"], 0.5)
        self.assertAlmostEqual(compressed["champion_challenger_start_fraction"], 0.3)
        self.assertAlmostEqual(expanded["champion_challenger_temperature"], 0.1)


if __name__ == "__main__":
    unittest.main()
