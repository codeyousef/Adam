from __future__ import annotations

import importlib
import unittest


class Research22BatchTests(unittest.TestCase):
    def test_research22_runner_defines_expected_families_and_seeds(self) -> None:
        module = importlib.import_module("experiments.run_research22_rigorous_edge_replication_batch")
        self.assertEqual(module.FRESH_BLOCK_SEEDS, [113, 114, 115])
        family_names = [family.name for family in module.build_families()]
        self.assertEqual(
            family_names,
            [
                "research22 rigorous edge control upper ladder",
                "research22 rigorous edge joint upper ladder",
                "research22 rigorous edge joint champion challenger staged hard",
            ],
        )

        families = {family.name: family.updates["phase4"] for family in module.build_families()}
        self.assertEqual(families["research22 rigorous edge control upper ladder"]["phase4_dataset"], "behavioral_constraints_v2_rigorous")
        self.assertFalse(families["research22 rigorous edge control upper ladder"]["phase4_joint_training"])
        self.assertTrue(families["research22 rigorous edge joint upper ladder"]["phase4_joint_training"])
        self.assertAlmostEqual(
            families["research22 rigorous edge joint champion challenger staged hard"]["champion_challenger_weight"],
            0.5,
        )
        self.assertAlmostEqual(
            families["research22 rigorous edge joint champion challenger staged hard"]["champion_challenger_start_fraction"],
            0.3,
        )


if __name__ == "__main__":
    unittest.main()
