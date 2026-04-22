from __future__ import annotations

import importlib
import unittest


class ProductionWinningCandidateTests(unittest.TestCase):
    def test_production_config_locks_winning_phase4_recipe(self) -> None:
        module = importlib.import_module("experiments.run_production_winning_candidate")
        cfg = module.build_production_config()
        phase4 = cfg["phase4"]

        self.assertEqual(cfg["profile"], "production_winning_candidate")
        self.assertEqual(phase4["sizes"], [300_000_000, 600_000_000, 1_200_000_000])
        self.assertEqual(phase4["phase4_dataset"], "behavioral_constraints_v2_rigorous")
        self.assertTrue(phase4["phase4_balance_families"])
        self.assertTrue(phase4["phase4_joint_training"])
        self.assertEqual(phase4["reference_size"], 300_000_000)
        self.assertEqual(phase4["grad_accum_steps"], 4)
        self.assertTrue(phase4["common_random_numbers"])
        self.assertAlmostEqual(phase4["champion_challenger_weight"], 0.5)
        self.assertAlmostEqual(phase4["champion_challenger_margin"], 0.05)
        self.assertAlmostEqual(phase4["champion_challenger_temperature"], 0.1)
        self.assertAlmostEqual(phase4["champion_challenger_start_fraction"], 0.3)
        self.assertAlmostEqual(phase4["champion_challenger_ramp_fraction"], 0.2)


if __name__ == "__main__":
    unittest.main()
