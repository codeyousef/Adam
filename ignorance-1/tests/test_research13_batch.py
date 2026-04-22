from __future__ import annotations

import importlib
import unittest


class Research13BatchTests(unittest.TestCase):
    def test_research13_runner_defines_expected_families_and_seeds(self) -> None:
        module = importlib.import_module("experiments.run_research13_ranking_margin_batch")
        self.assertEqual(module.FRESH_BLOCK_SEEDS, [82, 83, 84])
        family_names = [family.name for family in module.build_families()]
        self.assertEqual(
            family_names,
            [
                "research13 incumbent control reseed",
                "research13 ranking margin light",
                "research13 ranking margin medium",
                "research13 ranking margin medium no pred ignorance light classifier",
                "research13 ranking margin medium production mix",
            ],
        )


if __name__ == "__main__":
    unittest.main()
