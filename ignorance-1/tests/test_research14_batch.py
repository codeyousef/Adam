from __future__ import annotations

import importlib
import unittest


class Research14BatchTests(unittest.TestCase):
    def test_research14_runner_defines_expected_families_and_seeds(self) -> None:
        module = importlib.import_module("experiments.run_research14_champion_margin_batch")
        self.assertEqual(module.FRESH_BLOCK_SEEDS, [85, 86, 87])
        family_names = [family.name for family in module.build_families()]
        self.assertEqual(
            family_names,
            [
                "research14 incumbent control reseed",
                "research14 staged champion margin hard",
                "research14 staged champion margin smooth focal",
                "research14 immediate champion margin hard",
            ],
        )


if __name__ == "__main__":
    unittest.main()
