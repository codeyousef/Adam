from __future__ import annotations

import importlib
import unittest


class Research15BatchTests(unittest.TestCase):
    def test_research15_runner_defines_expected_families_and_seeds(self) -> None:
        module = importlib.import_module("experiments.run_research15_champion_challenger_batch")
        self.assertEqual(module.FRESH_BLOCK_SEEDS, [88, 89, 90])
        family_names = [family.name for family in module.build_families()]
        self.assertEqual(
            family_names,
            [
                "research15 incumbent control reseed",
                "research15 champion challenger staged hard",
                "research15 champion challenger staged smooth",
                "research15 champion challenger immediate hard",
            ],
        )


if __name__ == "__main__":
    unittest.main()
