from __future__ import annotations

import importlib
import unittest


class Research12BatchTests(unittest.TestCase):
    def test_research12_runner_defines_expected_families_and_seeds(self) -> None:
        module = importlib.import_module("experiments.run_research12_structural_batch")
        self.assertEqual(module.FRESH_BLOCK_SEEDS, [79, 80, 81])
        family_names = [family.name for family in module.build_families()]
        self.assertEqual(
            family_names,
            [
                "research12 incumbent control reseed",
                "research12 alignment production mix retry",
                "research12 alignment production mix lite",
                "research12 alignment prediction heavy moderated",
                "research12 alignment production mix plus lighter aux",
                "research12 control no pred ignorance light classifier",
                "research12 anchor300m flatter scaling",
            ],
        )


if __name__ == "__main__":
    unittest.main()
