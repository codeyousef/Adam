from __future__ import annotations

import importlib
import unittest


class Research18BatchTests(unittest.TestCase):
    def test_research18_runner_defines_expected_families_and_seeds(self) -> None:
        module = importlib.import_module("experiments.run_research18_semantic_contrast_overnight_batch")
        self.assertEqual(module.FRESH_BLOCK_SEEDS, [97, 98, 99])
        family_names = [family.name for family in module.build_families()]
        self.assertEqual(
            family_names,
            [
                "research18 benchmark control upper ladder",
                "research18 semantic contrast upper ladder",
                "research18 semantic contrast balanced upper ladder",
                "research18 semantic contrast joint upper ladder",
                "research18 semantic contrast balanced joint upper ladder",
                "research18 semantic contrast balanced upper ladder ranking light",
                "research18 semantic contrast balanced joint upper ladder ranking light",
                "research18 semantic contrast balanced joint upper ladder champion challenger staged hard",
                "research18 semantic contrast balanced joint upper ladder champion challenger staged smooth",
                "research18 semantic contrast balanced joint upper ladder champion challenger immediate hard",
                "research18 semantic contrast balanced joint upper ladder champion challenger plus ranking",
            ],
        )


if __name__ == "__main__":
    unittest.main()
