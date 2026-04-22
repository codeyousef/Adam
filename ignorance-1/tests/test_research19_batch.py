from __future__ import annotations

import importlib
import unittest


class Research19BatchTests(unittest.TestCase):
    def test_research19_runner_defines_expected_families_and_seeds(self) -> None:
        module = importlib.import_module("experiments.run_research19_semantic_contrast_deconfounded_batch")
        self.assertEqual(module.FRESH_BLOCK_SEEDS, [100, 101, 102])
        family_names = [family.name for family in module.build_families()]
        self.assertEqual(
            family_names,
            [
                "research19 benchmark control upper ladder",
                "research19 semantic contrast upper ladder replicate",
                "research19 semantic contrast upper ladder ranking light",
                "research19 semantic contrast upper ladder ranking medium",
                "research19 semantic contrast upper ladder champion challenger staged smooth",
                "research19 semantic contrast upper ladder champion challenger staged smooth plus ranking",
            ],
        )


if __name__ == "__main__":
    unittest.main()
