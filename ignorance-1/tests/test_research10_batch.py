from __future__ import annotations

import importlib
import unittest


class Research10BatchTests(unittest.TestCase):
    def test_research10_runner_defines_expected_families_and_seeds(self) -> None:
        module = importlib.import_module("experiments.run_research10_embedding_refinement_batch")
        self.assertEqual(module.FRESH_BLOCK_SEEDS, [73, 74, 75])
        family_names = [family.name for family in module.build_families()]
        self.assertEqual(
            family_names,
            [
                "research10 embedding heavy control",
                "research10 embedding heavy lighter classifier",
                "research10 embedding heavy no pred ignorance",
                "research10 embedding heavy lower mse",
            ],
        )


if __name__ == "__main__":
    unittest.main()
