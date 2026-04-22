from __future__ import annotations

import importlib
import unittest


class Research17BatchTests(unittest.TestCase):
    def test_research17_runner_defines_expected_families_and_seeds(self) -> None:
        module = importlib.import_module("experiments.run_research17_semantic_contrast_batch")
        self.assertEqual(module.FRESH_BLOCK_SEEDS, [94, 95, 96])
        family_names = [family.name for family in module.build_families()]
        self.assertEqual(
            family_names,
            [
                "research17 benchmark control upper ladder",
                "research17 semantic contrast benchmark upper ladder",
                "research17 semantic contrast benchmark joint upper ladder",
            ],
        )


if __name__ == "__main__":
    unittest.main()
