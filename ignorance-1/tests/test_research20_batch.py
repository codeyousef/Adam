from __future__ import annotations

import importlib
import unittest


class Research20BatchTests(unittest.TestCase):
    def test_research20_runner_defines_expected_families_and_seeds(self) -> None:
        module = importlib.import_module("experiments.run_research20_behavioral_contracts_batch")
        self.assertEqual(module.FRESH_BLOCK_SEEDS, [107, 108, 109])
        family_names = [family.name for family in module.build_families()]
        self.assertEqual(
            family_names,
            [
                "research20 semantic contrast upper ladder control",
                "research20 behavioral constraints v2 upper ladder",
                "research20 behavioral constraints v2 rigorous edge upper ladder",
                "research20 behavioral constraints v2 adversarial negatives upper ladder",
                "research20 semantic contrast minimal pairs upper ladder",
            ],
        )


if __name__ == "__main__":
    unittest.main()
