from __future__ import annotations

import importlib
import unittest


class Research16BatchTests(unittest.TestCase):
    def test_research16_runner_defines_expected_families_and_seeds(self) -> None:
        module = importlib.import_module("experiments.run_research16_joint_champion_challenger_batch")
        self.assertEqual(module.FRESH_BLOCK_SEEDS, [91, 92, 93])
        family_names = [family.name for family in module.build_families()]
        self.assertEqual(
            family_names,
            [
                "research16 incumbent control reseed upper ladder",
                "research16 joint champion challenger staged hard upper ladder",
                "research16 joint champion challenger staged smooth upper ladder",
                "research16 joint champion challenger immediate hard upper ladder",
            ],
        )


if __name__ == "__main__":
    unittest.main()
