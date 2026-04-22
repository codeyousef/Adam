from __future__ import annotations

import importlib
import unittest


class Research11BatchTests(unittest.TestCase):
    def test_research11_runner_defines_expected_families_and_seeds(self) -> None:
        module = importlib.import_module("experiments.run_research11_compute_fair_batch")
        self.assertEqual(module.FRESH_BLOCK_SEEDS, [76, 77, 78])
        family_names = [family.name for family in module.build_families()]
        self.assertEqual(
            family_names,
            [
                "research11 baseline incumbent control",
                "research11 compute fair incumbent control",
                "research11 compute fair simple core",
                "research11 compute fair simple core embedding heavy",
            ],
        )


if __name__ == "__main__":
    unittest.main()
