from __future__ import annotations

import tempfile
import textwrap
import unittest
from pathlib import Path

from src.training.phase4 import _phase4_select_batch_examples
from src.utils.config import load_config
from src.utils.data import Phase4ContrastExample


class Phase4FamilyBalancingTests(unittest.TestCase):
    def _example(self, family: str, index: int) -> Phase4ContrastExample:
        return Phase4ContrastExample(
            prompt=f"prompt {family} {index}",
            code=f"code {family} {index}",
            hard_negatives=[f"neg {family} {index}-{neg}" for neg in range(3)],
            ood_queries=[f"ood {family} {index}"],
            family=family,
        )

    def test_phase4_select_batch_examples_balances_families_round_robin(self) -> None:
        examples = [
            self._example("alpha", 0),
            self._example("beta", 0),
            self._example("gamma", 0),
            self._example("alpha", 1),
            self._example("beta", 1),
            self._example("gamma", 1),
        ]

        batch0 = _phase4_select_batch_examples(examples, batch_size=3, batch_index=0, balance_families=True)
        batch1 = _phase4_select_batch_examples(examples, batch_size=3, batch_index=1, balance_families=True)

        self.assertEqual([example.family for example in batch0], ["alpha", "beta", "gamma"])
        self.assertEqual([example.prompt for example in batch1], ["prompt alpha 1", "prompt beta 1", "prompt gamma 1"])

    def test_phase4_select_batch_examples_preserves_legacy_order_without_balancing(self) -> None:
        examples = [self._example("alpha", 0), self._example("alpha", 1), self._example("beta", 0)]

        batch = _phase4_select_batch_examples(examples, batch_size=2, batch_index=0, balance_families=False)

        self.assertEqual([example.prompt for example in batch], ["prompt alpha 0", "prompt alpha 1"])

    def test_load_config_accepts_phase4_balance_families(self) -> None:
        config_text = textwrap.dedent(
            """
            seed: 42
            device: cpu
            profile: smoke
            phase1:
              embed_dim: 96
              encoder_layers: 2
              encoder_heads: 2
              predictor_layers: 2
              predictor_heads: 2
              lambdas: [0.01]
              projections: 4
              batch_size: 2
              steps: 2
              seq_len: 16
              vocab_size: 128
              patch_size: 4
              lr: 0.001
            phase2:
              batch_size: 2
              epochs: 1
              lr: 0.001
              retrieval_k: 1
              answer_threshold: 0.5
              direct_penalty: 1.0
            phase3:
              horizon: 2
              num_samples: 2
              num_elites: 1
              num_iterations: 1
              tasks: 1
            phase4:
              sizes: [300000000, 600000000, 1200000000]
              steps: 8
              batch_size: 2
              lr: 0.0001
              max_vram_gb: 24
              phase4_dataset: semantic_contrast_v1
              phase4_balance_families: true
            """
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.yaml"
            path.write_text(config_text)
            config = load_config(path)

        self.assertTrue(config.phase4.phase4_balance_families)


if __name__ == "__main__":
    unittest.main()
