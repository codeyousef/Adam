from __future__ import annotations

import tempfile
import textwrap
import unittest
from pathlib import Path

from src.training.phase4 import _lr_multiplier
from src.utils.config import load_config


class Phase4WSDTests(unittest.TestCase):
    def test_wsd_scheduler_has_plateau_then_linear_cooldown(self) -> None:
        total_steps = 100
        warm_value = _lr_multiplier(
            4,
            total_steps,
            scheduler="wsd",
            warmup_fraction=0.1,
            min_lr_ratio=0.0,
            wsd_decay_start_fraction=0.8,
            wsd_cooldown_shape="linear",
        )
        plateau_value = _lr_multiplier(
            50,
            total_steps,
            scheduler="wsd",
            warmup_fraction=0.1,
            min_lr_ratio=0.0,
            wsd_decay_start_fraction=0.8,
            wsd_cooldown_shape="linear",
        )
        cooldown_value = _lr_multiplier(
            90,
            total_steps,
            scheduler="wsd",
            warmup_fraction=0.1,
            min_lr_ratio=0.0,
            wsd_decay_start_fraction=0.8,
            wsd_cooldown_shape="linear",
        )

        self.assertLess(warm_value, 1.0)
        self.assertAlmostEqual(plateau_value, 1.0)
        self.assertLess(cooldown_value, plateau_value)
        self.assertGreater(cooldown_value, 0.0)

    def test_load_config_accepts_wsd_phase4_fields(self) -> None:
        config_text = textwrap.dedent(
            """
            seed: 42
            device: cpu
            phase1:
              embed_dim: 192
              encoder_layers: 4
              encoder_heads: 6
              predictor_layers: 4
              predictor_heads: 6
              lambdas: [0.1]
              projections: 64
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
              answer_threshold: 0.2
              direct_penalty: 0.2
            phase3:
              horizon: 2
              num_samples: 4
              num_elites: 2
              num_iterations: 1
              tasks: 1
            phase4:
              sizes: [15000000]
              steps: 8
              batch_size: 2
              lr: 0.0001
              max_vram_gb: 1.0
              scheduler: wsd
              warmup_fraction: 0.1
              min_lr_ratio: 0.0
              wsd_decay_start_fraction: 0.8
              wsd_cooldown_shape: linear
            """
        ).strip()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.yaml"
            path.write_text(config_text)
            config = load_config(path)

        self.assertEqual(config.phase4.scheduler, "wsd")
        self.assertAlmostEqual(config.phase4.wsd_decay_start_fraction, 0.8)
        self.assertEqual(config.phase4.wsd_cooldown_shape, "linear")


if __name__ == "__main__":
    unittest.main()
