from __future__ import annotations

import tempfile
import textwrap
import unittest
from pathlib import Path

import torch

from src.training.phase4 import _phase4_champion_challenger_loss, _phase4_champion_challenger_weight
from src.utils.config import load_config


class Phase4ChampionChallengerTests(unittest.TestCase):
    def test_champion_challenger_loss_targets_best_nonlargest_competitor(self) -> None:
        champion_loss = torch.tensor(0.55, dtype=torch.float32, requires_grad=True)
        competitor_losses = {
            40_000_000: torch.tensor(0.72, dtype=torch.float32, requires_grad=True),
            150_000_000: torch.tensor(0.60, dtype=torch.float32, requires_grad=True),
            600_000_000: torch.tensor(0.68, dtype=torch.float32, requires_grad=True),
        }

        loss, metadata = _phase4_champion_challenger_loss(
            champion_loss=champion_loss,
            competitor_losses=competitor_losses,
            margin=0.05,
            temperature=0.1,
        )
        loss.backward()

        self.assertEqual(metadata["challenger_size"], 150_000_000)
        self.assertAlmostEqual(metadata["challenger_loss"], 0.60, places=6)
        self.assertIsNotNone(champion_loss.grad)
        self.assertIsNone(competitor_losses[150_000_000].grad)

    def test_champion_challenger_weight_supports_warmup_and_ramp(self) -> None:
        self.assertAlmostEqual(
            _phase4_champion_challenger_weight(0.25, step=0, total_steps=10, start_fraction=0.3, ramp_fraction=0.2),
            0.0,
        )
        self.assertAlmostEqual(
            _phase4_champion_challenger_weight(0.25, step=3, total_steps=10, start_fraction=0.3, ramp_fraction=0.2),
            0.125,
            places=6,
        )
        self.assertAlmostEqual(
            _phase4_champion_challenger_weight(0.25, step=4, total_steps=10, start_fraction=0.3, ramp_fraction=0.2),
            0.25,
            places=6,
        )

    def test_load_config_accepts_phase4_champion_challenger_fields(self) -> None:
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
              sizes: [15000000, 1200000000]
              steps: 8
              batch_size: 2
              lr: 0.0001
              max_vram_gb: 24
              champion_challenger_weight: 0.25
              champion_challenger_margin: 0.05
              champion_challenger_temperature: 0.1
              champion_challenger_start_fraction: 0.3
              champion_challenger_ramp_fraction: 0.2
            """
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.yaml"
            path.write_text(config_text)
            config = load_config(path)

        self.assertAlmostEqual(config.phase4.champion_challenger_weight, 0.25)
        self.assertAlmostEqual(config.phase4.champion_challenger_margin, 0.05)
        self.assertAlmostEqual(config.phase4.champion_challenger_temperature, 0.1)
        self.assertAlmostEqual(config.phase4.champion_challenger_start_fraction, 0.3)
        self.assertAlmostEqual(config.phase4.champion_challenger_ramp_fraction, 0.2)


if __name__ == "__main__":
    unittest.main()
