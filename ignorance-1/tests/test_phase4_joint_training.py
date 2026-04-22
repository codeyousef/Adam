from __future__ import annotations

import tempfile
import textwrap
import unittest
from pathlib import Path

import torch

from src.training.phase4 import _ladder_update_mask, _phase4_champion_challenger_loss
from src.utils.config import load_config


class Phase4JointTrainingTests(unittest.TestCase):
    def test_ladder_update_mask_hits_target_steps(self) -> None:
        mask = _ladder_update_mask(total_ladder_steps=7, target_updates=3)
        self.assertEqual(sum(1 for enabled in mask if enabled), 3)
        self.assertEqual(len(mask), 7)
        self.assertTrue(mask[0])
        self.assertTrue(mask[-1])

    def test_joint_training_config_fields_load(self) -> None:
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
              sizes: [600000000, 1200000000]
              steps: 8
              batch_size: 2
              lr: 0.0001
              max_vram_gb: 24
              phase4_joint_training: true
              champion_challenger_weight: 0.5
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

        self.assertTrue(config.phase4.phase4_joint_training)
        self.assertAlmostEqual(config.phase4.champion_challenger_weight, 0.5)

    def test_champion_challenger_loss_detaches_selected_challenger(self) -> None:
        champion_loss = torch.tensor(0.7, dtype=torch.float32, requires_grad=True)
        competitor_losses = {
            600_000_000: torch.tensor(0.62, dtype=torch.float32, requires_grad=True),
            300_000_000: torch.tensor(0.68, dtype=torch.float32, requires_grad=True),
        }
        loss, metadata = _phase4_champion_challenger_loss(
            champion_loss=champion_loss,
            competitor_losses=competitor_losses,
            margin=0.05,
            temperature=0.1,
        )
        loss.backward()
        self.assertEqual(metadata["challenger_size"], 600_000_000)
        self.assertIsNotNone(champion_loss.grad)
        self.assertIsNone(competitor_losses[600_000_000].grad)


if __name__ == "__main__":
    unittest.main()
