from __future__ import annotations

import tempfile
import textwrap
import unittest
from pathlib import Path

import torch

from src.losses.alignment import paired_alignment_loss, retrieval_margin_loss
from src.training.phase4 import _phase4_core_loss
from src.utils.config import load_config


class Phase4RankingMarginTests(unittest.TestCase):
    def test_core_loss_can_add_retrieval_margin_term(self) -> None:
        z_text = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        z_code = torch.tensor([[0.9, 0.1], [0.1, 0.9]], dtype=torch.float32)
        z_pred = torch.tensor([[0.8, 0.2], [0.2, 0.8]], dtype=torch.float32)
        negative_pool = torch.tensor([[0.7, 0.3], [0.3, 0.7]], dtype=torch.float32)

        result = _phase4_core_loss(
            z_text=z_text,
            z_code=z_code,
            z_pred=z_pred,
            negative_pool=negative_pool,
            alignment_prediction_weight=1.0,
            alignment_embedding_weight=0.5,
            alignment_mse_weight=0.25,
            ranking_margin_weight=0.2,
            ranking_margin=0.35,
        )
        base_loss, _ = paired_alignment_loss(
            z_text,
            z_code,
            z_pred,
            negative_pool=negative_pool,
            prediction_weight=1.0,
            embedding_weight=0.5,
            mse_weight=0.25,
        )
        expected_margin = 0.5 * (
            retrieval_margin_loss(z_pred, z_code, negative_pool=negative_pool, margin=0.35)
            + retrieval_margin_loss(z_text, z_code, negative_pool=negative_pool, margin=0.35)
        )
        expected = base_loss + 0.2 * expected_margin
        self.assertAlmostEqual(float(result.detach().cpu().item()), float(expected.detach().cpu().item()), places=6)

    def test_load_config_accepts_phase4_ranking_margin_fields(self) -> None:
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
              sizes: [15000000]
              steps: 8
              batch_size: 2
              lr: 0.0001
              max_vram_gb: 24
              ranking_margin_weight: 0.2
              ranking_margin: 0.35
            """
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.yaml"
            path.write_text(config_text)
            config = load_config(path)

        self.assertAlmostEqual(config.phase4.ranking_margin_weight, 0.2)
        self.assertAlmostEqual(config.phase4.ranking_margin, 0.35)


if __name__ == "__main__":
    unittest.main()
