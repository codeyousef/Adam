from __future__ import annotations

import tempfile
import textwrap
import unittest
from pathlib import Path

import torch

from src.losses.alignment import retrieval_margin_loss
from src.training.phase4 import _phase4_core_loss, _phase4_ranking_loss, _phase4_ranking_weight
from src.utils.config import load_config


class Phase4StagedRankingTests(unittest.TestCase):
    def test_ranking_weight_supports_warmup_and_ramp(self) -> None:
        self.assertAlmostEqual(_phase4_ranking_weight(0.2, step=0, total_steps=10, start_fraction=0.3, ramp_fraction=0.2), 0.0)
        self.assertAlmostEqual(_phase4_ranking_weight(0.2, step=2, total_steps=10, start_fraction=0.3, ramp_fraction=0.2), 0.0)
        self.assertAlmostEqual(_phase4_ranking_weight(0.2, step=3, total_steps=10, start_fraction=0.3, ramp_fraction=0.2), 0.1, places=6)
        self.assertAlmostEqual(_phase4_ranking_weight(0.2, step=4, total_steps=10, start_fraction=0.3, ramp_fraction=0.2), 0.2, places=6)
        self.assertAlmostEqual(_phase4_ranking_weight(0.2, step=9, total_steps=10, start_fraction=0.3, ramp_fraction=0.2), 0.2, places=6)

    def test_ranking_loss_can_apply_focal_hardness_weighting(self) -> None:
        z_text = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        z_code = torch.tensor([[0.7, 0.3], [0.3, 0.7]], dtype=torch.float32)
        z_pred = torch.tensor([[0.6, 0.4], [0.4, 0.6]], dtype=torch.float32)
        negative_pool = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)

        baseline = _phase4_ranking_loss(
            z_text=z_text,
            z_code=z_code,
            z_pred=z_pred,
            negative_pool=negative_pool,
            ranking_margin=0.35,
            ranking_effective_weight=0.2,
            ranking_focal_gamma=0.0,
        )
        focal = _phase4_ranking_loss(
            z_text=z_text,
            z_code=z_code,
            z_pred=z_pred,
            negative_pool=negative_pool,
            ranking_margin=0.35,
            ranking_effective_weight=0.2,
            ranking_focal_gamma=2.0,
        )

        raw = 0.5 * (
            retrieval_margin_loss(z_pred, z_code, negative_pool=negative_pool, margin=0.35)
            + retrieval_margin_loss(z_text, z_code, negative_pool=negative_pool, margin=0.35)
        )
        expected_baseline = 0.2 * raw
        self.assertAlmostEqual(float(baseline.detach().cpu().item()), float(expected_baseline.detach().cpu().item()), places=6)
        self.assertGreater(float(focal.detach().cpu().item()), float(baseline.detach().cpu().item()))

    def test_ranking_loss_can_disable_embedding_side_term(self) -> None:
        z_text = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        z_code = torch.tensor([[0.7, 0.3], [0.3, 0.7]], dtype=torch.float32)
        z_pred = torch.tensor([[0.6, 0.4], [0.4, 0.6]], dtype=torch.float32)
        negative_pool = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)

        predictor_only = _phase4_ranking_loss(
            z_text=z_text,
            z_code=z_code,
            z_pred=z_pred,
            negative_pool=negative_pool,
            ranking_margin=0.35,
            ranking_effective_weight=0.2,
            ranking_focal_gamma=0.0,
            ranking_prediction_weight=1.0,
            ranking_embedding_weight=0.0,
        )

        expected = 0.2 * retrieval_margin_loss(z_pred, z_code, negative_pool=negative_pool, margin=0.35)
        self.assertAlmostEqual(float(predictor_only.detach().cpu().item()), float(expected.detach().cpu().item()), places=6)

    def test_core_loss_can_use_explicit_effective_weight(self) -> None:
        z_text = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        z_code = torch.tensor([[0.7, 0.3], [0.3, 0.7]], dtype=torch.float32)
        z_pred = torch.tensor([[0.6, 0.4], [0.4, 0.6]], dtype=torch.float32)
        negative_pool = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)

        without_ranking = _phase4_core_loss(
            z_text=z_text,
            z_code=z_code,
            z_pred=z_pred,
            negative_pool=negative_pool,
            alignment_prediction_weight=1.0,
            alignment_embedding_weight=0.5,
            alignment_mse_weight=0.25,
            ranking_margin_weight=0.2,
            ranking_margin=0.35,
            ranking_effective_weight=0.0,
            ranking_focal_gamma=0.0,
        )
        with_ranking = _phase4_core_loss(
            z_text=z_text,
            z_code=z_code,
            z_pred=z_pred,
            negative_pool=negative_pool,
            alignment_prediction_weight=1.0,
            alignment_embedding_weight=0.5,
            alignment_mse_weight=0.25,
            ranking_margin_weight=0.2,
            ranking_margin=0.35,
            ranking_effective_weight=0.2,
            ranking_focal_gamma=0.0,
        )
        self.assertGreater(float(with_ranking.detach().cpu().item()), float(without_ranking.detach().cpu().item()))

    def test_load_config_accepts_phase4_staged_ranking_fields(self) -> None:
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
              ranking_focal_gamma: 2.0
              ranking_prediction_weight: 1.0
              ranking_embedding_weight: 0.0
              ranking_start_fraction: 0.3
              ranking_ramp_fraction: 0.2
              ranking_largest_only: true
            """
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.yaml"
            path.write_text(config_text)
            config = load_config(path)

        self.assertAlmostEqual(config.phase4.ranking_focal_gamma, 2.0)
        self.assertAlmostEqual(config.phase4.ranking_prediction_weight, 1.0)
        self.assertAlmostEqual(config.phase4.ranking_embedding_weight, 0.0)
        self.assertAlmostEqual(config.phase4.ranking_start_fraction, 0.3)
        self.assertAlmostEqual(config.phase4.ranking_ramp_fraction, 0.2)
        self.assertTrue(config.phase4.ranking_largest_only)


if __name__ == "__main__":
    unittest.main()
