from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from src.training.phase4 import _phase4_epistemic_boundary_loss
from src.utils.config import load_config


class Phase4EpistemicBoundaryTests(unittest.TestCase):
    def test_epistemic_boundary_loss_matches_softplus_margin_gap(self) -> None:
        coding_logits = torch.tensor([[1.2], [0.7]], dtype=torch.float32)
        ood_logits = torch.tensor([[0.1], [0.4]], dtype=torch.float32)
        loss = _phase4_epistemic_boundary_loss(
            coding_logits=coding_logits,
            ood_logits=ood_logits,
            epistemic_margin=0.5,
        )
        expected = torch.nn.functional.softplus(0.5 - (coding_logits - ood_logits)).mean()
        self.assertTrue(torch.allclose(loss, expected))

    def test_load_config_accepts_epistemic_boundary_fields(self) -> None:
        config_text = """
seed: 42
device: cpu
profile: smoke
phase1:
  embed_dim: 192
  encoder_layers: 2
  encoder_heads: 6
  predictor_layers: 2
  predictor_heads: 6
  lambdas: [0.01]
  projections: 128
  batch_size: 2
  steps: 2
  seq_len: 16
  vocab_size: 128
  patch_size: 8
  lr: 0.001
phase2:
  batch_size: 2
  epochs: 1
  lr: 0.001
  retrieval_k: 1
  answer_threshold: 0.5
  direct_penalty: 0.25
phase3:
  horizon: 2
  num_samples: 2
  num_elites: 1
  num_iterations: 1
  tasks: 1
phase4:
  sizes: [300000000, 600000000, 1200000000]
  steps: 4
  batch_size: 2
  lr: 0.00005
  max_vram_gb: 24.0
  epistemic_boundary_weight: 0.1
  epistemic_margin: 0.3
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.yaml"
            path.write_text(config_text)
            config = load_config(path)
        self.assertAlmostEqual(config.phase4.epistemic_boundary_weight, 0.1)
        self.assertAlmostEqual(config.phase4.epistemic_margin, 0.3)


if __name__ == "__main__":
    unittest.main()
