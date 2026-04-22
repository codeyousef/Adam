from __future__ import annotations

import tempfile
import textwrap
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from src.losses.alignment import ignorance_penalty
from src.training.phase4 import _phase4_auxiliary_loss
from src.utils.config import load_config


class Phase4ObjectiveWeightsTests(unittest.TestCase):
    def test_auxiliary_loss_respects_configurable_weights(self) -> None:
        z_ood = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        z_ood_pred = torch.tensor([[0.8, 0.2], [0.2, 0.8]], dtype=torch.float32)
        code_candidates = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        coding_logits = torch.tensor([1.5, 0.5], dtype=torch.float32)
        ood_logits = torch.tensor([-0.5, -1.0], dtype=torch.float32)

        result = _phase4_auxiliary_loss(
            z_ood=z_ood,
            z_ood_pred=z_ood_pred,
            code_candidates=code_candidates,
            coding_logits=coding_logits,
            ood_logits=ood_logits,
            ignorance_ood_weight=0.1,
            ignorance_pred_weight=0.3,
            classifier_weight=0.05,
        )

        clf_loss = F.binary_cross_entropy_with_logits(coding_logits, torch.ones_like(coding_logits))
        clf_loss = clf_loss + F.binary_cross_entropy_with_logits(ood_logits, torch.zeros_like(ood_logits))
        expected = (
            0.1 * ignorance_penalty(z_ood, code_candidates)
            + 0.3 * ignorance_penalty(z_ood_pred, code_candidates)
            + 0.05 * clf_loss
        )
        self.assertAlmostEqual(float(result.detach().cpu().item()), float(expected.detach().cpu().item()), places=6)

    def test_auxiliary_loss_can_apply_epistemic_boundary_on_prediction_branch_only(self) -> None:
        z_ood = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        z_ood_pred = torch.tensor([[0.8, 0.2], [0.2, 0.8]], dtype=torch.float32)
        code_candidates = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        coding_logits = torch.tensor([0.1, -0.1], dtype=torch.float32)
        ood_logits = torch.tensor([0.0, -0.2], dtype=torch.float32)
        pred_coding_logits = torch.tensor([0.9, 0.7], dtype=torch.float32)
        pred_ood_logits = torch.tensor([0.2, 0.1], dtype=torch.float32)

        result = _phase4_auxiliary_loss(
            z_ood=z_ood,
            z_ood_pred=z_ood_pred,
            code_candidates=code_candidates,
            coding_logits=coding_logits,
            pred_coding_logits=pred_coding_logits,
            ood_logits=ood_logits,
            pred_ood_logits=pred_ood_logits,
            ignorance_ood_weight=0.0,
            ignorance_pred_weight=0.0,
            classifier_weight=0.0,
            epistemic_boundary_weight=0.5,
            epistemic_margin=0.2,
            epistemic_query_weight=0.0,
            epistemic_prediction_weight=1.0,
        )

        expected = 0.5 * F.softplus(0.2 - (pred_coding_logits - pred_ood_logits)).mean()
        self.assertAlmostEqual(float(result.detach().cpu().item()), float(expected.detach().cpu().item()), places=6)

    def test_auxiliary_loss_can_apply_classifier_head_on_prediction_branch_only(self) -> None:
        z_ood = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        z_ood_pred = torch.tensor([[0.8, 0.2], [0.2, 0.8]], dtype=torch.float32)
        code_candidates = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        coding_logits = torch.tensor([0.1, -0.1], dtype=torch.float32)
        ood_logits = torch.tensor([0.0, -0.2], dtype=torch.float32)
        pred_coding_logits = torch.tensor([0.9, 0.7], dtype=torch.float32)
        pred_ood_logits = torch.tensor([0.2, 0.1], dtype=torch.float32)

        result = _phase4_auxiliary_loss(
            z_ood=z_ood,
            z_ood_pred=z_ood_pred,
            code_candidates=code_candidates,
            coding_logits=coding_logits,
            pred_coding_logits=pred_coding_logits,
            ood_logits=ood_logits,
            pred_ood_logits=pred_ood_logits,
            ignorance_ood_weight=0.0,
            ignorance_pred_weight=0.0,
            classifier_weight=0.5,
            classifier_query_weight=0.0,
            classifier_prediction_weight=1.0,
        )

        expected = 0.5 * (
            F.binary_cross_entropy_with_logits(pred_coding_logits, torch.ones_like(pred_coding_logits))
            + F.binary_cross_entropy_with_logits(pred_ood_logits, torch.zeros_like(pred_ood_logits))
        )
        self.assertAlmostEqual(float(result.detach().cpu().item()), float(expected.detach().cpu().item()), places=6)

    def test_load_config_accepts_phase4_objective_weights(self) -> None:
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
              ignorance_ood_weight: 0.15
              ignorance_pred_weight: 0.05
              classifier_weight: 0.1
              classifier_query_weight: 0.25
              classifier_prediction_weight: 0.75
              epistemic_query_weight: 0.25
              epistemic_prediction_weight: 0.75
            """
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.yaml"
            path.write_text(config_text)
            config = load_config(path)

        self.assertAlmostEqual(config.phase4.ignorance_ood_weight, 0.15)
        self.assertAlmostEqual(config.phase4.ignorance_pred_weight, 0.05)
        self.assertAlmostEqual(config.phase4.classifier_weight, 0.1)
        self.assertAlmostEqual(config.phase4.classifier_query_weight, 0.25)
        self.assertAlmostEqual(config.phase4.classifier_prediction_weight, 0.75)
        self.assertAlmostEqual(config.phase4.epistemic_query_weight, 0.25)
        self.assertAlmostEqual(config.phase4.epistemic_prediction_weight, 0.75)


if __name__ == "__main__":
    unittest.main()
