from __future__ import annotations

import unittest

import torch

from src.losses.alignment import prototype_alignment_loss


class TargetAnchorV13Tests(unittest.TestCase):
    def test_code_target_prototypes_can_be_optimized_without_query_prototype_weight(self) -> None:
        code = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        pred = torch.tensor([[0.9, 0.1], [0.1, 0.9]], dtype=torch.float32)
        prototypes = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        labels = torch.tensor([0, 1], dtype=torch.long)

        code_loss, _ = prototype_alignment_loss(code, prototypes, labels, temperature=0.07)
        pred_loss, _ = prototype_alignment_loss(pred, prototypes, labels, temperature=0.07)
        total_loss = 0.10 * code_loss + 0.08 * pred_loss

        self.assertGreater(float(total_loss), 0.0)
        self.assertLess(float(code_loss), 0.3)
        self.assertLess(float(pred_loss), 0.4)


if __name__ == "__main__":
    unittest.main()
