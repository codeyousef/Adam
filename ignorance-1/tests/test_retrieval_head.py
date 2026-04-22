from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import torch

from src.models.jepa import JEPAConfig, JEPAModel


_MODULE_PATH = Path(__file__).resolve().parents[1] / "test_2.7b.py"
_SPEC = importlib.util.spec_from_file_location("strict_eval_test_2_7b", _MODULE_PATH)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


class RetrievalHeadTests(unittest.TestCase):
    def test_retrieval_project_is_identity_when_head_disabled(self) -> None:
        config = JEPAConfig(embed_dim=192)
        model = JEPAModel(config)
        z = torch.randn(2, config.embed_dim)

        projected = model.retrieval_project(z)

        self.assertTrue(torch.allclose(projected, z))

    def test_retrieval_encode_uses_wider_head_when_enabled(self) -> None:
        config = JEPAConfig(
            embed_dim=192,
            encoder_heads=4,
            use_retrieval_head=True,
            retrieval_head_dim=512,
            retrieval_head_hidden_dim=384,
        )
        model = JEPAModel(config)
        input_ids = torch.randint(0, config.vocab_size, (2, 64))

        z_raw = model.encode(input_ids)
        z_retrieval = model.retrieval_encode(input_ids)
        z_pred_raw = model.predict(z_raw, action_id=1)
        z_pred_retrieval = model.retrieval_project(z_pred_raw)
        logits = model.query_logits(z_raw)

        self.assertEqual(tuple(z_raw.shape), (2, 192))
        self.assertEqual(tuple(z_retrieval.shape), (2, 512))
        self.assertEqual(tuple(z_pred_retrieval.shape), (2, 512))
        self.assertEqual(tuple(logits.shape), (2,))

    def test_retrieval_facets_return_fixed_budget_multi_vector_outputs(self) -> None:
        config = JEPAConfig(
            embed_dim=192,
            encoder_heads=4,
            use_retrieval_facets=True,
            retrieval_num_facets=4,
            retrieval_facet_dim=256,
            retrieval_facet_hidden_dim=384,
        )
        model = JEPAModel(config)
        input_ids = torch.randint(0, config.vocab_size, (2, 64))

        z_raw = model.encode(input_ids)
        query_facets = model.retrieval_facets(z_raw, role="query")
        code_facets = model.retrieval_facets(z_raw, role="code")

        self.assertEqual(tuple(query_facets.shape), (2, 4, 256))
        self.assertEqual(tuple(code_facets.shape), (2, 4, 256))

    def test_retrieval_facets_can_use_separate_query_code_projectors(self) -> None:
        config = JEPAConfig(
            embed_dim=192,
            encoder_heads=4,
            use_retrieval_facets=True,
            retrieval_num_facets=4,
            retrieval_facet_dim=128,
            retrieval_facet_hidden_dim=256,
            retrieval_facet_separate_query_code=True,
        )
        model = JEPAModel(config)
        latent = torch.randn(2, config.embed_dim)

        query_facets = model.retrieval_facets(latent, role="query")
        code_facets = model.retrieval_facets(latent, role="code")

        self.assertEqual(tuple(query_facets.shape), (2, 4, 128))
        self.assertEqual(tuple(code_facets.shape), (2, 4, 128))
        self.assertFalse(torch.allclose(query_facets, code_facets))

    def test_infer_retrieval_head_config_from_state_dict(self) -> None:
        config = JEPAConfig(embed_dim=192)
        state_dict = {
            "retrieval_head.input_proj.weight": torch.zeros(384, 192),
            "retrieval_head.input_proj.bias": torch.zeros(384),
            "retrieval_head.output_proj.weight": torch.zeros(512, 384),
            "retrieval_head.output_proj.bias": torch.zeros(512),
        }

        _MODULE._infer_retrieval_head_config(state_dict, config)

        self.assertTrue(config.use_retrieval_head)
        self.assertEqual(config.retrieval_head_hidden_dim, 384)
        self.assertEqual(config.retrieval_head_dim, 512)

    def test_infer_retrieval_facet_config_from_state_dict(self) -> None:
        config = JEPAConfig(embed_dim=192)
        state_dict = {
            "query_retrieval_facet_head.slot_bias": torch.zeros(4, 128),
            "query_retrieval_facet_head.input_proj.weight": torch.zeros(256, 192),
            "code_retrieval_facet_head.slot_bias": torch.zeros(4, 128),
            "code_retrieval_facet_head.input_proj.weight": torch.zeros(256, 192),
        }

        _MODULE._infer_retrieval_facet_config(state_dict, config)

        self.assertTrue(config.use_retrieval_facets)
        self.assertTrue(config.retrieval_facet_separate_query_code)
        self.assertEqual(config.retrieval_num_facets, 4)
        self.assertEqual(config.retrieval_facet_dim, 128)
        self.assertEqual(config.retrieval_facet_hidden_dim, 256)


if __name__ == "__main__":
    unittest.main()
