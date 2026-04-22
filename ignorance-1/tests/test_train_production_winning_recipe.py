from __future__ import annotations

from pathlib import Path
import unittest

import yaml

from train_production import (
    AttrDict,
    _resolve_phase4_training_repeats,
    _resolve_production_phase4_settings,
    _resolve_production_training_budget,
)
from src.training.phase4 import _proxy_config
from src.models.jepa import approximate_model_params
from src.utils.config import load_config


class TrainProductionWinningRecipeTests(unittest.TestCase):
    def test_resolves_phase4_aliases_for_winning_recipe(self) -> None:
        config = AttrDict(
            {
                "phase4_dataset": "behavioral_constraints_v2_rigorous",
                "phase4_balance_families": True,
                "ignorance_ood_weight": 0.2,
                "ignorance_pred_weight": 0.2,
                "classifier_weight": 0.25,
                "alignment_prediction_weight": 1.0,
                "alignment_embedding_weight": 0.5,
                "alignment_mse_weight": 0.25,
                "ranking_margin_weight": 0.0,
                "ranking_margin": 0.2,
            }
        )

        resolved = _resolve_production_phase4_settings(config)

        self.assertEqual(resolved["phase4_dataset"], "behavioral_constraints_v2_rigorous")
        self.assertTrue(resolved["phase4_balance_families"])
        self.assertTrue(resolved["use_phase4_contrast_data"])
        self.assertAlmostEqual(resolved["ood_weight"], 0.2)
        self.assertAlmostEqual(resolved["pred_ood_weight"], 0.2)
        self.assertAlmostEqual(resolved["clf_weight"], 0.25)
        self.assertAlmostEqual(resolved["alignment_prediction_weight"], 1.0)
        self.assertAlmostEqual(resolved["alignment_embedding_weight"], 0.5)
        self.assertAlmostEqual(resolved["alignment_mse_weight"], 0.25)
        self.assertTrue(resolved["alignment_symmetric"])
        self.assertFalse(resolved["alignment_decoupled"])
        self.assertAlmostEqual(resolved["retrieval_margin_weight"], 0.0)
        self.assertAlmostEqual(resolved["retrieval_margin"], 0.2)

    def test_resolves_alignment_decoupled_flag_when_requested(self) -> None:
        config = AttrDict({"alignment_symmetric": False, "alignment_decoupled": True})

        resolved = _resolve_production_phase4_settings(config)

        self.assertFalse(resolved["alignment_symmetric"])
        self.assertTrue(resolved["alignment_decoupled"])

    def test_resolves_ranking_controls_when_requested(self) -> None:
        config = AttrDict(
            {
                "retrieval_margin_prediction_weight": 1.0,
                "retrieval_margin_embedding_weight": 0.0,
                "ranking_margin_weight": 0.12,
                "ranking_margin": 0.24,
                "ranking_focal_gamma": 1.5,
                "ranking_prediction_weight": 1.0,
                "ranking_embedding_weight": 0.0,
                "ranking_start_fraction": 0.3,
                "ranking_ramp_fraction": 0.2,
                "ranking_largest_only": False,
            }
        )

        resolved = _resolve_production_phase4_settings(config)

        self.assertAlmostEqual(resolved["retrieval_margin_prediction_weight"], 1.0)
        self.assertAlmostEqual(resolved["retrieval_margin_embedding_weight"], 0.0)
        self.assertAlmostEqual(resolved["ranking_margin_weight"], 0.12)
        self.assertAlmostEqual(resolved["ranking_margin"], 0.24)
        self.assertAlmostEqual(resolved["ranking_focal_gamma"], 1.5)
        self.assertAlmostEqual(resolved["ranking_prediction_weight"], 1.0)
        self.assertAlmostEqual(resolved["ranking_embedding_weight"], 0.0)
        self.assertAlmostEqual(resolved["ranking_start_fraction"], 0.3)
        self.assertAlmostEqual(resolved["ranking_ramp_fraction"], 0.2)
        self.assertFalse(resolved["ranking_largest_only"])

    def test_resolves_epistemic_boundary_controls_when_requested(self) -> None:
        config = AttrDict({"epistemic_boundary_weight": 0.06, "epistemic_margin": 0.28})

        resolved = _resolve_production_phase4_settings(config)

        self.assertAlmostEqual(resolved["epistemic_boundary_weight"], 0.06)
        self.assertAlmostEqual(resolved["epistemic_margin"], 0.28)

    def test_resolves_prediction_primary_classifier_controls_when_requested(self) -> None:
        config = AttrDict({"clf_weight": 0.06, "classifier_query_weight": 0.0, "classifier_prediction_weight": 1.0})

        resolved = _resolve_production_phase4_settings(config)

        self.assertAlmostEqual(resolved["clf_weight"], 0.06)
        self.assertAlmostEqual(resolved["classifier_query_weight"], 0.0)
        self.assertAlmostEqual(resolved["classifier_prediction_weight"], 1.0)

    def test_resolves_support_slate_localization_controls_when_requested(self) -> None:
        config = AttrDict(
            {
                "support_slate_localization_weight": 0.10,
                "support_slate_prediction_weight": 0.35,
                "support_slate_same_family_weight": 2.5,
                "support_slate_cross_family_weight": 1.25,
                "support_slate_temperature": 0.08,
                "support_slate_margin_weight": 0.05,
                "support_slate_margin": 0.08,
                "support_slate_cross_family_negatives": 4,
            }
        )

        resolved = _resolve_production_phase4_settings(config)

        self.assertAlmostEqual(resolved["support_slate_localization_weight"], 0.10)
        self.assertAlmostEqual(resolved["support_slate_prediction_weight"], 0.35)
        self.assertAlmostEqual(resolved["support_slate_same_family_weight"], 2.5)
        self.assertAlmostEqual(resolved["support_slate_cross_family_weight"], 1.25)
        self.assertAlmostEqual(resolved["support_slate_temperature"], 0.08)
        self.assertAlmostEqual(resolved["support_slate_margin_weight"], 0.05)
        self.assertAlmostEqual(resolved["support_slate_margin"], 0.08)
        self.assertEqual(resolved["support_slate_cross_family_negatives"], 4)

    def test_production_mode_requires_explicit_steps(self) -> None:
        config = AttrDict(
            {
                "sizes": [2_700_000_000],
                "steps": 112,
                "batch_size": 4,
                "lr": 0.00005,
                "production_mode": True,
            }
        )

        with self.assertRaises(ValueError):
            _resolve_production_training_budget(config, 2_700_000_000)

    def test_production_mode_uses_explicit_training_budget_instead_of_proxy_scaling(self) -> None:
        config = AttrDict(
            {
                "sizes": [2_700_000_000],
                "steps": 112,
                "batch_size": 4,
                "lr": 0.00005,
                "reference_size": 300_000_000,
                "step_scale_power": 0.55,
                "max_step_multiplier": 5.0,
                "lr_scale_power": 0.2,
                "max_lr_divisor": 2.5,
                "production_mode": True,
                "production_steps": 6000,
            }
        )

        budget = _resolve_production_training_budget(config, 2_700_000_000)

        self.assertEqual(budget["steps"], 6000)
        self.assertAlmostEqual(budget["lr"], 0.00005)
        self.assertAlmostEqual(budget["step_multiplier"], 1.0)
        self.assertAlmostEqual(budget["lr_divisor"], 1.0)
        self.assertEqual(budget["source"], "explicit_production_steps")

    def test_production_mode_uses_explicit_phase4_repeat_budget(self) -> None:
        config = AttrDict(
            {
                "batch_size": 4,
                "production_mode": True,
                "production_phase4_repeats": 4096,
            }
        )

        repeats = _resolve_phase4_training_repeats(config)

        self.assertEqual(repeats, 4096)

    def test_production_winning_2_7b_config_targets_real_2_7b_model(self) -> None:
        cfg = load_config("config/production_winning_2_7b.yaml")

        self.assertEqual(cfg.phase4.sizes, [2_700_000_000])
        self.assertEqual(cfg.phase4.proxy_recipe, "v6_overnight")
        self.assertEqual(cfg.phase4.phase4_dataset, "behavioral_constraints_v2_rigorous")
        self.assertTrue(cfg.phase4.phase4_balance_families)
        self.assertTrue(cfg.phase4.production_mode)
        self.assertEqual(cfg.phase4.production_steps, 6000)
        self.assertEqual(cfg.phase4.production_phase4_repeats, 4096)

        model_cfg = _proxy_config(cfg.phase4.sizes[0], cfg.phase4.proxy_recipe)
        approx_params = approximate_model_params(model_cfg)
        self.assertGreaterEqual(approx_params, 2_500_000_000)

    def test_production_winning_2_7b_config_embeds_retrieval_evidence_strict_eval_defaults(self) -> None:
        data = yaml.safe_load(Path("config/production_winning_2_7b.yaml").read_text())
        strict_eval = data["strict_eval"]

        self.assertAlmostEqual(data["confidence_threshold"], 0.312)
        self.assertAlmostEqual(data["lexical_weight"], 0.6)
        self.assertEqual(strict_eval["confidence_mode"], "neighborhood_posterior")
        self.assertEqual(strict_eval["confidence_support_topk"], 5)
        self.assertAlmostEqual(strict_eval["confidence_support_temperature"], 0.1)
        self.assertEqual(strict_eval["rerank_topk"], 5)
        self.assertEqual(strict_eval["rerank_shortlist_mode"], "pred_query_union_local")

    def test_v6_overnight_7b_config_differs_from_2p7b(self) -> None:
        cfg_2p7 = _proxy_config(2_700_000_000, "v6_overnight")
        cfg_7b = _proxy_config(7_000_000_000, "v6_overnight")

        self.assertNotEqual(cfg_7b, cfg_2p7)

    def test_v6_overnight_7b_param_count_exceeds_2p7b(self) -> None:
        cfg_2p7 = _proxy_config(2_700_000_000, "v6_overnight")
        cfg_7b = _proxy_config(7_000_000_000, "v6_overnight")
        params_2p7 = approximate_model_params(cfg_2p7)
        params_7b = approximate_model_params(cfg_7b)

        self.assertGreater(params_7b, params_2p7)
        self.assertGreater(params_7b, 5_500_000_000)

    def test_default_model_brand_is_sinai(self) -> None:
        from src.models.jepa import MODEL_BRAND_NAME

        self.assertEqual(MODEL_BRAND_NAME, "Sinai")


if __name__ == "__main__":
    unittest.main()
