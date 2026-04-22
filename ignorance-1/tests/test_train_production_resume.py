from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from train_production import (
    AttrDict,
    _build_training_pairs,
    _load_resume_state_dict,
    _maybe_promote_completed_tmp_checkpoint,
    _optimizer_name_for_runtime,
    _parse_resume_step_from_tmp_path,
    _prune_allowed_resume_state_dict_keys,
    _resume_state_dict_mismatch_is_allowed,
    _resolve_resume_completed_steps,
    _resolve_training_seed,
    _set_training_seed,
)


class TrainProductionResumeTests(unittest.TestCase):
    def test_parse_resume_step_from_tmp_path_defaults_to_zero_without_suffix(self) -> None:
        self.assertEqual(_parse_resume_step_from_tmp_path("artifacts/model.pt.tmp"), 0)

    def test_parse_resume_step_from_tmp_path_reads_step_suffix(self) -> None:
        self.assertEqual(_parse_resume_step_from_tmp_path("artifacts/model.pt.step0500.tmp"), 500)

    def test_resolve_resume_completed_steps_uses_explicit_override(self) -> None:
        config = AttrDict({"resume_completed_steps": 500})
        self.assertEqual(_resolve_resume_completed_steps(config, 0), 500)

    def test_resolve_resume_completed_steps_falls_back_to_parsed_step(self) -> None:
        config = AttrDict({})
        self.assertEqual(_resolve_resume_completed_steps(config, 1000), 1000)

    def test_resolve_training_seed_prefers_top_level_seed(self) -> None:
        full_config = {"seed": 123, "phase4": {"batch_size": 2}}
        config = AttrDict(full_config["phase4"])

        self.assertEqual(_resolve_training_seed(full_config, config), 123)

    def test_set_training_seed_replays_python_and_torch_rng(self) -> None:
        _set_training_seed(17)
        first_python = random.random()
        first_torch = torch.rand(4)

        _set_training_seed(17)
        second_python = random.random()
        second_torch = torch.rand(4)

        self.assertAlmostEqual(first_python, second_python)
        self.assertTrue(torch.allclose(first_torch, second_torch))

    @patch("train_production.make_benchmark_text_code_pairs", return_value=[("bench", "c3")])
    @patch("train_production.make_text_code_pairs", return_value=[("q1", "c1"), ("q2", "c2"), ("q3", "c3")])
    def test_build_training_pairs_is_stable_after_reseeding(self, *_mocks) -> None:
        _set_training_seed(23)
        first = _build_training_pairs(batch_size=2)

        _set_training_seed(23)
        second = _build_training_pairs(batch_size=2)

        self.assertEqual(first, second)

    def test_load_resume_state_dict_uses_cpu_map_location(self) -> None:
        with patch("train_production.torch.load", return_value={"weight": torch.ones(1)}) as mock_load:
            state = _load_resume_state_dict("artifacts/model.pt.tmp")

        self.assertIn("weight", state)
        mock_load.assert_called_once_with("artifacts/model.pt.tmp", map_location="cpu")

    def test_optimizer_name_for_runtime_falls_back_from_paged_adamw32bit_when_requested(self) -> None:
        config = AttrDict({"optimizer": "paged_adamw32bit", "optimizer_fallback": "adamw8bit"})

        resolved = _optimizer_name_for_runtime(config, force_safe_optimizer=True)

        self.assertEqual(resolved, "adamw8bit")

    def test_optimizer_name_for_runtime_keeps_requested_optimizer_without_safe_flag(self) -> None:
        config = AttrDict({"optimizer": "paged_adamw32bit", "optimizer_fallback": "adamw8bit"})

        resolved = _optimizer_name_for_runtime(config, force_safe_optimizer=False)

        self.assertEqual(resolved, "paged_adamw32bit")

    def test_resume_state_dict_mismatch_allows_missing_retrieval_head_weights(self) -> None:
        self.assertTrue(
            _resume_state_dict_mismatch_is_allowed(
                missing=[
                    "retrieval_head.input_proj.weight",
                    "retrieval_head.input_proj.bias",
                    "retrieval_head.output_proj.weight",
                    "retrieval_head.output_proj.bias",
                ],
                unexpected=[],
            )
        )
        self.assertTrue(
            _resume_state_dict_mismatch_is_allowed(
                missing=[
                    "query_retrieval_facet_head.input_proj.weight",
                    "query_retrieval_facet_head.output_proj.weight",
                    "query_retrieval_facet_head.slot_bias",
                    "code_retrieval_facet_head.input_proj.weight",
                    "code_retrieval_facet_head.output_proj.weight",
                    "code_retrieval_facet_head.slot_bias",
                ],
                unexpected=[],
            )
        )

    def test_resume_state_dict_mismatch_rejects_non_retrieval_missing_keys(self) -> None:
        self.assertFalse(
            _resume_state_dict_mismatch_is_allowed(
                missing=["encoder.proj.0.weight"],
                unexpected=[],
            )
        )
        self.assertFalse(
            _resume_state_dict_mismatch_is_allowed(
                missing=["retrieval_head.input_proj.weight"],
                unexpected=["unexpected.weight"],
            )
        )

    def test_prune_allowed_resume_state_dict_keys_drops_retrieval_shape_mismatches(self) -> None:
        resume_state = {
            "encoder.proj.0.weight": torch.ones(2, 2),
            "retrieval_facet_head.slot_bias": torch.ones(4, 256),
            "retrieval_facet_head.output_proj.weight": torch.ones(1024, 512),
            "retrieval_facet_head.output_proj.bias": torch.ones(1024),
        }
        model_state = {
            "encoder.proj.0.weight": torch.zeros(2, 2),
            "retrieval_facet_head.slot_bias": torch.zeros(6, 256),
            "retrieval_facet_head.output_proj.weight": torch.zeros(1536, 512),
            "retrieval_facet_head.output_proj.bias": torch.zeros(1536),
        }

        pruned, dropped = _prune_allowed_resume_state_dict_keys(resume_state, model_state)

        self.assertIn("encoder.proj.0.weight", pruned)
        self.assertNotIn("retrieval_facet_head.slot_bias", pruned)
        self.assertNotIn("retrieval_facet_head.output_proj.weight", pruned)
        self.assertNotIn("retrieval_facet_head.output_proj.bias", pruned)
        self.assertCountEqual(
            dropped,
            [
                "retrieval_facet_head.slot_bias",
                "retrieval_facet_head.output_proj.weight",
                "retrieval_facet_head.output_proj.bias",
            ],
        )

    def test_prune_allowed_resume_state_dict_keys_drops_shared_facet_keys_when_switching_to_split_heads(self) -> None:
        resume_state = {
            "encoder.proj.0.weight": torch.ones(2, 2),
            "retrieval_facet_head.input_proj.weight": torch.ones(512, 256),
            "retrieval_facet_head.output_proj.weight": torch.ones(1024, 512),
            "retrieval_facet_head.output_proj.bias": torch.ones(1024),
            "retrieval_facet_head.slot_bias": torch.ones(4, 256),
        }
        model_state = {
            "encoder.proj.0.weight": torch.zeros(2, 2),
            "query_retrieval_facet_head.input_proj.weight": torch.zeros(512, 256),
            "query_retrieval_facet_head.output_proj.weight": torch.zeros(1024, 512),
            "query_retrieval_facet_head.output_proj.bias": torch.zeros(1024),
            "query_retrieval_facet_head.slot_bias": torch.zeros(4, 256),
            "code_retrieval_facet_head.input_proj.weight": torch.zeros(512, 256),
            "code_retrieval_facet_head.output_proj.weight": torch.zeros(1024, 512),
            "code_retrieval_facet_head.output_proj.bias": torch.zeros(1024),
            "code_retrieval_facet_head.slot_bias": torch.zeros(4, 256),
        }

        pruned, dropped = _prune_allowed_resume_state_dict_keys(resume_state, model_state)

        self.assertEqual(set(pruned), {"encoder.proj.0.weight"})
        self.assertCountEqual(
            dropped,
            [
                "retrieval_facet_head.input_proj.weight",
                "retrieval_facet_head.output_proj.weight",
                "retrieval_facet_head.output_proj.bias",
                "retrieval_facet_head.slot_bias",
            ],
        )

    def test_promote_completed_tmp_checkpoint_renames_tmp_to_final(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            final_path = root / "model.pt"
            tmp_path = root / "model.pt.step0500.tmp"
            torch.save({"weight": torch.ones(1)}, tmp_path)

            promoted = _maybe_promote_completed_tmp_checkpoint(
                str(final_path),
                completed_steps=500,
                total_steps=500,
            )

            self.assertTrue(promoted)
            self.assertTrue(final_path.exists())
            self.assertFalse(tmp_path.exists())
            state = torch.load(final_path, map_location="cpu")
            self.assertIn("weight", state)

    def test_promote_completed_tmp_checkpoint_does_nothing_for_incomplete_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            final_path = root / "model.pt"
            tmp_path = root / "model.pt.step0500.tmp"
            torch.save({"weight": torch.ones(1)}, tmp_path)

            promoted = _maybe_promote_completed_tmp_checkpoint(
                str(final_path),
                completed_steps=500,
                total_steps=6000,
            )

            self.assertFalse(promoted)
            self.assertFalse(final_path.exists())
            self.assertTrue(tmp_path.exists())


if __name__ == "__main__":
    unittest.main()
