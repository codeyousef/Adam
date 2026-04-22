from __future__ import annotations

import random
import tempfile
import textwrap
import unittest
from pathlib import Path

from src.training.phase4 import _build_shared_splits
from src.utils.config import load_config
from src.utils.data import Phase4ContrastExample, make_phase4_contrast_examples, sample_phase4_ood_queries


class Phase4SemanticContrastTests(unittest.TestCase):
    def test_make_phase4_contrast_examples_provide_hard_negatives_and_ood_queries(self) -> None:
        examples = make_phase4_contrast_examples(repeats=1, rng=random.Random(0))

        self.assertGreaterEqual(len(examples), 8)
        self.assertGreaterEqual(len({example.family for example in examples}), 8)
        for example in examples:
            self.assertTrue(example.prompt)
            self.assertTrue(example.code)
            self.assertGreaterEqual(len(example.hard_negatives), 3)
            self.assertGreaterEqual(len(example.ood_queries), 1)
            self.assertTrue(all(hard_negative != example.code for hard_negative in example.hard_negatives))

    def test_behavioral_constraints_v2_examples_use_behavioral_contracts_with_clause_ids(self) -> None:
        examples = make_phase4_contrast_examples(repeats=1, rng=random.Random(0), dataset="behavioral_constraints_v2")

        self.assertGreaterEqual(len(examples), 4)
        for example in examples:
            self.assertIn("Behavioral contract", example.prompt)
            self.assertIn("C1", example.prompt)
            self.assertIn("Edge cases", example.prompt)
            self.assertGreaterEqual(len(example.hard_negatives), 3)
            self.assertTrue(any("contradict" in query.lower() or "underspecified" in query.lower() for query in example.ood_queries))

    def test_behavioral_constraints_v2_rigorous_examples_add_property_style_edge_checks(self) -> None:
        examples = make_phase4_contrast_examples(repeats=1, rng=random.Random(0), dataset="behavioral_constraints_v2_rigorous")

        self.assertGreaterEqual(len(examples), 4)
        self.assertTrue(all("Property-style checks" in example.prompt for example in examples))
        self.assertTrue(any("idempot" in example.prompt.lower() or "invariant" in example.prompt.lower() for example in examples))

    def test_behavioral_constraints_v2_adversarial_examples_include_filtered_near_miss_language(self) -> None:
        examples = make_phase4_contrast_examples(repeats=1, rng=random.Random(0), dataset="behavioral_constraints_v2_adversarial")

        self.assertGreaterEqual(len(examples), 4)
        self.assertTrue(all("Adversarially filtered near misses" in example.prompt for example in examples))
        self.assertTrue(all(len(example.hard_negatives) >= 4 for example in examples))

    def test_semantic_contrast_minimal_pairs_examples_reference_minimal_spec_edits(self) -> None:
        examples = make_phase4_contrast_examples(repeats=1, rng=random.Random(0), dataset="semantic_contrast_minimal_pairs_v1")

        self.assertGreaterEqual(len(examples), 4)
        self.assertTrue(all("Minimal pair variant" in example.prompt for example in examples))
        self.assertTrue(any("single clause" in example.prompt.lower() for example in examples))

    def test_mixed_boundary_curriculum_examples_include_both_adversarial_and_minimal_pair_views(self) -> None:
        examples = make_phase4_contrast_examples(
            repeats=1,
            rng=random.Random(0),
            dataset="behavioral_constraints_v2_mixed_boundary_curriculum_v1",
        )

        self.assertGreaterEqual(len(examples), 8)
        prompts = [example.prompt for example in examples]
        self.assertTrue(any("Adversarially filtered near misses" in prompt for prompt in prompts))
        self.assertTrue(any("Minimal pair variant" in prompt for prompt in prompts))

    def test_taxonomy_coverage_examples_add_factor_mix_and_critic_refinement_language(self) -> None:
        examples = make_phase4_contrast_examples(
            repeats=1,
            rng=random.Random(0),
            dataset="behavioral_constraints_v2_taxonomy_coverage_v1",
        )

        self.assertGreaterEqual(len(examples), 8)
        prompts = [example.prompt for example in examples]
        self.assertTrue(all("Taxonomy coverage plan" in prompt for prompt in prompts))
        self.assertTrue(any("Factor mix" in prompt for prompt in prompts))
        self.assertTrue(any("Critic refinement" in prompt for prompt in prompts))
        self.assertTrue(any("coverage level" in prompt.lower() for prompt in prompts))

    def test_taxonomy_mixed_boundary_curriculum_examples_include_taxonomy_and_boundary_views(self) -> None:
        examples = make_phase4_contrast_examples(
            repeats=1,
            rng=random.Random(0),
            dataset="behavioral_constraints_v2_taxonomy_mixed_boundary_curriculum_v1",
        )

        self.assertGreaterEqual(len(examples), 24)
        prompts = [example.prompt for example in examples]
        self.assertTrue(any("Taxonomy coverage plan" in prompt for prompt in prompts))
        self.assertTrue(any("Adversarially filtered near misses" in prompt for prompt in prompts))
        self.assertTrue(any("Minimal pair variant" in prompt for prompt in prompts))

    def test_build_shared_splits_can_carry_taxonomy_support_discipline_examples(self) -> None:
        splits = _build_shared_splits(
            batch_size=2,
            num_splits=2,
            common_random_numbers=True,
            base_seed=123,
            dataset="behavioral_constraints_v2_taxonomy_support_discipline_v1",
        )

        self.assertEqual(len(splits), 2)
        prompts = [
            example.prompt
            for split in splits
            for example in (split.train_examples or []) + (split.val_examples or [])
        ]
        self.assertTrue(prompts)
        self.assertTrue(any("Taxonomy coverage plan" in prompt for prompt in prompts))
        self.assertTrue(any("Direct support discipline" in prompt for prompt in prompts))
        self.assertTrue(any("Modifier discipline contrast" in prompt for prompt in prompts))

    def test_sample_phase4_ood_queries_can_use_example_specific_prompts(self) -> None:
        example = Phase4ContrastExample(
            prompt="Implement a running total calculator.",
            code="def running_total(values):\n    ...\n",
            hard_negatives=["def running_total(values):\n    return []\n"] * 3,
            ood_queries=["Describe a contradictory TODO list without writing code."],
            family="running_total",
        )

        sampled = sample_phase4_ood_queries(3, examples=[example], rng=random.Random(0))

        self.assertEqual(sampled, ["Describe a contradictory TODO list without writing code."] * 3)

    def test_build_shared_splits_can_carry_semantic_contrast_examples(self) -> None:
        splits = _build_shared_splits(
            batch_size=2,
            num_splits=2,
            common_random_numbers=True,
            base_seed=123,
            dataset="semantic_contrast_v1",
        )

        self.assertEqual(len(splits), 2)
        for split in splits:
            self.assertIsNotNone(split.train_examples)
            self.assertIsNotNone(split.val_examples)
            self.assertTrue(split.train_examples)
            self.assertTrue(split.val_examples)
            self.assertEqual(len(split.train_pairs), len(split.train_examples))
            self.assertEqual(len(split.val_pairs), len(split.val_examples))
            self.assertEqual(split.train_pairs[0], (split.train_examples[0].prompt, split.train_examples[0].code))
            self.assertEqual(split.val_pairs[0], (split.val_examples[0].prompt, split.val_examples[0].code))

    def test_build_shared_splits_can_carry_behavioral_constraints_v2_examples(self) -> None:
        splits = _build_shared_splits(
            batch_size=2,
            num_splits=2,
            common_random_numbers=True,
            base_seed=123,
            dataset="behavioral_constraints_v2",
        )

        self.assertEqual(len(splits), 2)
        for split in splits:
            self.assertIsNotNone(split.train_examples)
            self.assertTrue(split.train_examples)
            self.assertTrue(all("Behavioral contract" in example.prompt for example in split.train_examples))

    def test_load_config_accepts_phase4_dataset_field(self) -> None:
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
              sizes: [300000000, 600000000, 1200000000]
              steps: 8
              batch_size: 2
              lr: 0.0001
              max_vram_gb: 24
              phase4_dataset: semantic_contrast_v1
            """
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.yaml"
            path.write_text(config_text)
            config = load_config(path)

        self.assertEqual(config.phase4.phase4_dataset, "semantic_contrast_v1")

    def test_load_config_accepts_structural_phase4_switches(self) -> None:
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
              sizes: [300000000, 600000000, 1200000000]
              steps: 8
              batch_size: 2
              lr: 0.0001
              max_vram_gb: 24
              phase4_dataset: behavioral_constraints_v1
              phase4_factorized_hard_negatives: true
              phase4_ood_mode: answerability_split_v1
              phase4_prompt_template: evaluator_v1
            """
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.yaml"
            path.write_text(config_text)
            config = load_config(path)

        self.assertEqual(config.phase4.phase4_dataset, "behavioral_constraints_v1")
        self.assertTrue(config.phase4.phase4_factorized_hard_negatives)
        self.assertEqual(config.phase4.phase4_ood_mode, "answerability_split_v1")
        self.assertEqual(config.phase4.phase4_prompt_template, "evaluator_v1")


if __name__ == "__main__":
    unittest.main()
