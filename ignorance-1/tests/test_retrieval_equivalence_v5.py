from __future__ import annotations

import unittest

import torch

from src.utils.data import make_phase4_contrast_examples, make_retrieval_training_examples


class RetrievalEquivalenceV5Tests(unittest.TestCase):
    def test_retrieval_examples_include_equivalence_metadata(self) -> None:
        examples = make_retrieval_training_examples(repeats=8, benchmark_repeats=4)
        self.assertGreater(len(examples), 0)
        sample = examples[0]
        self.assertTrue(hasattr(sample, "equivalence_id"))
        self.assertTrue(hasattr(sample, "alternate_queries"))
        self.assertTrue(hasattr(sample, "synthesis_queries"))

    def test_retrieval_examples_group_paraphrases_under_shared_equivalence_id(self) -> None:
        examples = make_retrieval_training_examples(repeats=16, benchmark_repeats=8)
        grouped: dict[str, list[object]] = {}
        for example in examples:
            grouped.setdefault(example.equivalence_id, []).append(example)
        self.assertTrue(any(len(items) >= 2 for items in grouped.values()))
        self.assertTrue(any(example.alternate_queries for example in examples))

    def test_multi_positive_alignment_accepts_any_positive_match(self) -> None:
        from src.losses.alignment import multi_positive_alignment_loss

        query = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
        positive_bag = torch.tensor([[[1.0, 0.0], [0.9, 0.1]]], dtype=torch.float32)
        loss = multi_positive_alignment_loss(query, positive_bag)
        self.assertLess(float(loss), 0.2)

    def test_multi_positive_margin_uses_best_positive_in_bag(self) -> None:
        from src.losses.alignment import multi_positive_margin_loss

        query = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
        positive_bag = torch.tensor([[[1.0, 0.0], [0.2, 0.98]]], dtype=torch.float32)
        negative_pool = torch.tensor([[0.6, 0.8]], dtype=torch.float32)
        loss = multi_positive_margin_loss(query, positive_bag, negative_pool=negative_pool, margin=0.2)
        self.assertLess(float(loss), 0.05)

    def test_support_slate_dataset_includes_cross_family_negatives(self) -> None:
        examples = make_phase4_contrast_examples(
            repeats=1,
            dataset="behavioral_constraints_v2_support_slate_localization_v1",
        )
        self.assertGreater(len(examples), 0)
        self.assertTrue(any(example.cross_family_negatives for example in examples))
        for example in examples:
            self.assertTrue(all(code != example.code for code in example.cross_family_negatives))

    def test_support_slate_localization_loss_prefers_direct_support(self) -> None:
        from src.losses.alignment import support_slate_localization_loss

        query = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
        positive = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
        same_family_negative_bag = torch.tensor([[[0.2, 0.98]]], dtype=torch.float32)
        same_family_negative_mask = torch.tensor([[True]])
        cross_family_negative_bag = torch.tensor([[[-1.0, 0.0]]], dtype=torch.float32)
        cross_family_negative_mask = torch.tensor([[True]])

        loss = support_slate_localization_loss(
            query,
            positive,
            same_family_negative_bag=same_family_negative_bag,
            same_family_negative_mask=same_family_negative_mask,
            cross_family_negative_bag=cross_family_negative_bag,
            cross_family_negative_mask=cross_family_negative_mask,
            temperature=0.08,
            same_family_weight=2.5,
            cross_family_weight=1.0,
        )
        self.assertLess(float(loss), 0.2)

    def test_support_slate_localization_loss_respects_same_family_weighting(self) -> None:
        from src.losses.alignment import support_slate_localization_loss

        query = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
        positive = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
        same_family_negative_bag = torch.tensor([[[0.98, 0.02]]], dtype=torch.float32)
        same_family_negative_mask = torch.tensor([[True]])

        low_weight_loss = support_slate_localization_loss(
            query,
            positive,
            same_family_negative_bag=same_family_negative_bag,
            same_family_negative_mask=same_family_negative_mask,
            temperature=0.08,
            same_family_weight=1.0,
        )
        high_weight_loss = support_slate_localization_loss(
            query,
            positive,
            same_family_negative_bag=same_family_negative_bag,
            same_family_negative_mask=same_family_negative_mask,
            temperature=0.08,
            same_family_weight=4.0,
        )
        self.assertGreater(float(high_weight_loss), float(low_weight_loss))

    def test_equivalence_family_map_collects_primary_and_alternate_queries(self) -> None:
        from train_production import _build_equivalence_family_map_from_examples

        examples = make_retrieval_training_examples(repeats=8, benchmark_repeats=4)
        family_map = _build_equivalence_family_map_from_examples(examples)
        self.assertTrue(family_map)
        self.assertTrue(any(len(views) >= 2 for views in family_map.values()))

    def test_sample_positive_query_views_uses_synthesis_when_enabled(self) -> None:
        from train_production import _build_equivalence_family_map_from_examples, _sample_positive_query_views_from_examples

        examples = make_retrieval_training_examples(repeats=8, benchmark_repeats=4)
        family_map = _build_equivalence_family_map_from_examples(examples)
        views = _sample_positive_query_views_from_examples(examples[:4], family_map, include_synthesis=True)
        self.assertEqual(len(views), 4)
        self.assertTrue(any(view != example.query for view, example in zip(views, examples[:4])))

    def test_retrieval_examples_cover_all_phase4_families(self) -> None:
        examples = make_retrieval_training_examples(repeats=16, benchmark_repeats=8)
        self.assertEqual(
            {example.family for example in examples},
            {"sorting", "strip_lines", "json_parse", "debounce", "frequency", "merge_dicts", "fetch_json", "startswith_js"},
        )

    def test_phase4_support_discipline_examples_include_equivalence_metadata(self) -> None:
        examples = make_phase4_contrast_examples(
            repeats=1,
            dataset="behavioral_constraints_v2_taxonomy_support_discipline_v1",
        )
        self.assertEqual(
            {example.family for example in examples},
            {"sorting", "strip_lines", "json_parse", "debounce", "frequency", "merge_dicts", "fetch_json", "startswith_js"},
        )
        self.assertEqual(
            {example.equivalence_id for example in examples},
            {"sorting", "strip_lines", "json_parse", "debounce", "frequency", "merge_dicts", "fetch_json", "startswith_js"},
        )
        self.assertTrue(all(example.alternate_queries for example in examples))
        self.assertTrue(all(example.synthesis_queries for example in examples))

    def test_phase4_examples_can_feed_equivalence_positive_query_sampling(self) -> None:
        from train_production import _build_equivalence_family_map_from_examples, _sample_positive_query_views_from_examples

        examples = make_phase4_contrast_examples(
            repeats=1,
            dataset="behavioral_constraints_v2_taxonomy_support_discipline_v1",
        )
        family_map = _build_equivalence_family_map_from_examples(examples)
        self.assertEqual(
            set(family_map),
            {"sorting", "strip_lines", "json_parse", "debounce", "frequency", "merge_dicts", "fetch_json", "startswith_js"},
        )
        views = _sample_positive_query_views_from_examples(examples, family_map, include_synthesis=True)
        self.assertEqual(len(views), len(examples))
        self.assertTrue(all(view != example.prompt for view, example in zip(views, examples)))


if __name__ == "__main__":
    unittest.main()
