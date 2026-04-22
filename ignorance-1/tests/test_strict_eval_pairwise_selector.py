import importlib.util
from pathlib import Path
import tempfile
import unittest

import torch
import yaml


_MODULE_PATH = Path(__file__).resolve().parents[1] / "test_2.7b.py"
_SPEC = importlib.util.spec_from_file_location("strict_eval_test_2_7b", _MODULE_PATH)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


class StrictEvalPairwiseSelectorTests(unittest.TestCase):
    def test_ordered_unique_texts_preserve_first_occurrence(self) -> None:
        self.assertEqual(
            _MODULE._ordered_unique_texts([
                "beta",
                "alpha",
                "beta",
                "gamma",
                "alpha",
            ]),
            ["beta", "alpha", "gamma"],
        )

    def test_build_eval_code_snippets_is_seeded_and_stable(self) -> None:
        first = _MODULE._build_eval_code_snippets(repeats=10)
        second = _MODULE._build_eval_code_snippets(repeats=10)
        self.assertEqual(first, second)
        self.assertGreater(len(first), 0)

    def test_arg_parser_registers_retrieval_facet_flags_once(self) -> None:
        parser = _MODULE._build_arg_parser()
        option_strings = [
            option
            for action in parser._actions
            for option in action.option_strings
        ]

        self.assertEqual(option_strings.count("--retrieval-facet-score-mode"), 1)
        self.assertEqual(option_strings.count("--retrieval-global-facet-blend"), 1)
        self.assertEqual(option_strings.count("--retrieval-facet-softmax-temperature"), 1)

        args = parser.parse_args([
            "15000000",
            "model.pt",
            "--retrieval-facet-score-mode",
            "softmax_maxsim",
            "--retrieval-global-facet-blend",
            "0.35",
            "--retrieval-facet-softmax-temperature",
            "0.2",
        ])
        self.assertEqual(args.retrieval_facet_score_mode, "softmax_maxsim")
        self.assertAlmostEqual(args.retrieval_global_facet_blend, 0.35)
        self.assertAlmostEqual(args.retrieval_facet_softmax_temperature, 0.2)

    def test_arg_parser_registers_retrieval_evidence_confidence_flags(self) -> None:
        parser = _MODULE._build_arg_parser()
        option_strings = [
            option
            for action in parser._actions
            for option in action.option_strings
        ]

        self.assertIn("--confidence-mode", option_strings)
        self.assertIn("--confidence-support-topk", option_strings)
        self.assertIn("--confidence-support-temperature", option_strings)
        self.assertIn("--confidence-parafence-variants", option_strings)

        args = parser.parse_args([
            "15000000",
            "model.pt",
            "--confidence-mode",
            "agreement_augmented",
            "--confidence-support-topk",
            "5",
            "--confidence-support-temperature",
            "0.2",
            "--confidence-parafence-variants",
            "2",
        ])
        self.assertEqual(args.confidence_mode, "agreement_augmented")
        self.assertEqual(args.confidence_support_topk, 5)
        self.assertAlmostEqual(args.confidence_support_temperature, 0.2)
        self.assertEqual(args.confidence_parafence_variants, 2)

    def test_config_file_can_supply_retrieval_evidence_confidence_defaults(self) -> None:
        parser = _MODULE._build_arg_parser()
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "strict_eval.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "confidence_threshold": 0.312,
                        "lexical_weight": 0.6,
                        "strict_eval": {
                            "confidence_mode": "neighborhood_posterior",
                            "confidence_support_topk": 5,
                            "confidence_support_temperature": 0.1,
                            "confidence_parafence_variants": 2,
                        },
                    },
                    sort_keys=False,
                )
            )
            args = parser.parse_args([
                "15000000",
                "model.pt",
                "--config",
                str(config_path),
            ])
            resolved = _MODULE._apply_config_overrides(args, parser)

        self.assertAlmostEqual(resolved.confidence_threshold, 0.312)
        self.assertAlmostEqual(resolved.lexical_weight, 0.6)
        self.assertEqual(resolved.confidence_mode, "neighborhood_posterior")
        self.assertEqual(resolved.confidence_support_topk, 5)
        self.assertAlmostEqual(resolved.confidence_support_temperature, 0.1)
        self.assertEqual(resolved.confidence_parafence_variants, 2)

    def test_support_discipline_eval_builds_supported_and_unsupported_cases(self) -> None:
        docs, metadata_by_doc, cases = _MODULE._build_support_discipline_eval()

        self.assertGreater(len(docs), 0)
        self.assertEqual(len(docs), len(metadata_by_doc))
        self.assertTrue(any(case["type"] == "Objective - Supported" for case in cases))
        self.assertTrue(any(case["type"] == "Objective - Unsupported" for case in cases))
        self.assertTrue(any(bool(meta["is_direct"]) for meta in metadata_by_doc.values()))
        self.assertTrue(any(not bool(meta["is_direct"]) for meta in metadata_by_doc.values()))

    def test_support_discipline_summary_tracks_direct_hits_and_abstention(self) -> None:
        metrics = _MODULE._summarize_support_discipline_results(
            [
                {
                    "type": "Objective - Supported",
                    "family": "sorting",
                    "retrieved": "doc-a",
                    "retrieved_family": "sorting",
                    "retrieved_is_direct": True,
                    "confidence": 0.82,
                },
                {
                    "type": "Objective - Supported",
                    "family": "json_parse",
                    "retrieved": "doc-b",
                    "retrieved_family": "json_parse",
                    "retrieved_is_direct": False,
                    "confidence": 0.51,
                },
                {
                    "type": "Objective - Unsupported",
                    "family": "sorting",
                    "retrieved": "<IGNORANT>",
                    "retrieved_family": None,
                    "retrieved_is_direct": False,
                    "confidence": 0.10,
                },
                {
                    "type": "Objective - Unsupported",
                    "family": "json_parse",
                    "retrieved": "doc-c",
                    "retrieved_family": "json_parse",
                    "retrieved_is_direct": False,
                    "confidence": 0.40,
                },
            ]
        )

        self.assertAlmostEqual(metrics["objective_supported_direct_rate"], 0.5)
        self.assertAlmostEqual(metrics["objective_supported_wrong_chunk_rate"], 0.5)
        self.assertAlmostEqual(metrics["objective_in_domain_unsupported_abstention_rate"], 0.5)
        self.assertAlmostEqual(metrics["objective_supported_confidence"], 0.665)
        self.assertAlmostEqual(metrics["objective_in_domain_unsupported_confidence"], 0.25)
        self.assertAlmostEqual(metrics["objective_confidence_gap"], 0.415)

    def test_strict_status_enforces_support_discipline_objectives(self) -> None:
        summary = {
            "has_confidence_head": True,
            "avg_known_exact_similarity": 0.80,
            "avg_known_paraphrase_similarity": 0.70,
            "synthesis_similarity": 0.60,
            "avg_ignorant_similarity": 0.10,
            "ignorance_gap": 0.60,
            "avg_known_margin": 0.10,
            "avg_known_confidence": 0.70,
            "avg_ood_confidence": 0.10,
            "code_diagnostics": {"avg_offdiag_similarity": 0.40, "participation_ratio_fraction": 0.20},
            "query_diagnostics": {"avg_offdiag_similarity": 0.40, "participation_ratio_fraction": 0.20},
            "objective_supported_direct_rate": 0.50,
            "objective_supported_wrong_chunk_rate": 0.30,
            "objective_in_domain_unsupported_abstention_rate": 0.50,
            "objective_confidence_gap": 0.10,
        }

        strict_status, strict_failures = _MODULE._strict_status(summary)

        self.assertEqual(strict_status, "❌ FAIL")
        self.assertTrue(any("direct support retrieval hygiene too low" in failure for failure in strict_failures))
        self.assertTrue(any("same-family wrong-chunk rate too high" in failure for failure in strict_failures))
        self.assertTrue(any("in-domain unsupported abstention too low" in failure for failure in strict_failures))
        self.assertTrue(any("supported vs in-domain unsupported confidence gap too small" in failure for failure in strict_failures))

    def test_pairwise_mode_none_returns_pointwise_scores(self) -> None:
        pointwise = torch.tensor([0.70, 0.69, 0.10])
        scores = _MODULE._pairwise_borda_scores(
            torch.tensor([0.70, 0.69, 0.10]),
            torch.tensor([0.70, 0.69, 0.10]),
            torch.tensor([0.70, 0.69, 0.10]),
            torch.tensor([0.70, 0.69, 0.10]),
            torch.tensor([0.70, 0.69, 0.10]),
            torch.tensor([0.00, 0.00, 0.00]),
            pointwise,
            mode="none",
        )
        self.assertTrue(torch.allclose(scores, pointwise))

    def test_combined_scores_can_use_hard_late_interaction_facets(self) -> None:
        index = _MODULE.VectorIndex(
            ["doc-a", "doc-b"],
            torch.tensor([[1.0, 0.0], [0.95, 0.05]], dtype=torch.float32),
            facet_embeddings=torch.tensor(
                [
                    [[1.0, 0.0], [1.0, 0.0]],
                    [[1.0, 0.0], [0.0, 1.0]],
                ],
                dtype=torch.float32,
            ),
            facet_score_mode="hard_maxsim",
            global_facet_blend=0.0,
        )
        query = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
        query_facets = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float32)

        scores = _MODULE._combined_scores(index, "query", query, 0.0, query_facets=query_facets)

        self.assertEqual(int(torch.argmax(scores).item()), 1)
        self.assertGreater(float(scores[1]), float(scores[0]))

    def test_combined_scores_can_blend_global_and_facet_channels(self) -> None:
        index = _MODULE.VectorIndex(
            ["doc-a", "doc-b"],
            torch.tensor([[1.0, 0.0], [0.6, 0.8]], dtype=torch.float32),
            facet_embeddings=torch.tensor(
                [
                    [[1.0, 0.0], [1.0, 0.0]],
                    [[1.0, 0.0], [0.0, 1.0]],
                ],
                dtype=torch.float32,
            ),
            facet_score_mode="hard_maxsim",
            global_facet_blend=0.98,
        )
        query = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
        query_facets = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float32)

        scores = _MODULE._combined_scores(index, "query", query, 0.0, query_facets=query_facets)

        self.assertEqual(int(torch.argmax(scores).item()), 0)

    def test_search_text_returns_all_topk_hits(self) -> None:
        index = _MODULE.VectorIndex(
            ["alpha react typescript", "beta react", "gamma nextjs"],
            torch.tensor(
                [
                    [1.0, 0.0],
                    [0.9, 0.1],
                    [0.0, 1.0],
                ],
                dtype=torch.float32,
            ),
        )

        result = index.search_text(
            "react typescript",
            torch.tensor([[1.0, 0.0]], dtype=torch.float32),
            k=2,
            lexical_weight=0.2,
        )

        self.assertEqual(len(result.ids), 2)
        self.assertEqual(result.ids[0], "alpha react typescript")
        self.assertEqual(result.ids[1], "beta react")
        self.assertEqual(tuple(result.scores.shape), (2,))

    def test_support_consistency_scores_favor_matching_operation(self) -> None:
        scores = _MODULE._support_consistency_scores(
            [
                "def solve(numbers):\n    return sorted(numbers)",
                "const parsed = JSON.parse(responseText);",
            ],
            "Order an array of integers from smallest to largest",
            device="cpu",
            dtype=torch.float32,
        )
        self.assertGreater(float(scores[0]), float(scores[1]))

    def test_neighborhood_posterior_confidence_prefers_supported_neighbors(self) -> None:
        known_confidence, known_details = _MODULE._confidence_from_retrieval_evidence(
            mode="neighborhood_posterior",
            top_scores=torch.tensor([0.32, 0.31, 0.30]),
            pred_scores=torch.tensor([0.32, 0.31, 0.30]),
            query_scores=torch.tensor([0.31, 0.30, 0.29]),
            support_scores=torch.tensor([1.0, 1.0, 0.8]),
            consensus_scores=torch.tensor([0.99, 0.98, 0.98]),
            head_confidence=0.46,
        )
        ood_confidence, ood_details = _MODULE._confidence_from_retrieval_evidence(
            mode="neighborhood_posterior",
            top_scores=torch.tensor([0.24, 0.24, 0.23]),
            pred_scores=torch.tensor([0.24, 0.24, 0.23]),
            query_scores=torch.tensor([0.24, 0.24, 0.23]),
            support_scores=torch.tensor([0.0, 0.0, 0.0]),
            consensus_scores=torch.tensor([0.98, 0.98, 0.98]),
            head_confidence=0.46,
        )
        self.assertGreater(known_confidence, 0.60)
        self.assertLess(ood_confidence, 0.35)
        self.assertGreater(known_details["support_posterior"], ood_details["support_posterior"])

    def test_evidential_support_confidence_reports_evidence_strength(self) -> None:
        confidence, details = _MODULE._confidence_from_retrieval_evidence(
            mode="evidential_support",
            top_scores=torch.tensor([0.34, 0.33, 0.32]),
            pred_scores=torch.tensor([0.34, 0.33, 0.32]),
            query_scores=torch.tensor([0.33, 0.33, 0.31]),
            support_scores=torch.tensor([1.0, 1.0, 1.0]),
            consensus_scores=torch.tensor([0.99, 0.98, 0.98]),
            head_confidence=0.46,
        )
        self.assertGreater(confidence, 0.60)
        self.assertIn("evidence_strength", details)
        self.assertGreater(details["evidence_strength"], 0.0)

    def test_agreement_augmented_confidence_uses_variant_agreement(self) -> None:
        higher_confidence, _ = _MODULE._confidence_from_retrieval_evidence(
            mode="agreement_augmented",
            top_scores=torch.tensor([0.30, 0.29, 0.28]),
            pred_scores=torch.tensor([0.30, 0.29, 0.28]),
            query_scores=torch.tensor([0.30, 0.28, 0.27]),
            support_scores=torch.tensor([1.0, 1.0, 0.8]),
            consensus_scores=torch.tensor([0.99, 0.98, 0.98]),
            head_confidence=0.46,
            variant_agreement=0.5,
        )
        lower_confidence, _ = _MODULE._confidence_from_retrieval_evidence(
            mode="agreement_augmented",
            top_scores=torch.tensor([0.30, 0.29, 0.28]),
            pred_scores=torch.tensor([0.30, 0.29, 0.28]),
            query_scores=torch.tensor([0.30, 0.28, 0.27]),
            support_scores=torch.tensor([1.0, 1.0, 0.8]),
            consensus_scores=torch.tensor([0.99, 0.98, 0.98]),
            head_confidence=0.46,
            variant_agreement=0.0,
        )
        self.assertGreater(higher_confidence, lower_confidence)

    def test_evidence_borda_can_flip_top_candidate(self) -> None:
        pointwise = torch.tensor([0.7000, 0.6950, 0.1000])
        pairwise = _MODULE._pairwise_borda_scores(
            pred_scores=torch.tensor([0.90, 0.80, 0.20]),
            query_scores=torch.tensor([0.40, 0.95, 0.10]),
            agreement_scores=torch.tensor([0.45, 0.92, 0.10]),
            lexical_scores=torch.tensor([0.30, 0.82, 0.05]),
            consensus_scores=torch.tensor([0.25, 0.78, 0.02]),
            support_scores=torch.tensor([0.00, 0.00, 0.00]),
            pointwise_scores=pointwise,
            mode="evidence_borda",
        )
        self.assertEqual(int(torch.argmax(pairwise).item()), 1)
        self.assertGreater(float(pairwise[1]), float(pairwise[0]))

    def test_citecheck_borda_can_flip_on_support_consistency(self) -> None:
        pointwise = torch.tensor([0.7100, 0.7050, 0.1000])
        pairwise = _MODULE._pairwise_borda_scores(
            pred_scores=torch.tensor([0.91, 0.89, 0.20]),
            query_scores=torch.tensor([0.45, 0.72, 0.10]),
            agreement_scores=torch.tensor([0.44, 0.81, 0.10]),
            lexical_scores=torch.tensor([0.28, 0.63, 0.05]),
            consensus_scores=torch.tensor([0.20, 0.77, 0.02]),
            support_scores=torch.tensor([0.05, 0.95, 0.00]),
            pointwise_scores=pointwise,
            mode="citecheck_borda",
        )
        self.assertEqual(int(torch.argmax(pairwise).item()), 1)
        self.assertGreater(float(pairwise[1]), float(pairwise[0]))

    def test_citecheck_floor_borda_only_flips_when_support_floor_is_active(self) -> None:
        pointwise = torch.tensor([0.7100, 0.7050, 0.1000])
        baseline = _MODULE._pairwise_borda_scores(
            pred_scores=torch.tensor([0.95, 0.90, 0.20]),
            query_scores=torch.tensor([0.95, 0.90, 0.10]),
            agreement_scores=torch.tensor([0.95, 0.90, 0.10]),
            lexical_scores=torch.tensor([0.95, 0.90, 0.05]),
            consensus_scores=torch.tensor([0.95, 0.90, 0.02]),
            support_scores=torch.tensor([0.05, 0.95, 0.00]),
            pointwise_scores=pointwise,
            mode="citecheck_borda",
        )
        gated = _MODULE._pairwise_borda_scores(
            pred_scores=torch.tensor([0.95, 0.90, 0.20]),
            query_scores=torch.tensor([0.95, 0.90, 0.10]),
            agreement_scores=torch.tensor([0.95, 0.90, 0.10]),
            lexical_scores=torch.tensor([0.95, 0.90, 0.05]),
            consensus_scores=torch.tensor([0.95, 0.90, 0.02]),
            support_scores=torch.tensor([0.05, 0.95, 0.00]),
            pointwise_scores=pointwise,
            mode="citecheck_floor_borda",
            support_floor_active=True,
        )
        inert = _MODULE._pairwise_borda_scores(
            pred_scores=torch.tensor([0.95, 0.90, 0.20]),
            query_scores=torch.tensor([0.95, 0.90, 0.10]),
            agreement_scores=torch.tensor([0.95, 0.90, 0.10]),
            lexical_scores=torch.tensor([0.95, 0.90, 0.05]),
            consensus_scores=torch.tensor([0.95, 0.90, 0.02]),
            support_scores=torch.tensor([0.05, 0.95, 0.00]),
            pointwise_scores=pointwise,
            mode="citecheck_floor_borda",
            support_floor_active=False,
        )
        self.assertEqual(int(torch.argmax(baseline).item()), 0)
        self.assertEqual(int(torch.argmax(gated).item()), 1)
        self.assertEqual(int(torch.argmax(inert).item()), 0)

    def test_citecheck_support_floor_gate_can_flip_low_margin_case(self) -> None:
        docs = [
            "const parsed = JSON.parse(payload);\n",
            "def solve(numbers):\n    return sorted(numbers)\n",
            "with open(path) as handle:\n    rows = [line.strip() for line in handle]\n",
        ]
        low_margin_index = _MODULE.VectorIndex(
            docs,
            torch.tensor(
                [
                    [1.00, 0.00],
                    [0.99, 0.01],
                    [0.00, 1.00],
                ]
            ),
        )
        _, gated_idx = _MODULE._reranked_top_scores(
            low_margin_index,
            "Sort a numeric list ascending and return the result.",
            torch.tensor([[1.0, 0.0]]),
            torch.tensor([[1.0, 0.0]]),
            lexical_weight=0.0,
            rerank_topk=3,
            rerank_query_weight=0.0,
            rerank_agreement_weight=0.0,
            rerank_lexical_weight=0.0,
            rerank_support_weight=0.0,
            rerank_consensus_weight=0.0,
            rerank_consensus_temperature=0.10,
            rerank_consensus_floor=0.80,
            rerank_consensus_margin_gate=0.05,
            rerank_shortlist_mode="pred",
            rerank_pairwise_mode="citecheck_floor_borda",
            rerank_support_floor_margin_gate=0.02,
        )
        self.assertEqual(docs[int(gated_idx[0].item())], docs[1])

    def test_supportspec_citecheck_floor_borda_uses_support_floor_on_low_margin_case(self) -> None:
        pointwise = torch.tensor([0.7100, 0.7050, 0.1000])
        gated = _MODULE._pairwise_borda_scores(
            pred_scores=torch.tensor([0.95, 0.90, 0.20]),
            query_scores=torch.tensor([0.95, 0.90, 0.10]),
            agreement_scores=torch.tensor([0.95, 0.90, 0.10]),
            lexical_scores=torch.tensor([0.95, 0.90, 0.05]),
            consensus_scores=torch.tensor([0.95, 0.90, 0.02]),
            support_scores=torch.tensor([0.05, 0.95, 0.00]),
            spec_scores=torch.tensor([0.0, 0.0, 0.0]),
            pointwise_scores=pointwise,
            mode="supportspec_citecheck_floor_borda",
            support_floor_active=True,
            spec_floor_active=False,
        )
        inert = _MODULE._pairwise_borda_scores(
            pred_scores=torch.tensor([0.95, 0.90, 0.20]),
            query_scores=torch.tensor([0.95, 0.90, 0.10]),
            agreement_scores=torch.tensor([0.95, 0.90, 0.10]),
            lexical_scores=torch.tensor([0.95, 0.90, 0.05]),
            consensus_scores=torch.tensor([0.95, 0.90, 0.02]),
            support_scores=torch.tensor([0.05, 0.95, 0.00]),
            spec_scores=torch.tensor([0.0, 0.0, 0.0]),
            pointwise_scores=pointwise,
            mode="supportspec_citecheck_floor_borda",
            support_floor_active=False,
            spec_floor_active=False,
        )
        self.assertEqual(int(torch.argmax(gated).item()), 1)
        self.assertEqual(int(torch.argmax(inert).item()), 0)

    def test_supportspec_borda_can_flip_on_joint_support_and_contract(self) -> None:
        pointwise = torch.tensor([0.7100, 0.7050, 0.1000])
        pairwise = _MODULE._pairwise_borda_scores(
            pred_scores=torch.tensor([0.93, 0.89, 0.20]),
            query_scores=torch.tensor([0.55, 0.72, 0.10]),
            agreement_scores=torch.tensor([0.52, 0.84, 0.10]),
            lexical_scores=torch.tensor([0.32, 0.63, 0.05]),
            consensus_scores=torch.tensor([0.24, 0.77, 0.02]),
            support_scores=torch.tensor([0.05, 0.95, 0.00]),
            spec_scores=torch.tensor([0.10, 1.00, 0.00]),
            pointwise_scores=pointwise,
            mode="supportspec_borda",
        )
        self.assertEqual(int(torch.argmax(pairwise).item()), 1)
        self.assertGreater(float(pairwise[1]), float(pairwise[0]))

    def test_parafence_query_variants_add_deterministic_wrappers(self) -> None:
        variants = _MODULE._parafence_query_variants(
            "Sort a numeric list ascending and return the result.",
            max_variants=3,
        )
        self.assertGreaterEqual(len(variants), 3)
        self.assertEqual(variants[0], "Sort a numeric list ascending and return the result.")
        self.assertIn("Order an array of integers from smallest to largest.", variants)
        self.assertIn("Write code to sort a numeric list ascending and return the result.", variants)

    def test_parafence_query_variants_do_not_wrap_ood_queries_as_code(self) -> None:
        variants = _MODULE._parafence_query_variants(
            "What is the weather in Tokyo today?",
            max_variants=3,
        )
        self.assertEqual(variants, ["What is the weather in Tokyo today?"])

    def test_parafence_vote_scores_can_flip_to_stable_winner(self) -> None:
        base = torch.tensor([2.01, 2.00, 0.10])
        variant_selection_rows = torch.tensor(
            [
                [0.40, 0.90, 0.10],
                [0.35, 0.95, 0.10],
                [0.30, 0.92, 0.10],
            ]
        )
        stability = _MODULE._parafence_vote_scores(variant_selection_rows)
        combined = base + 0.75 * stability
        self.assertEqual(int(torch.argmax(stability).item()), 1)
        self.assertEqual(int(torch.argmax(combined).item()), 1)

    def test_resolved_safe_expand_topk_only_expands_for_low_margin(self) -> None:
        expanded = _MODULE._resolved_safe_expand_topk(
            torch.tensor([0.7000, 0.6950, 0.1000]),
            base_topk=4,
            safe_expand_topk=6,
            safe_expand_margin=0.01,
        )
        stable = _MODULE._resolved_safe_expand_topk(
            torch.tensor([0.7000, 0.6800, 0.1000]),
            base_topk=4,
            safe_expand_topk=6,
            safe_expand_margin=0.01,
        )
        unsorted_stable = _MODULE._resolved_safe_expand_topk(
            torch.tensor([0.3333, 0.4049, 0.6667]),
            base_topk=3,
            safe_expand_topk=6,
            safe_expand_margin=0.01,
        )
        self.assertEqual(expanded, 6)
        self.assertEqual(stable, 4)
        self.assertEqual(unsorted_stable, 3)

    def test_answer_spec_scores_require_full_multistep_coverage(self) -> None:
        scores = _MODULE._answer_spec_scores(
            [
                "# task: Read a file, parse the JSON in it, and sort the result.\nwith open(path) as handle:\n    values = sorted(json.load(handle))\n",
                "# task: Parse a json string into a javascript object.\nconst parsed = JSON.parse(payload);\n",
            ],
            "Read a file, parse the JSON in it, and sort the result.",
            device="cpu",
            dtype=torch.float32,
            mode="hard_multistep",
        )
        self.assertGreater(float(scores[0]), 0.0)
        self.assertEqual(float(scores[1]), 0.0)
        self.assertGreater(float(scores[0]), float(scores[1]))

    def test_answer_spec_scores_support_pref_multistep_floor_requires_full_multistep_coverage(self) -> None:
        scores = _MODULE._answer_spec_scores(
            [
                "# task: Read a file, parse the JSON in it, and sort the result.\nwith open(path) as handle:\n    values = sorted(json.load(handle))\n",
                "# task: Parse a json string into a javascript object.\nconst parsed = JSON.parse(payload);\n",
            ],
            "Read a file, parse the JSON in it, and sort the result.",
            device="cpu",
            dtype=torch.float32,
            mode="support_pref_multistep_floor",
        )
        self.assertGreater(float(scores[0]), 0.0)
        self.assertEqual(float(scores[1]), 0.0)
        self.assertGreater(float(scores[0]), float(scores[1]))

    def test_answer_spec_scores_code_pref_multistep_floor_requires_full_multistep_and_prefers_code(self) -> None:
        scores = _MODULE._answer_spec_scores(
            [
                "# task: Read a file, parse the JSON in it, and sort the result.\nwith open(path) as handle:\n    values = sorted(json.load(handle))\n",
                "Read the file, parse the JSON, and then sort the result before returning it.",
                "# task: Parse a json string into a javascript object.\nconst parsed = JSON.parse(payload);\n",
            ],
            "Read a file, parse the JSON in it, and sort the result.",
            device="cpu",
            dtype=torch.float32,
            mode="code_pref_multistep_floor",
        )
        self.assertGreater(float(scores[0]), float(scores[1]))
        self.assertEqual(float(scores[2]), 0.0)
        self.assertGreater(float(scores[0]), float(scores[2]))

    def test_answer_spec_scores_code_pref_soft_multistep_keeps_partial_candidate_nonzero(self) -> None:
        candidates = [
            "# task: Read a file, parse the JSON in it, and sort the result.\nwith open(path) as handle:\n    values = sorted(json.load(handle))\n",
            "Read the file, parse the JSON, and then sort the result before returning it.",
            "# task: Parse a json string into a javascript object.\nconst parsed = JSON.parse(payload);\n",
        ]
        baseline = _MODULE._answer_spec_scores(
            candidates,
            "Read a file, parse the JSON in it, and sort the result.",
            device="cpu",
            dtype=torch.float32,
            mode="soft",
        )
        scores = _MODULE._answer_spec_scores(
            candidates,
            "Read a file, parse the JSON in it, and sort the result.",
            device="cpu",
            dtype=torch.float32,
            mode="code_pref_soft_multistep",
        )
        self.assertGreater(float(scores[0] - scores[1]), float(baseline[0] - baseline[1]))
        self.assertGreater(float(scores[2]), 0.0)

    def test_answer_spec_gate_can_flip_low_margin_multistep_case(self) -> None:
        docs = [
            "# task: Parse a json string into a javascript object.\nconst parsed = JSON.parse(payload);\n",
            "# task: Read a file, parse the JSON in it, and sort the result.\nwith open(path) as handle:\n    values = sorted(json.load(handle))\n",
            "# task: Read each line from a text file and strip whitespace.\nwith open(path) as handle:\n    rows = [line.strip() for line in handle]\n",
        ]
        index = _MODULE.VectorIndex(
            docs,
            torch.tensor(
                [
                    [1.00, 0.00],
                    [0.99, 0.01],
                    [0.00, 1.00],
                ]
            ),
        )
        _, baseline_idx = _MODULE._reranked_top_scores(
            index,
            "Read a file, parse the JSON in it, and sort the result.",
            torch.tensor([[1.0, 0.0]]),
            torch.tensor([[1.0, 0.0]]),
            lexical_weight=0.0,
            rerank_topk=3,
            rerank_query_weight=0.0,
            rerank_agreement_weight=0.0,
            rerank_lexical_weight=0.0,
            rerank_support_weight=0.0,
            rerank_consensus_weight=0.0,
            rerank_consensus_temperature=0.10,
            rerank_consensus_floor=0.80,
            rerank_consensus_margin_gate=0.05,
            rerank_shortlist_mode="pred",
            rerank_pairwise_mode="none",
        )
        _, gated_idx = _MODULE._reranked_top_scores(
            index,
            "Read a file, parse the JSON in it, and sort the result.",
            torch.tensor([[1.0, 0.0]]),
            torch.tensor([[1.0, 0.0]]),
            lexical_weight=0.0,
            rerank_topk=3,
            rerank_query_weight=0.0,
            rerank_agreement_weight=0.0,
            rerank_lexical_weight=0.0,
            rerank_support_weight=0.0,
            rerank_consensus_weight=0.0,
            rerank_consensus_temperature=0.10,
            rerank_consensus_floor=0.80,
            rerank_consensus_margin_gate=0.05,
            rerank_shortlist_mode="pred",
            rerank_pairwise_mode="none",
            rerank_answerspec_mode="hard_multistep",
            rerank_answerspec_margin_gate=0.02,
        )
        self.assertEqual(docs[int(baseline_idx[0].item())], docs[0])
        self.assertEqual(docs[int(gated_idx[0].item())], docs[1])

    def test_support_pref_multistep_floor_gate_can_flip_low_margin_multistep_case(self) -> None:
        docs = [
            "# task: Parse a json string into a javascript object.\nconst parsed = JSON.parse(payload);\n",
            "# task: Read a file, parse the JSON in it, and sort the result.\nwith open(path) as handle:\n    values = sorted(json.load(handle))\n",
            "# task: Read each line from a text file and strip whitespace.\nwith open(path) as handle:\n    rows = [line.strip() for line in handle]\n",
        ]
        index = _MODULE.VectorIndex(
            docs,
            torch.tensor(
                [
                    [1.00, 0.00],
                    [0.99, 0.01],
                    [0.00, 1.00],
                ]
            ),
        )
        _, gated_idx = _MODULE._reranked_top_scores(
            index,
            "Read a file, parse the JSON in it, and sort the result.",
            torch.tensor([[1.0, 0.0]]),
            torch.tensor([[1.0, 0.0]]),
            lexical_weight=0.0,
            rerank_topk=3,
            rerank_query_weight=0.0,
            rerank_agreement_weight=0.0,
            rerank_lexical_weight=0.0,
            rerank_support_weight=0.0,
            rerank_consensus_weight=0.0,
            rerank_consensus_temperature=0.10,
            rerank_consensus_floor=0.80,
            rerank_consensus_margin_gate=0.05,
            rerank_shortlist_mode="pred",
            rerank_pairwise_mode="none",
            rerank_answerspec_mode="support_pref_multistep_floor",
            rerank_answerspec_margin_gate=0.02,
        )
        self.assertEqual(docs[int(gated_idx[0].item())], docs[1])

    def test_supportspec_floor_borda_prefers_full_multistep_finalist(self) -> None:
        pred_scores = torch.tensor([0.99, 0.98])
        query_scores = torch.tensor([0.99, 0.98])
        agreement_scores = torch.tensor([0.99, 0.98])
        lexical_scores = torch.tensor([0.99, 0.98])
        consensus_scores = torch.tensor([0.99, 0.98])
        support_scores = torch.tensor([0.99, 0.98])
        spec_scores = torch.tensor([0.0, 1.0])
        pointwise_scores = torch.tensor([0.99, 0.98])

        baseline = _MODULE._pairwise_borda_scores(
            pred_scores,
            query_scores,
            agreement_scores,
            lexical_scores,
            consensus_scores,
            support_scores,
            spec_scores,
            pointwise_scores,
            mode="supportspec_borda",
            spec_floor_active=True,
        )
        floored = _MODULE._pairwise_borda_scores(
            pred_scores,
            query_scores,
            agreement_scores,
            lexical_scores,
            consensus_scores,
            support_scores,
            spec_scores,
            pointwise_scores,
            mode="supportspec_floor_borda",
            spec_floor_active=True,
        )
        self.assertEqual(int(torch.argmax(baseline).item()), 0)
        self.assertEqual(int(torch.argmax(floored).item()), 1)

    def test_supportspec_citecheck_floor_borda_prefers_full_multistep_finalist(self) -> None:
        floored = _MODULE._pairwise_borda_scores(
            pred_scores=torch.tensor([0.99, 0.98]),
            query_scores=torch.tensor([0.99, 0.98]),
            agreement_scores=torch.tensor([0.99, 0.98]),
            lexical_scores=torch.tensor([0.99, 0.98]),
            consensus_scores=torch.tensor([0.99, 0.98]),
            support_scores=torch.tensor([0.99, 0.98]),
            spec_scores=torch.tensor([0.0, 1.0]),
            pointwise_scores=torch.tensor([0.99, 0.98]),
            mode="supportspec_citecheck_floor_borda",
            spec_floor_active=True,
            support_floor_active=True,
        )
        self.assertEqual(int(torch.argmax(floored).item()), 1)

    def test_pairwise_mode_preserves_returned_score_scale(self) -> None:
        index = _MODULE.VectorIndex(
            [
                "def solve(numbers):\n    return sorted(numbers)",
                "with open(input_path) as handle:\n    lines = [line.strip() for line in handle]",
                "const parsed = JSON.parse(responseText);",
            ],
            torch.tensor(
                [
                    [1.0, 0.0],
                    [0.7, 0.3],
                    [0.0, 1.0],
                ]
            ),
        )
        top_scores, _ = _MODULE._reranked_top_scores(
            index,
            "Parse a json string into a javascript object.",
            torch.tensor([[0.6, 0.4]]),
            torch.tensor([[0.0, 1.0]]),
            lexical_weight=0.60,
            rerank_topk=3,
            rerank_query_weight=0.32,
            rerank_agreement_weight=0.18,
            rerank_lexical_weight=0.02,
            rerank_support_weight=0.24,
            rerank_consensus_weight=0.35,
            rerank_consensus_temperature=0.0176,
            rerank_consensus_floor=0.9165,
            rerank_consensus_margin_gate=0.0088,
            rerank_shortlist_mode="pred_query_union_local",
            rerank_pairwise_mode="citecheck_borda",
        )
        self.assertGreaterEqual(top_scores.numel(), 1)
        self.assertLessEqual(float(top_scores.max()), 1.0 + 1e-6)

    def test_verifier_uplift_can_raise_low_margin_verified_winner(self) -> None:
        boosted = _MODULE._apply_verifier_uplift(
            torch.tensor([0.4787, 0.4786]),
            torch.tensor([1.19, -0.18]),
            torch.tensor([1.0, 1.0]),
            rerank_verifier_uplift_weight=0.40,
            rerank_verifier_gap_scale=1.0,
        )
        self.assertGreater(float(boosted[0]), 0.85)
        self.assertAlmostEqual(float(boosted[1]), 0.4786, places=4)
        self.assertLessEqual(float(boosted.max()), 1.0 + 1e-6)

    def test_verifier_uplift_requires_nonzero_support_or_spec_signal(self) -> None:
        boosted = _MODULE._apply_verifier_uplift(
            torch.tensor([0.4787, 0.4786]),
            torch.tensor([1.19, -0.18]),
            torch.tensor([0.0, 0.0]),
            rerank_verifier_uplift_weight=0.40,
            rerank_verifier_gap_scale=1.0,
        )
        self.assertAlmostEqual(float(boosted[0]), 0.4787, places=4)
        self.assertAlmostEqual(float(boosted[1]), 0.4786, places=4)

    def test_selective_gate_abstains_on_low_margin_low_top1(self) -> None:
        should_abstain = _MODULE._should_selectively_abstain(
            torch.tensor([0.6900, 0.6840, 0.6810]),
            mode="margin_mean_gap",
            selective_gate_margin_threshold=0.010,
            selective_gate_mean_gap_threshold=0.012,
            selective_gate_similarity_floor=0.70,
        )
        self.assertTrue(should_abstain)

    def test_selective_gate_keeps_strong_top_candidate(self) -> None:
        should_abstain = _MODULE._should_selectively_abstain(
            torch.tensor([0.7350, 0.7290, 0.7260]),
            mode="margin_mean_gap",
            selective_gate_margin_threshold=0.010,
            selective_gate_mean_gap_threshold=0.012,
            selective_gate_similarity_floor=0.70,
        )
        self.assertFalse(should_abstain)


if __name__ == "__main__":
    unittest.main()
