from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import torch

from src.utils.retrieval import RetrievalResult
from experiments.sinai_webdev_demo_harness import (
    CorpusChunk,
    SinaiModelBackedDemoHarness,
    SinaiWebDevDemoHarness,
    format_demo_response,
    load_corpus_chunks,
)


_MODULE_PATH = Path(__file__).resolve().parents[1] / "test_2.7b.py"
_SPEC = importlib.util.spec_from_file_location("strict_eval_test_2_7b", _MODULE_PATH)
assert _SPEC and _SPEC.loader
_STRICT_EVAL = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_STRICT_EVAL)


class FakeTokenizer:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def batch_encode(self, texts: list[str], seq_len: int, device: str) -> torch.Tensor:
        self.batch_sizes.append(len(texts))
        rows = []
        for text in texts:
            base = float(len(text))
            rows.append([base + i for i in range(seq_len)])
        return torch.tensor(rows, dtype=torch.float32)


class FakeModel:
    def encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        base = input_ids[:, :4].float()
        return base

    def retrieval_project(self, latent: torch.Tensor) -> torch.Tensor:
        return latent

    def query_confidence(self, latent: torch.Tensor) -> torch.Tensor:
        return torch.full((latent.shape[0],), 0.9, dtype=torch.float32)


class FakeConfig:
    max_seq_len = 4
    use_retrieval_facets = False


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required for GPU harness validation")
class SinaiWebDevDemoHarnessGpuTests(unittest.TestCase):
    def test_load_production_model_on_gpu_and_answer_supported_query(self) -> None:
        checkpoint_path = Path(__file__).resolve().parents[1] / "artifacts" / "ignorance_1_2.7b_v340_neighborhood_posterior_taxonomy_coverage.pt"
        model, config, tokenizer, has_confidence_head = _STRICT_EVAL.load_model_for_demo(
            model_path=str(checkpoint_path),
            size=2_700_000_000,
            force_cpu=False,
        )

        chunks = [
            CorpusChunk(
                chunk_id="good",
                text="Type React event handlers as React.ChangeEventHandler<HTMLInputElement> so the input event is typed correctly.",
                title="Typing Events",
                url="https://example.com/typescript-events",
                source_id="typescript",
                priority="P0",
                content_type="reference",
                section_path=["Typing Events"],
                is_deprecated=False,
            ),
            CorpusChunk(
                chunk_id="bad",
                text="The FIFA World Cup final is a football match played to determine the winner of the tournament.",
                title="Football",
                url="https://example.com/football",
                source_id="mdn",
                priority="P1",
                content_type="guide",
                section_path=["Sports"],
                is_deprecated=False,
            ),
        ]
        harness = SinaiModelBackedDemoHarness(
            model=model,
            config=config,
            tokenizer=tokenizer,
            chunks=chunks,
            device=next(model.parameters()).device.type,
            has_confidence_head=has_confidence_head,
            support_threshold=-1.0,
            confidence_threshold=0.0,
            index_batch_size=2,
        )

        result = harness.answer("How do I type a React event handler in TypeScript?", top_k=2)

        self.assertTrue(result.supported)
        self.assertEqual(result.citations[0].chunk_id, "good")
        self.assertEqual(next(model.parameters()).device.type, "cuda")
        self.assertTrue(result.used_model_evidence)


    def test_model_backed_harness_abstains_when_only_lexical_fallback_matches(self) -> None:
        checkpoint_path = Path(__file__).resolve().parents[1] / "artifacts" / "ignorance_1_2.7b_v340_neighborhood_posterior_taxonomy_coverage.pt"
        model, config, tokenizer, has_confidence_head = _STRICT_EVAL.load_model_for_demo(
            model_path=str(checkpoint_path),
            size=2_700_000_000,
            force_cpu=False,
        )

        chunks = [
            CorpusChunk(
                chunk_id="bad",
                text="The FIFA World Cup final is a football match played to determine the winner of the tournament.",
                title="Football",
                url="https://example.com/football",
                source_id="mdn",
                priority="P1",
                content_type="guide",
                section_path=["Sports"],
                is_deprecated=False,
            ),
        ]
        harness = SinaiModelBackedDemoHarness(
            model=model,
            config=config,
            tokenizer=tokenizer,
            chunks=chunks,
            device=next(model.parameters()).device.type,
            has_confidence_head=has_confidence_head,
            support_threshold=0.0,
            confidence_threshold=0.0,
            index_batch_size=1,
        )

        result = harness.answer("Who won the 2018 World Cup final?", top_k=1)

        self.assertFalse(result.supported)
        self.assertEqual(result.citations, [])


class SinaiWebDevDemoHarnessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.base = Path(self.tempdir.name)
        (self.base / "normalized" / "chunks").mkdir(parents=True)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def _write_chunk(
        self,
        chunk_id: str,
        text: str,
        *,
        title: str,
        url: str,
        source_id: str,
        priority: str = "P0",
        content_type: str = "guide",
        deprecated: bool = False,
        section_path: list[str] | None = None,
    ) -> None:
        payload = {
            "chunk_id": chunk_id,
            "chunk_text": text,
            "metadata": {
                "doc_title": title,
                "url": url,
                "source_id": source_id,
                "priority": priority,
                "content_type": content_type,
                "is_deprecated": deprecated,
                "section_path": section_path or [],
            },
        }
        (self.base / "normalized" / "chunks" / f"{chunk_id}.json").write_text(json.dumps(payload))

    def test_load_corpus_chunks_reads_normalized_chunks(self) -> None:
        self._write_chunk(
            "react-1",
            "Use event handlers to respond to user interactions.",
            title="Responding to Events",
            url="https://react.dev/learn/responding-to-events",
            source_id="react",
            section_path=["Learn", "Responding to Events"],
        )

        chunks = load_corpus_chunks(self.base)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].chunk_id, "react-1")
        self.assertEqual(chunks[0].section_path, ["Learn", "Responding to Events"])

    def test_load_corpus_chunks_prefers_chunk_title_over_doc_title(self) -> None:
        chunks_dir = self.base / "normalized" / "chunks"
        metadata_dir = self.base / "normalized" / "metadata"
        metadata_dir.mkdir(parents=True)
        (chunks_dir / "react-typescript__chunk001.json").write_text(json.dumps({
            "chunk_id": "react-typescript__chunk001",
            "doc_id": "react-typescript",
            "title": "DOM Events",
            "chunk_text": "When extracting a React event handler, type the event explicitly.",
            "metadata": {
                "url": "https://react.dev/learn/typescript",
                "source_id": "react",
                "priority": "P0",
                "content_type": "tutorial",
                "section_path": ["TypeScript", "DOM Events"],
            },
        }))
        (metadata_dir / "react-typescript.json").write_text(json.dumps({
            "doc_title": "typescript",
            "url": "https://react.dev/learn/typescript",
            "source_id": "react",
            "section_path": ["learn"],
        }))

        chunks = load_corpus_chunks(self.base)

        self.assertEqual(chunks[0].title, "DOM Events")

    def test_answer_prefers_on_topic_non_deprecated_support(self) -> None:
        self._write_chunk(
            "next-app-router",
            "Use the App Router for layouts, nested routes, loading UI, and server components in new applications.",
            title="App Router Overview",
            url="https://nextjs.org/docs/app",
            source_id="nextjs",
            section_path=["App Router"],
        )
        self._write_chunk(
            "next-pages-router",
            "The Pages Router is the previous routing system and older applications may still use it.",
            title="Pages Router",
            url="https://nextjs.org/docs/pages",
            source_id="nextjs",
            deprecated=True,
            section_path=["Pages Router"],
        )

        harness = SinaiWebDevDemoHarness(load_corpus_chunks(self.base), support_threshold=0.45)
        result = harness.answer("How should I structure routing in a Next.js App Router app?", top_k=2)

        self.assertTrue(result.supported)
        self.assertEqual(result.citations[0].chunk_id, "next-app-router")
        self.assertAlmostEqual(result.citations[0].score, 0.4933, places=3)
        self.assertNotIn("do not have enough support", result.answer.lower())

    def test_answer_abstains_when_support_is_weak(self) -> None:
        self._write_chunk(
            "react-events",
            "React event handlers run in response to user interactions like clicks and input.",
            title="Responding to Events",
            url="https://react.dev/learn/responding-to-events",
            source_id="react",
        )

        harness = SinaiWebDevDemoHarness(load_corpus_chunks(self.base))
        result = harness.answer("Who won the 2018 World Cup final?", top_k=2)

        self.assertFalse(result.supported)
        self.assertEqual(result.citations, [])
        self.assertIn("do not have enough support", result.answer.lower())

    def test_format_demo_response_includes_citations_and_confidence(self) -> None:
        chunk = CorpusChunk(
            chunk_id="ts-events",
            text="Type React event handlers as React.ChangeEvent<HTMLInputElement> when reading input values.",
            title="Typing Events",
            url="https://www.typescriptlang.org/docs/handbook/jsx.html",
            source_id="typescript",
            priority="P0",
            content_type="reference",
            section_path=["JSX", "Typing Events"],
            is_deprecated=False,
        )
        harness = SinaiWebDevDemoHarness([chunk])
        result = harness.answer("How do I type a React event handler in TypeScript?", top_k=1)

        formatted = format_demo_response(result)

        self.assertIn("Supported: yes", formatted)
        self.assertIn("Confidence:", formatted)
        self.assertIn("Typing Events", formatted)
        self.assertIn("https://www.typescriptlang.org/docs/handbook/jsx.html", formatted)

    def test_load_production_model_infers_retrieval_config_and_supports_queries(self) -> None:
        checkpoint_path = Path(__file__).resolve().parents[1] / "artifacts" / "ignorance_1_2.7b_v340_neighborhood_posterior.pt"
        self.assertTrue(checkpoint_path.exists(), f"missing checkpoint: {checkpoint_path}")

        model, config, tokenizer, has_confidence_head = _STRICT_EVAL.load_model_for_demo(
            model_path=str(checkpoint_path),
            size=2_700_000_000,
            force_cpu=True,
        )

        self.assertTrue(has_confidence_head)
        self.assertGreater(config.max_seq_len, 0)

        query = "How do I type a React event handler in TypeScript?"
        query_tensor = tokenizer.batch_encode([query], config.max_seq_len, "cpu")
        with torch.no_grad():
            z_query_raw = model.encode(query_tensor)
            z_query = model.retrieval_project(z_query_raw)
            confidence = model.query_confidence(z_query_raw).item()

        self.assertEqual(z_query.shape[0], 1)
        self.assertGreater(z_query.shape[1], 0)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)

    def test_model_backed_harness_builds_index_in_batches(self) -> None:
        tokenizer = FakeTokenizer()
        chunks = [
            CorpusChunk(
                chunk_id=f"chunk-{idx}",
                text=f"React event handler typing example {idx}",
                title=f"Chunk {idx}",
                url=f"https://example.com/{idx}",
                source_id="react",
                priority="P0",
                content_type="reference",
                section_path=["Examples"],
                is_deprecated=False,
            )
            for idx in range(5)
        ]
        SinaiModelBackedDemoHarness(
            model=FakeModel(),
            config=FakeConfig(),
            tokenizer=tokenizer,
            chunks=chunks,
            device="cpu",
            has_confidence_head=True,
            support_threshold=-1.0,
            confidence_threshold=0.0,
            index_batch_size=2,
        )

        self.assertEqual(tokenizer.batch_sizes, [2, 2, 1])

    def test_model_backed_harness_retrieves_supported_chunk(self) -> None:
        checkpoint_path = Path(__file__).resolve().parents[1] / "artifacts" / "ignorance_1_2.7b_v340_neighborhood_posterior.pt"
        model, config, tokenizer, has_confidence_head = _STRICT_EVAL.load_model_for_demo(
            model_path=str(checkpoint_path),
            size=2_700_000_000,
            force_cpu=True,
        )

        chunks = [
            CorpusChunk(
                chunk_id="good",
                text="Type React event handlers as React.ChangeEventHandler<HTMLInputElement> so the input event is typed correctly.",
                title="Typing Events",
                url="https://example.com/typescript-events",
                source_id="typescript",
                priority="P0",
                content_type="reference",
                section_path=["Typing Events"],
                is_deprecated=False,
            ),
            CorpusChunk(
                chunk_id="bad",
                text="The FIFA World Cup final is a football match played to determine the winner of the tournament.",
                title="Football",
                url="https://example.com/football",
                source_id="mdn",
                priority="P1",
                content_type="guide",
                section_path=["Sports"],
                is_deprecated=False,
            ),
        ]
        harness = SinaiModelBackedDemoHarness(
            model=model,
            config=config,
            tokenizer=tokenizer,
            chunks=chunks,
            device="cpu",
            has_confidence_head=has_confidence_head,
            support_threshold=-1.0,
            confidence_threshold=0.0,
        )

        result = harness.answer("How do I type a React event handler in TypeScript?", top_k=2)

        self.assertTrue(result.supported)
        self.assertEqual(result.citations[0].chunk_id, "good")
        self.assertGreaterEqual(result.confidence, 0.0)

    def test_model_backed_harness_reranks_supported_reference_above_broader_on_topic_chunk(self) -> None:
        tokenizer = FakeTokenizer()
        chunks = [
            CorpusChunk(
                chunk_id="broad-react",
                text="React supports event handlers and lets you pass functions to JSX elements to respond to user interactions in apps.",
                title="Responding to Events",
                url="https://react.dev/learn/responding-to-events",
                source_id="react",
                priority="P1",
                content_type="guide",
                section_path=["Learn"],
                is_deprecated=False,
            ),
            CorpusChunk(
                chunk_id="typed-event",
                text="Type React event handlers as React.ChangeEventHandler<HTMLInputElement> so input change events are typed correctly in TypeScript.",
                title="Typing Events",
                url="https://example.com/typescript-events",
                source_id="typescript",
                priority="P0",
                content_type="reference",
                section_path=["Typing Events"],
                is_deprecated=False,
            ),
        ]
        harness = SinaiModelBackedDemoHarness(
            model=FakeModel(),
            config=FakeConfig(),
            tokenizer=tokenizer,
            chunks=chunks,
            device="cpu",
            has_confidence_head=True,
            support_threshold=-1.0,
            confidence_threshold=0.0,
            index_batch_size=2,
        )
        harness.index.search_text = lambda *args, **kwargs: RetrievalResult(
            ids=["broad-react", "typed-event"],
            embeddings=torch.zeros((2, 4), dtype=torch.float32),
            scores=torch.tensor([0.95, 0.75], dtype=torch.float32),
        )

        result = harness.answer("How do I type a React event handler in TypeScript?", top_k=2)

        self.assertTrue(result.supported)
        self.assertEqual(result.citations[0].chunk_id, "typed-event")

    def test_model_backed_harness_expands_shortlist_with_strong_lexical_candidates(self) -> None:
        tokenizer = FakeTokenizer()
        chunks = [
            CorpusChunk(
                chunk_id="broad-react",
                text="React supports event handlers and lets you pass functions to JSX elements to respond to user interactions in apps.",
                title="Responding to Events",
                url="https://react.dev/learn/responding-to-events",
                source_id="react",
                priority="P1",
                content_type="guide",
                section_path=["Learn"],
                is_deprecated=False,
            ),
            CorpusChunk(
                chunk_id="typed-event",
                text="When working with DOM events in React, you can extract a handler and type it explicitly as React.ChangeEvent<HTMLInputElement>.",
                title="DOM Events",
                url="https://react.dev/learn/typescript",
                source_id="react",
                priority="P0",
                content_type="reference",
                section_path=["TypeScript", "DOM Events"],
                is_deprecated=False,
            ),
        ]
        harness = SinaiModelBackedDemoHarness(
            model=FakeModel(),
            config=FakeConfig(),
            tokenizer=tokenizer,
            chunks=chunks,
            device="cpu",
            has_confidence_head=True,
            support_threshold=0.6,
            confidence_threshold=0.0,
            index_batch_size=2,
        )
        harness.index.search_text = lambda *args, **kwargs: RetrievalResult(
            ids=["broad-react"],
            embeddings=torch.zeros((1, 4), dtype=torch.float32),
            scores=torch.tensor([0.95], dtype=torch.float32),
        )

        result = harness.answer("How do I type a React event handler in TypeScript?", top_k=2)

        self.assertTrue(result.supported)
        self.assertEqual(result.citations[0].chunk_id, "typed-event")

    def test_model_backed_harness_prefers_dom_events_guidance_over_generic_typescript_chunk(self) -> None:
        tokenizer = FakeTokenizer()
        chunks = [
            CorpusChunk(
                chunk_id="use-callback",
                text="### `useCallback` {/*typing-usecallback*/}\n\n<Note>\n\n[React Compiler](/learn/react-compiler) automatically memoizes values and functions, reducing the need for manual `useCallback` calls. You can use the compiler to handle memoization automatically.\n\n</Note>\n\nThe [`useCallback`](/reference/react/useCallback) provide a stable reference to a function as long as the dependencies passed into the second parameter are the same. Like `useMemo`, the function's type is inferred from the return value of the function in the first parameter, and you can be more explicit by providing a type argument to the Hook.\n\nWhen working in TypeScript strict mode `useCallback` requires adding types for the parameters in your callback. Depending on your code-style preferences, you could use the `*EventHandler` functions from the React types to provide the type for the event handler at the same time as defining the callback.",
                title="`useCallback` {/*typing-usecallback*/}",
                url="https://react.dev/learn/typescript",
                source_id="react",
                priority="P0",
                content_type="tutorial",
                section_path=["src", "content", "learn"],
                is_deprecated=False,
            ),
            CorpusChunk(
                chunk_id="dom-events",
                text="### DOM Events {/*typing-dom-events*/}\n\nWhen working with DOM events in React, the type of the event can often be inferred from the event handler. However, when you want to extract a function to be passed to an event handler, you will need to explicitly set the type of the event.\n\nfunction handleChange(event: React.ChangeEvent<HTMLInputElement>) {\n  setValue(event.currentTarget.value);\n}\n\nIf you need to use an event that is not included in this list, you can use the `React.SyntheticEvent` type.",
                title="DOM Events {/*typing-dom-events*/}",
                url="https://react.dev/learn/typescript",
                source_id="react",
                priority="P0",
                content_type="tutorial",
                section_path=["src", "content", "learn"],
                is_deprecated=False,
            ),
        ]
        harness = SinaiModelBackedDemoHarness(
            model=FakeModel(),
            config=FakeConfig(),
            tokenizer=tokenizer,
            chunks=chunks,
            device="cpu",
            has_confidence_head=True,
            support_threshold=-1.0,
            confidence_threshold=0.0,
            index_batch_size=2,
        )
        harness.index.search_text = lambda *args, **kwargs: RetrievalResult(
            ids=["use-callback", "dom-events"],
            embeddings=torch.zeros((2, 4), dtype=torch.float32),
            scores=torch.tensor([0.92, 0.84], dtype=torch.float32),
        )

        result = harness.answer("How do I type a React event handler in TypeScript?", top_k=2)

        self.assertTrue(result.supported)
        self.assertEqual(result.citations[0].chunk_id, "dom-events")

    def test_model_backed_harness_prefers_dynamic_routes_docs_over_generic_app_router_chunk(self) -> None:
        tokenizer = FakeTokenizer()
        chunks = [
            CorpusChunk(
                chunk_id="generic-app-router",
                text="### `prefetch`\n\nPrefetching happens when a `<Link />` component enters the user's viewport. For dynamic routes, the partial route down to the nearest segment with a `loading.js` boundary will be prefetched.",
                title="`prefetch`",
                url="https://nextjs.org/docs/app/api-reference/components/link",
                source_id="nextjs",
                priority="P0",
                content_type="reference",
                section_path=["docs", "01-app", "03-api-reference", "02-components"],
                is_deprecated=False,
            ),
            CorpusChunk(
                chunk_id="dynamic-routes",
                text="### Dynamic routes\n\nParameterize segments with square brackets. Use `[segment]` for a single param, `[...segment]` for catch‑all, and `[[...segment]]` for optional catch‑all. Access values via the `params` prop.",
                title="Dynamic routes",
                url="https://nextjs.org/docs/app/getting-started/project-structure",
                source_id="nextjs",
                priority="P0",
                content_type="guide",
                section_path=["docs", "01-app", "01-getting-started"],
                is_deprecated=False,
            ),
        ]
        harness = SinaiModelBackedDemoHarness(
            model=FakeModel(),
            config=FakeConfig(),
            tokenizer=tokenizer,
            chunks=chunks,
            device="cpu",
            has_confidence_head=True,
            support_threshold=-1.0,
            confidence_threshold=0.0,
            index_batch_size=2,
        )
        harness.index.search_text = lambda *args, **kwargs: RetrievalResult(
            ids=["generic-app-router", "dynamic-routes"],
            embeddings=torch.zeros((2, 4), dtype=torch.float32),
            scores=torch.tensor([0.94, 0.82], dtype=torch.float32),
        )

        result = harness.answer("How does the Next.js App Router define dynamic routes?", top_k=2)

        self.assertTrue(result.supported)
        self.assertEqual(result.citations[0].chunk_id, "dynamic-routes")

    def test_model_backed_harness_fallback_prefers_dom_events_over_usecallback(self) -> None:
        tokenizer = FakeTokenizer()
        chunks = [
            CorpusChunk(
                chunk_id="use-callback",
                text="### `useCallback` {/*typing-usecallback*/}\n\nThe [`useCallback`](/reference/react/useCallback) provide a stable reference to a function. When working in TypeScript strict mode `useCallback` requires adding types for the parameters in your callback. You could use the `*EventHandler` functions from the React types to provide the type for the event handler at the same time as defining the callback.",
                title="`useCallback` {/*typing-usecallback*/}",
                url="https://react.dev/learn/typescript",
                source_id="react",
                priority="P0",
                content_type="tutorial",
                section_path=["src", "content", "learn"],
                is_deprecated=False,
            ),
            CorpusChunk(
                chunk_id="dom-events",
                text="### DOM Events {/*typing-dom-events*/}\n\nWhen working with DOM events in React, when you want to extract a function to be passed to an event handler, you need to explicitly type the event as `React.ChangeEvent<HTMLInputElement>`.",
                title="DOM Events {/*typing-dom-events*/}",
                url="https://react.dev/learn/typescript",
                source_id="react",
                priority="P0",
                content_type="tutorial",
                section_path=["src", "content", "learn"],
                is_deprecated=False,
            ),
        ]
        harness = SinaiModelBackedDemoHarness(
            model=FakeModel(),
            config=FakeConfig(),
            tokenizer=tokenizer,
            chunks=chunks,
            device="cpu",
            has_confidence_head=True,
            support_threshold=-1.0,
            confidence_threshold=0.0,
            index_batch_size=2,
        )
        harness.index.search_text = lambda *args, **kwargs: RetrievalResult(
            ids=[],
            embeddings=torch.zeros((0, 4), dtype=torch.float32),
            scores=torch.tensor([], dtype=torch.float32),
        )

        result = harness.answer("How do I type a React event handler in TypeScript?", top_k=2)

        self.assertTrue(result.supported)
        self.assertEqual(result.citations[0].chunk_id, "dom-events")

    def test_model_backed_harness_fallback_prefers_dynamic_routes_over_prefetch(self) -> None:
        tokenizer = FakeTokenizer()
        chunks = [
            CorpusChunk(
                chunk_id="prefetch",
                text="### `prefetch`\n\nPrefetching happens when a `<Link />` component enters the viewport. For dynamic routes, the partial route down to the nearest segment with a loading boundary will be prefetched.",
                title="`prefetch`",
                url="https://nextjs.org/docs/app/api-reference/components/link",
                source_id="nextjs",
                priority="P0",
                content_type="reference",
                section_path=["docs", "01-app", "03-api-reference", "02-components"],
                is_deprecated=False,
            ),
            CorpusChunk(
                chunk_id="dynamic-routes",
                text="### Dynamic routes\n\nParameterize segments with square brackets like `[slug]`. Use dynamic segments to define routes from data in the App Router.",
                title="Dynamic routes",
                url="https://nextjs.org/docs/app/getting-started/project-structure",
                source_id="nextjs",
                priority="P0",
                content_type="guide",
                section_path=["docs", "01-app", "01-getting-started"],
                is_deprecated=False,
            ),
        ]
        harness = SinaiModelBackedDemoHarness(
            model=FakeModel(),
            config=FakeConfig(),
            tokenizer=tokenizer,
            chunks=chunks,
            device="cpu",
            has_confidence_head=True,
            support_threshold=-1.0,
            confidence_threshold=0.0,
            index_batch_size=2,
        )
        harness.index.search_text = lambda *args, **kwargs: RetrievalResult(
            ids=[],
            embeddings=torch.zeros((0, 4), dtype=torch.float32),
            scores=torch.tensor([], dtype=torch.float32),
        )

        result = harness.answer("How does the Next.js App Router define dynamic routes?", top_k=2)

        self.assertTrue(result.supported)
        self.assertEqual(result.citations[0].chunk_id, "dynamic-routes")

    def test_model_backed_harness_corpus_like_fallback_prefers_dom_events_chunk(self) -> None:
        tokenizer = FakeTokenizer()
        chunks = [
            CorpusChunk(
                chunk_id="use-callback",
                text="### `useCallback` {/*typing-usecallback*/}\n\n<Note>\n\n[React Compiler](/learn/react-compiler) automatically memoizes values and functions, reducing the need for manual `useCallback` calls. You can use the compiler to handle memoization automatically.\n\n</Note>\n\nThe [`useCallback`](/reference/react/useCallback) provide a stable reference to a function as long as the dependencies passed into the second parameter are the same. Like `useMemo`, the function's type is inferred from the return value of the function in the first parameter, and you can be more explicit by providing a type argument to the Hook.\n\nWhen working in TypeScript strict mode `useCallback` requires adding types for the parameters in your callback. Depending on your code-style preferences, you could use the `*EventHandler` functions from the React types to provide the type for the event handler at the same time as defining the callback.",
                title="`useCallback` {/*typing-usecallback*/}",
                url="https://github.com/reactjs/react.dev/src/content/learn/typescript",
                source_id="react",
                priority="P0",
                content_type="tutorial",
                section_path=["src", "content", "learn"],
                is_deprecated=False,
            ),
            CorpusChunk(
                chunk_id="dom-events",
                text="### DOM Events {/*typing-dom-events*/}\n\nWhen working with DOM events in React, the type of the event can often be inferred from the event handler. However, when you want to extract a function to be passed to an event handler, you will need to explicitly set the type of the event.\n\nfunction handleChange(event: React.ChangeEvent<HTMLInputElement>) {\n  setValue(event.currentTarget.value);\n}\n\nIf you need to use an event that is not included in this list, you can use the `React.SyntheticEvent` type.",
                title="DOM Events {/*typing-dom-events*/}",
                url="https://github.com/reactjs/react.dev/src/content/learn/typescript",
                source_id="react",
                priority="P0",
                content_type="tutorial",
                section_path=["src", "content", "learn"],
                is_deprecated=False,
            ),
        ]
        harness = SinaiModelBackedDemoHarness(
            model=FakeModel(),
            config=FakeConfig(),
            tokenizer=tokenizer,
            chunks=chunks,
            device="cpu",
            has_confidence_head=True,
            support_threshold=-1.0,
            confidence_threshold=0.0,
            index_batch_size=2,
        )
        harness.index.search_text = lambda *args, **kwargs: RetrievalResult(
            ids=[],
            embeddings=torch.zeros((0, 4), dtype=torch.float32),
            scores=torch.tensor([], dtype=torch.float32),
        )

        result = harness.answer("How do I type a React event handler in TypeScript?", top_k=2)

        self.assertTrue(result.supported)
        self.assertEqual(result.citations[0].chunk_id, "dom-events")

    def test_model_backed_harness_corpus_like_fallback_prefers_dynamic_routes_chunks(self) -> None:
        tokenizer = FakeTokenizer()
        chunks = [
            CorpusChunk(
                chunk_id="prefetch",
                text="### `prefetch`\n\nPrefetching happens when a `<Link />` component enters the user's viewport (initially or through scroll). Next.js prefetches and loads the linked route (denoted by the `href`) and its data in the background. For dynamic routes, the partial route down to the nearest segment with a `loading.js` boundary will be prefetched.",
                title="`prefetch`",
                url="https://github.com/vercel/next.js/docs/01-app/03-api-reference/02-components/link",
                source_id="nextjs",
                priority="P0",
                content_type="reference",
                section_path=["docs", "01-app", "03-api-reference", "02-components"],
                is_deprecated=False,
            ),
            CorpusChunk(
                chunk_id="typed-links",
                text="### Statically Typed Links\n\nNext.js can statically type links to prevent typos and other errors when using `next/link`, improving type safety when navigating between pages. Works in both the Pages and App Router.",
                title="Statically Typed Links",
                url="https://github.com/vercel/next.js/docs/01-app/03-api-reference/05-config/02-typescript",
                source_id="nextjs",
                priority="P0",
                content_type="reference",
                section_path=["docs", "01-app", "03-api-reference", "05-config"],
                is_deprecated=False,
            ),
            CorpusChunk(
                chunk_id="dynamic-routes",
                text="### Dynamic routes\n\nParameterize segments with square brackets. Use `[segment]` for a single param, `[...segment]` for catch‑all, and `[[...segment]]` for optional catch‑all. Access values via the `params` prop.",
                title="Dynamic routes",
                url="https://github.com/vercel/next.js/docs/01-app/01-getting-started/02-project-structure",
                source_id="nextjs",
                priority="P0",
                content_type="guide",
                section_path=["docs", "01-app", "01-getting-started"],
                is_deprecated=False,
            ),
            CorpusChunk(
                chunk_id="dynamic-segment",
                text="## Creating a dynamic segment\n\nDynamic segments allow you to create routes that are generated from data. To create a dynamic segment, wrap the segment name in square brackets like `[slug]`.",
                title="Creating a dynamic segment",
                url="https://github.com/vercel/next.js/docs/01-app/01-getting-started/03-layouts-and-pages",
                source_id="nextjs",
                priority="P0",
                content_type="guide",
                section_path=["docs", "01-app", "01-getting-started"],
                is_deprecated=False,
            ),
        ]
        harness = SinaiModelBackedDemoHarness(
            model=FakeModel(),
            config=FakeConfig(),
            tokenizer=tokenizer,
            chunks=chunks,
            device="cpu",
            has_confidence_head=True,
            support_threshold=-1.0,
            confidence_threshold=0.0,
            index_batch_size=2,
        )
        harness.index.search_text = lambda *args, **kwargs: RetrievalResult(
            ids=[],
            embeddings=torch.zeros((0, 4), dtype=torch.float32),
            scores=torch.tensor([], dtype=torch.float32),
        )

        result = harness.answer("How does the Next.js App Router define dynamic routes?", top_k=3)

        self.assertTrue(result.supported)
        self.assertEqual(result.citations[0].chunk_id, "dynamic-routes")

    def test_model_backed_harness_corpus_like_fallback_prefers_canonical_app_dynamic_route_docs(self) -> None:
        tokenizer = FakeTokenizer()
        chunks = [
            CorpusChunk(
                chunk_id="loading-dynamic-routes",
                text="### Dynamic routes without `loading.tsx`\n\nWhen navigating to a dynamic route, the client must wait for the server response before showing the result. We recommend adding `loading.tsx` to dynamic routes to enable partial prefetching.",
                title="Dynamic routes without `loading.tsx`",
                url="https://github.com/vercel/next.js/docs/01-app/01-getting-started/04-linking-and-navigating",
                source_id="nextjs",
                priority="P0",
                content_type="guide",
                section_path=["docs", "01-app", "01-getting-started"],
                is_deprecated=False,
            ),
            CorpusChunk(
                chunk_id="pages-dynamic-routes",
                text="## Pages with Dynamic Routes\n\nNext.js supports pages with dynamic routes. For example, if you create a file called `pages/posts/[id].js`, then it will be accessible at `posts/1`, `posts/2`, etc.",
                title="Pages with Dynamic Routes",
                url="https://github.com/vercel/next.js/docs/02-pages/03-building-your-application/01-routing/01-pages-and-layouts",
                source_id="nextjs",
                priority="P0",
                content_type="guide",
                section_path=["docs", "02-pages", "03-building-your-application"],
                is_deprecated=False,
            ),
            CorpusChunk(
                chunk_id="route-handlers-dynamic-route-segments",
                text="### Dynamic Route Segments\n\nRoute Handlers can use Dynamic Segments to create request handlers from dynamic data. Example: `app/items/[slug]/route.ts`.",
                title="Dynamic Route Segments",
                url="https://github.com/vercel/next.js/docs/01-app/03-api-reference/03-file-conventions/route",
                source_id="nextjs",
                priority="P0",
                content_type="reference",
                section_path=["docs", "01-app", "03-api-reference", "03-file-conventions"],
                is_deprecated=False,
            ),
            CorpusChunk(
                chunk_id="dynamic-routes",
                text="### Dynamic routes\n\nParameterize segments with square brackets. Use `[segment]` for a single param, `[...segment]` for catch‑all, and `[[...segment]]` for optional catch‑all. Access values via the `params` prop.",
                title="Dynamic routes",
                url="https://github.com/vercel/next.js/docs/01-app/01-getting-started/02-project-structure",
                source_id="nextjs",
                priority="P0",
                content_type="guide",
                section_path=["docs", "01-app", "01-getting-started"],
                is_deprecated=False,
            ),
            CorpusChunk(
                chunk_id="dynamic-routes-duplicate",
                text="### Dynamic routes\n\nThis duplicate chunk should be deduplicated against the canonical dynamic-routes citation because it shares the same title and URL.",
                title="Dynamic routes",
                url="https://github.com/vercel/next.js/docs/01-app/01-getting-started/02-project-structure",
                source_id="nextjs",
                priority="P0",
                content_type="guide",
                section_path=["docs", "01-app", "01-getting-started"],
                is_deprecated=False,
            ),
            CorpusChunk(
                chunk_id="dynamic-segment",
                text="## Creating a dynamic segment\n\nDynamic segments allow you to create routes that are generated from data. To create a dynamic segment, wrap the segment name in square brackets like `[slug]`.",
                title="Creating a dynamic segment",
                url="https://github.com/vercel/next.js/docs/01-app/01-getting-started/03-layouts-and-pages",
                source_id="nextjs",
                priority="P0",
                content_type="guide",
                section_path=["docs", "01-app", "01-getting-started"],
                is_deprecated=False,
            ),
        ]
        harness = SinaiModelBackedDemoHarness(
            model=FakeModel(),
            config=FakeConfig(),
            tokenizer=tokenizer,
            chunks=chunks,
            device="cpu",
            has_confidence_head=True,
            support_threshold=-1.0,
            confidence_threshold=0.0,
            index_batch_size=2,
        )
        harness.index.search_text = lambda *args, **kwargs: RetrievalResult(
            ids=[],
            embeddings=torch.zeros((0, 4), dtype=torch.float32),
            scores=torch.tensor([], dtype=torch.float32),
        )

        result = harness.answer("How does the Next.js App Router define dynamic routes?", top_k=3)

        self.assertTrue(result.supported)
        self.assertEqual(result.citations[0].chunk_id, "dynamic-routes")
        self.assertEqual(len({(c.title, c.url) for c in result.citations}), len(result.citations))

    def test_model_backed_harness_corpus_like_fallback_prefers_catch_all_segment_docs(self) -> None:
        tokenizer = FakeTokenizer()
        chunks = [
            CorpusChunk(
                chunk_id="entrypoint-page",
                text="### Step 6: Create the Entrypoint Page\n\nCreate React App uses `src/index.tsx` as the entry point. In Next.js (App Router), each folder inside the `app` directory corresponds to a route, and each folder should have a `page.tsx` file.",
                title="Step 6: Create the Entrypoint Page",
                url="https://github.com/vercel/next.js/docs/01-app/02-guides/migrating/from-create-react-app",
                source_id="nextjs",
                priority="P0",
                content_type="guide",
                section_path=["docs", "01-app", "02-guides"],
                is_deprecated=False,
            ),
            CorpusChunk(
                chunk_id="redirects-i18n",
                text="### Redirects with i18n support\n\nWhen implementing redirects with internationalization in the App Router, you can include locales in redirects, but only as hardcoded paths. This does not support dynamic or per-locale redirects.",
                title="Redirects with i18n support",
                url="https://github.com/vercel/next.js/docs/01-app/03-api-reference/05-config/01-next-config-js/redirects",
                source_id="nextjs",
                priority="P0",
                content_type="reference",
                section_path=["docs", "01-app", "03-api-reference"],
                is_deprecated=False,
            ),
            CorpusChunk(
                chunk_id="catch-all-glossary",
                text="## Catch-all Segments\n\nDynamic route segments that can match multiple URL parts using the `[...folder]/page.js` syntax. These segments capture all remaining URL segments and are useful for implementing features like documentation sites or file browsers.",
                title="Catch-all Segments",
                url="https://github.com/vercel/next.js/docs/01-app/04-glossary",
                source_id="nextjs",
                priority="P0",
                content_type="reference",
                section_path=["docs", "01-app", "04-glossary"],
                is_deprecated=False,
            ),
            CorpusChunk(
                chunk_id="catch-all-file-conventions",
                text="### Catch-all Segments\n\nDynamic Segments can be extended to catch-all subsequent segments by adding an ellipsis inside the brackets `[...folderName]`. For example, `app/shop/[...slug]/page.js` will match `/shop/clothes`, `/shop/clothes/tops`, and `/shop/clothes/tops/t-shirts`.",
                title="Catch-all Segments",
                url="https://github.com/vercel/next.js/docs/01-app/03-api-reference/03-file-conventions/dynamic-routes",
                source_id="nextjs",
                priority="P0",
                content_type="reference",
                section_path=["docs", "01-app", "03-api-reference", "03-file-conventions"],
                is_deprecated=False,
            ),
            CorpusChunk(
                chunk_id="optional-catch-all",
                text="### Optional Catch-all Segments\n\nCatch-all Segments can be made optional by including the parameter in double square brackets: `[[...folderName]]`. For example, `app/shop/[[...slug]]/page.js` also matches `/shop`.",
                title="Optional Catch-all Segments",
                url="https://github.com/vercel/next.js/docs/01-app/03-api-reference/03-file-conventions/dynamic-routes",
                source_id="nextjs",
                priority="P0",
                content_type="reference",
                section_path=["docs", "01-app", "03-api-reference", "03-file-conventions"],
                is_deprecated=False,
            ),
        ]
        harness = SinaiModelBackedDemoHarness(
            model=FakeModel(),
            config=FakeConfig(),
            tokenizer=tokenizer,
            chunks=chunks,
            device="cpu",
            has_confidence_head=True,
            support_threshold=-1.0,
            confidence_threshold=0.0,
            index_batch_size=2,
        )
        harness.index.search_text = lambda *args, **kwargs: RetrievalResult(
            ids=[],
            embeddings=torch.zeros((0, 4), dtype=torch.float32),
            scores=torch.tensor([], dtype=torch.float32),
        )

        result = harness.answer("How do catch-all segments work in the Next.js App Router?", top_k=3)

        self.assertTrue(result.supported)
        self.assertIn(result.citations[0].chunk_id, {"catch-all-glossary", "catch-all-file-conventions", "optional-catch-all"})


if __name__ == "__main__":
    unittest.main()
