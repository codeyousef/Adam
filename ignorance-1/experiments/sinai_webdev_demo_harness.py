from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from src.models.jepa import JEPAModel, JEPAConfig
from src.utils.data import SimpleTokenizer
from src.utils.retrieval import VectorIndex


STOPWORDS = {
    'a', 'an', 'and', 'app', 'are', 'be', 'best', 'can', 'component', 'components', 'differ', 'difference',
    'do', 'does', 'final', 'for', 'from', 'go', 'handle', 'how', 'i', 'in', 'is', 'it', 'my', 'of', 'on',
    'or', 'should', 'structure', 'the', 'their', 'to', 'use', 'versus', 'vs', 'what', 'when', 'who', 'won',
    'work', 'you', 'your'
}

DOMAIN_TOKENS = {
    'react', 'next', 'nextjs', 'next.js', 'typescript', 'type', 'typing', 'javascript', 'node', 'nodejs',
    'vite', 'npm', 'pnpm', 'router', 'routing', 'component', 'components', 'server', 'client', 'fetch',
    'json', 'event', 'handler', 'handlers', 'workspace', 'workspaces', 'grid', 'flexbox', 'css', 'layout',
    'process', 'route', 'routes', 'app_router', 'pages_router', 'segment', 'segments', 'slug', 'params'
}


@dataclass(frozen=True)
class CorpusChunk:
    chunk_id: str
    text: str
    title: str
    url: str
    source_id: str
    priority: str
    content_type: str
    section_path: list[str]
    is_deprecated: bool = False


@dataclass(frozen=True)
class RankedChunk:
    chunk_id: str
    title: str
    url: str
    source_id: str
    score: float
    snippet: str
    section_path: list[str]


@dataclass(frozen=True)
class DemoAnswer:
    query: str
    supported: bool
    answer: str
    citations: list[RankedChunk]
    confidence: float
    used_model_evidence: bool = False


def _tokenize(text: str) -> set[str]:
    normalized = text.lower().replace('next.js', 'nextjs').replace('app router', 'app_router').replace('pages router', 'pages_router')
    return set(re.findall(r"[a-z0-9_\.]+", normalized))


def _content_tokens(text: str) -> set[str]:
    return {token for token in _tokenize(text) if len(token) >= 3 and token not in STOPWORDS}


def _query_has_domain_signal(query_tokens: set[str]) -> bool:
    return any(token in DOMAIN_TOKENS for token in query_tokens)


def _lexical_support_ok(query: str, text: str) -> bool:
    query_tokens = _content_tokens(query)
    text_tokens = _content_tokens(text)
    overlap_tokens = query_tokens & text_tokens
    if not query_tokens or not overlap_tokens:
        return False
    if _query_has_domain_signal(query_tokens) and not (overlap_tokens & DOMAIN_TOKENS):
        return False
    return True


def _normalized_overlap_score(query: str, text: str) -> float:
    query_tokens = _content_tokens(query)
    text_tokens = _content_tokens(text)
    if not query_tokens or not text_tokens:
        return 0.0
    overlap = len(query_tokens & text_tokens)
    return overlap / max(len(query_tokens), 1)


def _support_consistency_score(query: str, chunk: CorpusChunk) -> float:
    query_tokens = _content_tokens(query)
    text_tokens = _content_tokens(chunk.text)
    title_tokens = _content_tokens(chunk.title)
    section_tokens = _content_tokens(" ".join(chunk.section_path))
    overlap = query_tokens & (text_tokens | title_tokens | section_tokens)
    if not query_tokens or not overlap:
        return 0.0

    score = len(overlap) / max(len(query_tokens), 1)
    lowered_query = query.lower()
    lowered_text = chunk.text.lower()
    lowered_title = chunk.title.lower()
    lowered_section = " ".join(chunk.section_path).lower()
    combined = " ".join((lowered_title, lowered_section, lowered_text))

    if "typescript" in query_tokens and ("typescript" in text_tokens or "typescript" in title_tokens or chunk.source_id == "typescript"):
        score += 0.20
    if "react" in query_tokens and ("react" in text_tokens or "react" in title_tokens or chunk.source_id == "react"):
        score += 0.08
    if "nextjs" in query_tokens and (chunk.source_id == "nextjs" or "nextjs" in text_tokens or "nextjs" in title_tokens):
        score += 0.18
    if {"event", "handler"} <= query_tokens and ("event" in lowered_text or "handler" in lowered_text):
        score += 0.12
    if "type" in query_tokens or "typing" in query_tokens:
        if any(term in combined for term in ("type", "typing", "typed", "changeeventhandler", "changeevent")):
            score += 0.18
    if "app_router" in query_tokens and ("app router" in lowered_text or "app router" in lowered_title or "app router" in lowered_section):
        score += 0.20
    if "route" in query_tokens or "routes" in query_tokens or "routing" in query_tokens:
        if any(term in combined for term in ("route", "routes", "routing", "dynamic route", "dynamic routes", "dynamic segment", "dynamic segments", "[slug]", "[segment]", "[...segment]", "[[...segment]]")):
            score += 0.22
    if "event" in query_tokens and "handler" in query_tokens:
        if any(term in combined for term in ("dom events", "syntheticevent", "changeevent", "changeeventhandler")):
            score += 0.28
    if "dom events" in lowered_title and "event" in query_tokens and "handler" in query_tokens:
        score += 0.30
    if "extract" in lowered_text and "changeevent" in lowered_text and "handler" in lowered_text:
        score += 0.24
    if "dom" in query_tokens and "event" in query_tokens and "dom events" in combined:
        score += 0.20
    if "dynamic" in query_tokens and any(term in combined for term in ("dynamic route", "dynamic routes", "dynamic segment", "dynamic segments")):
        score += 0.26
    if any(phrase in lowered_title for phrase in ("dynamic routes", "dynamic route segments", "creating a dynamic segment")) and "dynamic" in query_tokens:
        score += 0.34
    if "dynamic" in query_tokens and "route" in query_tokens:
        if any(term in combined for term in ("prefetch", "link", "typed links")) and not any(term in combined for term in ("[slug]", "[segment]", "dynamic route", "dynamic routes", "dynamic segment", "dynamic segments")):
            score -= 0.35
    if any(token in query_tokens for token in ("slug", "segment", "segments")) and any(term in combined for term in ("[slug]", "[segment]", "[...segment]", "[[...segment]]")):
        score += 0.18
    if chunk.content_type == "reference":
        score += 0.06
    if chunk.priority == "P0":
        score += 0.04
    if chunk.is_deprecated:
        score -= 0.10
    return score


def _candidate_rank_score(query: str, chunk: CorpusChunk, retrieval_score: float) -> float:
    lexical_score = _normalized_overlap_score(query, chunk.text)
    support_score = _support_consistency_score(query, chunk)
    adjusted = 0.55 * float(retrieval_score) + 0.20 * lexical_score + 0.25 * support_score
    if chunk.priority == 'P0':
        adjusted += 0.04
    if chunk.is_deprecated:
        adjusted -= 0.10
    return adjusted


def _fallback_candidate_score(query: str, chunk: CorpusChunk) -> float:
    lexical_score = _normalized_overlap_score(query, chunk.text)
    support_score = _support_consistency_score(query, chunk)
    lowered_query = query.lower()
    lowered_title = chunk.title.lower()
    lowered_url = chunk.url.lower()
    lowered_section = " ".join(chunk.section_path).lower()
    lowered_text = chunk.text.lower()
    score = 0.25 * lexical_score + 0.75 * support_score

    if "event handler" in lowered_query and "typescript" in lowered_query:
        if "dom events" in lowered_title:
            score += 0.10
        if "usecallback" in lowered_title:
            score -= 0.05

    if "dynamic routes" in lowered_query or ("dynamic" in lowered_query and "route" in lowered_query):
        if any(phrase in lowered_title for phrase in ("dynamic routes", "dynamic route segments", "creating a dynamic segment")):
            score += 0.40
        if any(phrase in lowered_title for phrase in ("prefetch", "statically typed links", "link")):
            score -= 0.40
        if "app router" in lowered_query:
            if "02-pages" in lowered_url or "pages with dynamic routes" in lowered_title or "getstaticprops" in lowered_text:
                score -= 0.60
            if "loading.tsx" in lowered_title or "loading.tsx" in lowered_text:
                score -= 0.30
            if "file-conventions/route" in lowered_url or "route handlers" in lowered_text:
                score -= 0.20
            if "01-app" in lowered_url or "01-app" in lowered_section:
                score += 0.10

    if "catch-all" in lowered_query or ("catch" in lowered_query and "segments" in lowered_query):
        if any(phrase in lowered_title for phrase in ("catch-all segments", "optional catch-all segments")):
            score += 0.95
        if any(token in lowered_text for token in ("[...foldername]", "[[...foldername]]", "[...slug]", "[[...slug]]")):
            score += 0.25
        if "03-file-conventions/dynamic-routes" in lowered_url or "04-glossary" in lowered_url:
            score += 0.20
        if any(phrase in lowered_title for phrase in ("entypoint page", "entrypoint page", "redirects", "prefetch", "statically typed links")):
            score -= 0.50
        if "01-app" in lowered_url or "01-app" in lowered_section:
            score += 0.10

    if chunk.priority == 'P0':
        score += 0.04
    if chunk.is_deprecated:
        score -= 0.10
    return score


def load_corpus_chunks(base: Path) -> list[CorpusChunk]:
    base = Path(base)
    chunks_dir = base / "normalized" / "chunks"
    metadata_dir = base / "normalized" / "metadata"
    metadata_by_doc_id: dict[str, dict] = {}
    if metadata_dir.exists():
        for meta_path in sorted(metadata_dir.glob("*.json")):
            payload = json.loads(meta_path.read_text())
            metadata_by_doc_id[meta_path.stem] = payload

    chunks: list[CorpusChunk] = []
    for path in sorted(chunks_dir.glob("*.json")):
        payload = json.loads(path.read_text())
        doc_id = payload.get("doc_id") or path.stem.rsplit("__chunk", 1)[0]
        meta = metadata_by_doc_id.get(doc_id, {})
        payload_meta = payload.get("metadata", {})
        merged = {**meta, **payload_meta}
        text = payload.get("chunk_text") or payload.get("text") or ""
        chunks.append(
            CorpusChunk(
                chunk_id=payload["chunk_id"],
                text=text,
                title=payload.get("title") or merged.get("doc_title") or merged.get("title") or payload["chunk_id"],
                url=merged.get("url") or "",
                source_id=merged.get("source_id") or "unknown",
                priority=merged.get("priority") or payload.get("priority") or "P1",
                content_type=merged.get("content_type") or "guide",
                section_path=list(merged.get("section_path") or []),
                is_deprecated=bool(merged.get("is_deprecated") or merged.get("deprecated")),
            )
        )
    return chunks


class SinaiModelBackedDemoHarness:
    def __init__(
        self,
        model: JEPAModel,
        config: JEPAConfig,
        tokenizer: SimpleTokenizer,
        chunks: list[CorpusChunk],
        *,
        device: str,
        has_confidence_head: bool,
        support_threshold: float = 0.6,
        confidence_threshold: float = 0.35,
        lexical_weight: float = 0.35,
        index_batch_size: int = 256,
    ):
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        self.chunks = chunks
        self.device = device
        self.has_confidence_head = has_confidence_head
        self.support_threshold = float(support_threshold)
        self.confidence_threshold = float(confidence_threshold)
        self.lexical_weight = float(lexical_weight)
        self.index_batch_size = max(1, int(index_batch_size))
        self.index = self._build_index(chunks)
        self.chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}

    def _build_index(self, chunks: list[CorpusChunk]) -> VectorIndex:
        doc_ids = [chunk.chunk_id for chunk in chunks]
        embedding_batches: list[torch.Tensor] = []
        facet_batches: list[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, len(chunks), self.index_batch_size):
                batch = chunks[start:start + self.index_batch_size]
                texts = [chunk.text for chunk in batch]
                tensors = self.tokenizer.batch_encode(texts, self.config.max_seq_len, self.device)
                z_code_raw = self.model.encode(tensors)
                z_code = self.model.retrieval_project(z_code_raw)
                embedding_batches.append(z_code.cpu())
                if self.config.use_retrieval_facets:
                    facet_batches.append(self.model.retrieval_facets(z_code_raw, role="code").cpu())
        embeddings = torch.cat(embedding_batches, dim=0) if embedding_batches else torch.empty((0, 0), dtype=torch.float32)
        facet_embeddings = torch.cat(facet_batches, dim=0) if facet_batches else None
        return VectorIndex(
            doc_ids,
            embeddings,
            facet_embeddings=facet_embeddings,
        )

    def retrieve(self, query: str, *, top_k: int = 3) -> tuple[list[RankedChunk], float, bool]:
        query_tensor = self.tokenizer.batch_encode([query], self.config.max_seq_len, self.device)
        retrieval_k = min(max(int(top_k) * 16, 8), max(len(self.chunks), 1))
        with torch.no_grad():
            z_query_raw = self.model.encode(query_tensor)
            z_query = self.model.retrieval_project(z_query_raw)
            z_query_facets = self.model.retrieval_facets(z_query_raw, role="query") if self.config.use_retrieval_facets else None
            confidence = float(self.model.query_confidence(z_query_raw).item()) if self.has_confidence_head else 1.0
            retrieval = self.index.search_text(
                query,
                z_query.cpu(),
                k=retrieval_k,
                lexical_weight=self.lexical_weight,
                query_facets=z_query_facets.cpu() if z_query_facets is not None else None,
            )
        ranked: list[RankedChunk] = []
        used_model_evidence = False
        seen_chunk_ids: set[str] = set()
        for chunk_id, score in zip(retrieval.ids, retrieval.scores.tolist(), strict=False):
            chunk = self.chunk_by_id[chunk_id]
            if not _lexical_support_ok(query, chunk.text):
                continue
            adjusted = _candidate_rank_score(query, chunk, float(score))
            ranked.append(
                RankedChunk(
                    chunk_id=chunk.chunk_id,
                    title=chunk.title,
                    url=chunk.url,
                    source_id=chunk.source_id,
                    score=round(adjusted, 4),
                    snippet=chunk.text.replace('\n', ' ')[:240],
                    section_path=chunk.section_path,
                )
            )
            seen_chunk_ids.add(chunk.chunk_id)
            used_model_evidence = True
        fallback_candidates: list[tuple[float, CorpusChunk]] = []
        query_tokens = _content_tokens(query)
        for chunk in self.chunks:
            if chunk.chunk_id in seen_chunk_ids:
                continue
            if not _lexical_support_ok(query, chunk.text):
                continue
            overlap = len(query_tokens & _content_tokens(chunk.text))
            if overlap == 0:
                continue
            score = _fallback_candidate_score(query, chunk)
            fallback_candidates.append((score, chunk))
        fallback_candidates.sort(key=lambda item: item[0], reverse=True)
        for score, chunk in fallback_candidates[: max(int(top_k) * 8, 8)]:
            ranked.append(
                RankedChunk(
                    chunk_id=chunk.chunk_id,
                    title=chunk.title,
                    url=chunk.url,
                    source_id=chunk.source_id,
                    score=round(float(score), 4),
                    snippet=chunk.text.replace('\n', ' ')[:240],
                    section_path=chunk.section_path,
                )
            )
        ranked.sort(key=lambda item: item.score, reverse=True)
        deduped: list[RankedChunk] = []
        seen_keys: set[tuple[str, str, str]] = set()
        for item in ranked:
            key = (item.source_id, item.title, item.url)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            deduped.append(item)
        return deduped[:top_k], confidence, used_model_evidence

    def answer(self, query: str, *, top_k: int = 3) -> DemoAnswer:
        citations, confidence, used_model_evidence = self.retrieve(query, top_k=top_k)
        best = citations[0] if citations else None
        strong_fallback_support = (
            best is not None
            and best.score >= max(self.support_threshold, 0.60)
            and (
                len(citations) == 1
                or best.score >= 1.10
                or (
                    len(citations) >= 2
                    and citations[1].score >= max(self.support_threshold, best.score - 0.25)
                )
            )
        )
        supported = (
            best is not None
            and best.score >= self.support_threshold
            and confidence >= self.confidence_threshold
            and _query_has_domain_signal(_content_tokens(query))
            and (
                used_model_evidence
                or strong_fallback_support
            )
        )
        if not supported:
            return DemoAnswer(
                query=query,
                supported=False,
                answer="I do not have enough support in Sinai WebDev Base v1 to answer that safely.",
                citations=[],
                confidence=confidence,
                used_model_evidence=used_model_evidence,
            )
        answer = f"Supported by {best.title} ({best.source_id}): {best.snippet}"
        return DemoAnswer(
            query=query,
            supported=True,
            answer=answer,
            citations=citations,
            confidence=confidence,
            used_model_evidence=used_model_evidence,
        )


class SinaiWebDevDemoHarness:
    def __init__(self, chunks: list[CorpusChunk], *, support_threshold: float = 0.6):
        self.chunks = chunks
        self.support_threshold = float(support_threshold)

    def retrieve(self, query: str, *, top_k: int = 3) -> list[RankedChunk]:
        ranked: list[tuple[float, CorpusChunk]] = []
        query_tokens = _content_tokens(query)
        if not _query_has_domain_signal(query_tokens):
            return []
        for chunk in self.chunks:
            if not _lexical_support_ok(query, chunk.text):
                continue
            overlap_tokens = query_tokens & _content_tokens(chunk.text)
            overlap = len(overlap_tokens)
            if overlap == 0:
                continue
            score = overlap / max(len(query_tokens), 1)
            if 'app router' in query.lower() and 'app router' in chunk.text.lower():
                score += 0.12
            if 'pages router' in query.lower() and 'pages router' in chunk.text.lower():
                score += 0.12
            if chunk.priority == 'P0':
                score += 0.04
            if chunk.is_deprecated:
                score -= 0.10
            ranked.append((score, chunk))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return [
            RankedChunk(
                chunk_id=chunk.chunk_id,
                title=chunk.title,
                url=chunk.url,
                source_id=chunk.source_id,
                score=round(float(score), 4),
                snippet=chunk.text.replace('\n', ' ')[:240],
                section_path=chunk.section_path,
            )
            for score, chunk in ranked[:top_k]
        ]

    def answer(self, query: str, *, top_k: int = 3) -> DemoAnswer:
        citations = self.retrieve(query, top_k=top_k)
        best = citations[0] if citations else None
        if best is None or best.score < self.support_threshold:
            return DemoAnswer(
                query=query,
                supported=False,
                answer="I do not have enough support in Sinai WebDev Base v1 to answer that safely.",
                citations=[],
                confidence=0.0,
                used_model_evidence=False,
            )

        answer = f"Supported by {best.title} ({best.source_id}): {best.snippet}"
        return DemoAnswer(query=query, supported=True, answer=answer, citations=citations, confidence=1.0, used_model_evidence=False)


def format_demo_response(result: DemoAnswer) -> str:
    lines = [
        f"Query: {result.query}",
        f"Supported: {'yes' if result.supported else 'no'}",
        f"Confidence: {result.confidence:.4f}",
        f"Answer: {result.answer}",
    ]
    if result.citations:
        lines.append("Citations:")
        for citation in result.citations:
            section = " > ".join(citation.section_path) if citation.section_path else citation.title
            lines.append(
                f"- [{citation.source_id}] {citation.title} | {section} | {citation.url} | score={citation.score:.4f}"
            )
    return "\n".join(lines)
