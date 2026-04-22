#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path('/mnt/Storage/Projects/catbelly_studio')
BASE = ROOT / 'sinai_webdev_base_v1'
DOCS_DIR = BASE / 'normalized' / 'documents'
OUT = BASE / 'reports' / 'demo_smoke_report.json'

TARGET_QUESTIONS = [
    'When should I use a Server Component versus a Client Component in Next.js?',
    'How do I type a React event handler in TypeScript?',
    'What is the difference between useEffect and event handlers in React?',
    'How do npm and pnpm workspaces differ operationally?',
    'What does Node process expose?',
    'How do I use fetch and handle JSON errors?',
    'How does CSS Grid differ from Flexbox for layout?',
    'How should I structure routing in a Next.js App Router app?',
]


@dataclass
class Match:
    doc_id: str
    title: str
    source_id: str
    url: str
    score: float
    snippet: str


def tokenize(text: str) -> set[str]:
    import re
    return set(re.findall(r'[a-z0-9_]+', text.lower()))


def load_docs() -> list[dict[str, Any]]:
    docs = []
    for path in DOCS_DIR.glob('*.json'):
        docs.append(json.loads(path.read_text()))
    return docs


def score_query(query: str, doc: dict[str, Any]) -> float:
    q = tokenize(query)
    text = doc['clean_markdown']
    d = tokenize(text[:12000])
    if not q or not d:
        return 0.0
    overlap = len(q & d)
    union = len(q | d)
    base = overlap / max(union, 1)
    meta = doc['metadata']
    bonus = 0.0
    if meta.get('source_id') == 'nextjs' and 'next.js' in query.lower():
        bonus += 0.08
    if meta.get('source_id') == 'react' and 'react' in query.lower():
        bonus += 0.08
    if meta.get('source_id') == 'typescript' and 'typescript' in query.lower():
        bonus += 0.08
    if meta.get('source_id') == 'node' and 'node' in query.lower():
        bonus += 0.08
    if meta.get('source_id') in {'npm', 'pnpm'} and 'workspace' in query.lower():
        bonus += 0.05
    if meta.get('content_type') == 'reference':
        bonus += 0.02
    if meta.get('priority') == 'P0':
        bonus += 0.03
    return base + bonus


def top_matches(query: str, docs: list[dict[str, Any]], k: int = 5) -> list[Match]:
    scored = []
    for doc in docs:
        score = score_query(query, doc)
        if score <= 0:
            continue
        scored.append((score, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    matches = []
    for score, doc in scored[:k]:
        text = doc['clean_markdown'].replace('\n', ' ')[:320]
        meta = doc['metadata']
        matches.append(Match(
            doc_id=doc['doc_id'],
            title=meta['doc_title'],
            source_id=meta['source_id'],
            url=meta['url'],
            score=round(score, 4),
            snippet=text,
        ))
    return matches


def main() -> None:
    docs = load_docs()
    report = {
        'corpus_id': 'sinai_webdev_base_v1',
        'document_count': len(docs),
        'questions': [],
        'harness_assessment': {
            'needs_harness': True,
            'current_state': 'corpus_built_but_no_demo_specific_supported_answering_layer',
            'minimum_needed': [
                'query encoder over corpus chunks/documents',
                'retrieval ranking over Sinai WebDev Base v1',
                'citation formatting from metadata.url + section_path',
                'support threshold for abstention',
                'answer composer constrained to retrieved support'
            ],
        },
    }
    for query in TARGET_QUESTIONS:
        matches = top_matches(query, docs)
        report['questions'].append({
            'query': query,
            'match_count': len(matches),
            'top_matches': [m.__dict__ for m in matches],
            'supported': bool(matches and matches[0].score >= 0.06),
        })
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
