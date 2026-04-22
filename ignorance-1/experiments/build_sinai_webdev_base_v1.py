#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

ROOT = Path('/mnt/Storage/Projects/catbelly_studio/sinai_webdev_base_v1')
SOURCES = ROOT / 'sources'
NORMALIZED_DOCS = ROOT / 'normalized' / 'documents'
NORMALIZED_CHUNKS = ROOT / 'normalized' / 'chunks'
NORMALIZED_META = ROOT / 'normalized' / 'metadata'
REPORTS = ROOT / 'reports'

for path in [SOURCES, NORMALIZED_DOCS, NORMALIZED_CHUNKS, NORMALIZED_META, REPORTS]:
    path.mkdir(parents=True, exist_ok=True)

NOW = datetime.now(timezone.utc).isoformat()


@dataclass
class SourceSpec:
    source_id: str
    url: str
    category: str
    authority_level: str
    content_type_default: str
    framework: str
    language: str
    version_label: str
    method: str
    clone_url: str | None = None
    subdir: str | None = None
    sparse_paths: list[str] | None = None


SPECS: list[SourceSpec] = [
    SourceSpec(
        source_id='mdn',
        url='https://github.com/mdn/content',
        category='official_docs',
        authority_level='canonical',
        content_type_default='reference',
        framework='web_platform',
        language='web',
        version_label='main',
        method='git',
        clone_url='https://github.com/mdn/content.git',
        subdir='mdn',
        sparse_paths=[
            'files/en-us/web/html',
            'files/en-us/web/css',
            'files/en-us/web/javascript',
            'files/en-us/web/http',
            'files/en-us/web/api',
        ],
    ),
    SourceSpec(
        source_id='typescript',
        url='https://github.com/microsoft/TypeScript-Website',
        category='official_docs',
        authority_level='canonical',
        content_type_default='guide',
        framework='typescript',
        language='typescript',
        version_label='main',
        method='git',
        clone_url='https://github.com/microsoft/TypeScript-Website.git',
        subdir='typescript',
        sparse_paths=[
            'packages/documentation/copy/en/handbook-v2',
            'packages/documentation/copy/en/reference',
            'packages/documentation/copy/en/release-notes',
            'packages/documentation/copy/en/project-config',
        ],
    ),
    SourceSpec(
        source_id='react',
        url='https://github.com/reactjs/react.dev',
        category='official_docs',
        authority_level='canonical',
        content_type_default='guide',
        framework='react',
        language='javascript',
        version_label='main',
        method='git',
        clone_url='https://github.com/reactjs/react.dev.git',
        subdir='react',
        sparse_paths=[
            'src/content/learn',
            'src/content/reference',
        ],
    ),
    SourceSpec(
        source_id='nextjs',
        url='https://github.com/vercel/next.js',
        category='official_docs',
        authority_level='canonical',
        content_type_default='guide',
        framework='nextjs',
        language='javascript',
        version_label='canary',
        method='git',
        clone_url='https://github.com/vercel/next.js.git',
        subdir='nextjs',
        sparse_paths=[
            'docs/01-app',
            'docs/02-pages',
            'docs/01-app/03-building-your-application',
            'docs/01-app/04-api-reference',
            'docs/02-pages/03-api-reference',
            'docs/01-app/01-getting-started',
        ],
    ),
    SourceSpec(
        source_id='node',
        url='https://github.com/nodejs/node',
        category='official_docs',
        authority_level='canonical',
        content_type_default='api',
        framework='node',
        language='javascript',
        version_label='main',
        method='git',
        clone_url='https://github.com/nodejs/node.git',
        subdir='node',
        sparse_paths=[
            'doc/api',
            'doc/guides',
            'doc/contributing',
        ],
    ),
    SourceSpec(
        source_id='vite',
        url='https://github.com/vitejs/vite',
        category='official_docs',
        authority_level='canonical',
        content_type_default='guide',
        framework='vite',
        language='javascript',
        version_label='main',
        method='git',
        clone_url='https://github.com/vitejs/vite.git',
        subdir='vite',
        sparse_paths=['docs'],
    ),
    SourceSpec(
        source_id='npm',
        url='https://github.com/npm/cli',
        category='official_docs',
        authority_level='canonical',
        content_type_default='reference',
        framework='npm',
        language='javascript',
        version_label='latest',
        method='git',
        clone_url='https://github.com/npm/cli.git',
        subdir='npm',
        sparse_paths=['docs/content'],
    ),
    SourceSpec(
        source_id='pnpm',
        url='https://github.com/pnpm/pnpm.io',
        category='official_docs',
        authority_level='canonical',
        content_type_default='reference',
        framework='pnpm',
        language='javascript',
        version_label='main',
        method='git',
        clone_url='https://github.com/pnpm/pnpm.io.git',
        subdir='pnpm',
        sparse_paths=['docs'],
    ),
]


def run(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, text=True, capture_output=True)


def ensure_sparse_clone(spec: SourceSpec) -> tuple[bool, str]:
    assert spec.clone_url and spec.subdir
    target = SOURCES / spec.subdir
    if target.exists() and (target / '.git').exists():
        return True, 'already_cloned'
    if target.exists():
        shutil.rmtree(target)
    cmd = ['git', 'clone', '--depth', '1', '--filter=blob:none', '--sparse', spec.clone_url, str(target)]
    result = run(cmd)
    if result.returncode != 0:
        return False, result.stderr.strip() or result.stdout.strip()
    if spec.sparse_paths:
        result2 = run(['git', 'sparse-checkout', 'set', *spec.sparse_paths], cwd=target)
        if result2.returncode != 0:
            return False, result2.stderr.strip() or result2.stdout.strip()
    return True, 'cloned'


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def clean_markdown(text: str) -> str:
    text = text.replace('\r\n', '\n')
    text = re.sub(r'^---\n.*?\n---\n', '', text, flags=re.S)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip() + '\n'


def title_from_text(text: str, fallback: str) -> str:
    for line in text.splitlines():
        if line.startswith('# '):
            return line[2:].strip()
    return fallback


def classify_content_type(path: Path) -> str:
    lower = '/'.join(path.parts).lower()
    if 'reference' in lower or '/api/' in lower:
        return 'reference'
    if 'learn' in lower or 'tutorial' in lower:
        return 'tutorial'
    if 'config' in lower:
        return 'config'
    if 'guide' in lower:
        return 'guide'
    return 'guide'


def priority_for(doc: str) -> str:
    d = doc.lower()
    p0_terms = [
        'fetch', 'json', 'server components', 'client components', 'useeffect',
        'hooks', 'routing', 'layout', 'grid', 'flexbox', 'narrowing',
        'generics', 'modules', 'process', 'http', 'fs', 'path', 'workspaces',
        'package.json', 'config', 'forms', 'events'
    ]
    if any(term in d for term in p0_terms):
        return 'P0'
    if 'deprecated' in d or 'legacy' in d or 'class component' in d:
        return 'P3'
    return 'P1'


def chunk_markdown(text: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, list[str]]] = []
    current_title = 'Introduction'
    current_lines: list[str] = []
    for line in text.splitlines():
        if re.match(r'^#{1,3} ', line):
            if current_lines:
                sections.append((current_title, current_lines))
            current_title = re.sub(r'^#{1,3} ', '', line).strip()
            current_lines = [line]
        else:
            current_lines.append(line)
    if current_lines:
        sections.append((current_title, current_lines))
    chunks: list[tuple[str, str]] = []
    for title, lines in sections:
        body = '\n'.join(lines).strip()
        if not body:
            continue
        words = body.split()
        if len(words) <= 900:
            chunks.append((title, body))
            continue
        start = 0
        idx = 1
        while start < len(words):
            end = min(start + 700, len(words))
            piece = ' '.join(words[start:end])
            suffix = f' (part {idx})' if start > 0 else ''
            chunks.append((title + suffix, piece))
            start = end
            idx += 1
    return chunks


def iter_docs(spec: SourceSpec) -> Iterable[Path]:
    target = SOURCES / spec.subdir
    for path in target.rglob('*'):
        if path.is_dir():
            continue
        if path.suffix.lower() not in {'.md', '.mdx', '.markdown'}:
            continue
        yield path


def normalize_source(spec: SourceSpec) -> dict:
    stats = {
        'documents': 0,
        'chunks': 0,
        'content_types': {},
        'priority_distribution': {},
        'deprecation_count': 0,
        'duplicate_count': 0,
        'failed': 0,
    }
    checksums: dict[str, str] = {}
    for path in iter_docs(spec):
        try:
            raw = path.read_text(encoding='utf-8', errors='ignore')
        except Exception:
            stats['failed'] += 1
            continue
        cleaned = clean_markdown(raw)
        checksum = sha256_text(cleaned)
        if checksum in checksums:
            stats['duplicate_count'] += 1
            continue
        checksums[checksum] = str(path)
        rel = path.relative_to(SOURCES / spec.subdir)
        doc_id = f"{spec.source_id}__{'__'.join(rel.with_suffix('').parts)}"
        title = title_from_text(cleaned, rel.stem)
        content_type = classify_content_type(rel)
        priority = priority_for(title + '\n' + cleaned[:2000])
        deprecated = 'deprecated' in cleaned.lower() or 'legacy' in cleaned.lower()
        if deprecated:
            stats['deprecation_count'] += 1
        section_path = list(rel.with_suffix('').parts[:-1])
        canonical_url = spec.url.rstrip('/') + '/' + '/'.join(rel.with_suffix('').parts)
        chunks = chunk_markdown(cleaned)
        chunk_ids: list[str] = []
        for i, (chunk_title, chunk_text) in enumerate(chunks, start=1):
            chunk_id = f'{doc_id}__chunk{i:03d}'
            chunk_ids.append(chunk_id)
            chunk_payload = {
                'chunk_id': chunk_id,
                'doc_id': doc_id,
                'title': chunk_title,
                'text': chunk_text,
                'priority': priority,
            }
            (NORMALIZED_CHUNKS / f'{chunk_id}.json').write_text(json.dumps(chunk_payload, ensure_ascii=False, indent=2))
        metadata = {
            'source_id': spec.source_id,
            'source_type': spec.category,
            'authority_level': spec.authority_level,
            'doc_title': title,
            'url': canonical_url,
            'section_path': section_path,
            'version_label': spec.version_label,
            'last_updated': '',
            'content_type': content_type,
            'framework': spec.framework,
            'language': spec.language,
            'topic_tags': [spec.framework, content_type, priority],
            'fetched_at': NOW,
            'checksum': checksum,
            'chunk_ids': chunk_ids,
            'priority': priority,
            'deprecated': deprecated,
            'source_path': str(rel),
        }
        doc_payload = {
            'doc_id': doc_id,
            'raw_path': str(path),
            'clean_markdown': cleaned,
            'metadata': metadata,
        }
        (NORMALIZED_DOCS / f'{doc_id}.json').write_text(json.dumps(doc_payload, ensure_ascii=False, indent=2))
        (NORMALIZED_META / f'{doc_id}.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
        stats['documents'] += 1
        stats['chunks'] += len(chunks)
        stats['content_types'][content_type] = stats['content_types'].get(content_type, 0) + 1
        stats['priority_distribution'][priority] = stats['priority_distribution'].get(priority, 0) + 1
    return stats


def main() -> None:
    fetch_report = {'fetched_urls': [], 'failed_urls': [], 'retries': [], 'canonicalized_urls': [], 'timestamps': {'started_at': NOW}}
    skipped = []
    source_stats = {}
    manifest_sources = []
    dedupe_report = {}

    for spec in SPECS:
        ok, detail = ensure_sparse_clone(spec)
        if not ok:
            fetch_report['failed_urls'].append({'source_id': spec.source_id, 'url': spec.url, 'error': detail})
            skipped.append({'source_id': spec.source_id, 'url': spec.url, 'reason': 'clone_failed', 'detail': detail})
            continue
        fetch_report['fetched_urls'].append({'source_id': spec.source_id, 'url': spec.url, 'detail': detail})
        fetch_report['canonicalized_urls'].append({'source_id': spec.source_id, 'url': spec.url})
        stats = normalize_source(spec)
        source_stats[spec.source_id] = stats
        dedupe_report[spec.source_id] = {'duplicate_count': stats['duplicate_count']}
        manifest_sources.append({
            'source_id': spec.source_id,
            'url': spec.url,
            'fetch_method': spec.method,
            'version_label': spec.version_label,
            'status': 'complete',
            'document_count': stats['documents'],
            'chunk_count': stats['chunks'],
        })

    fetch_report['timestamps']['finished_at'] = datetime.now(timezone.utc).isoformat()
    manifest = {
        'corpus_id': 'sinai_webdev_base_v1',
        'created_at': NOW,
        'source_count': len(manifest_sources),
        'sources': manifest_sources,
    }

    (ROOT / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    (REPORTS / 'fetch_report.json').write_text(json.dumps(fetch_report, ensure_ascii=False, indent=2))
    (REPORTS / 'skipped_urls.json').write_text(json.dumps(skipped, ensure_ascii=False, indent=2))
    (REPORTS / 'dedupe_report.json').write_text(json.dumps(dedupe_report, ensure_ascii=False, indent=2))
    (REPORTS / 'source_stats.json').write_text(json.dumps(source_stats, ensure_ascii=False, indent=2))
    print(json.dumps({'manifest': manifest, 'source_stats': source_stats}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
