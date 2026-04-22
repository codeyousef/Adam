#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.sinai_webdev_demo_harness import (
    SinaiModelBackedDemoHarness,
    SinaiWebDevDemoHarness,
    format_demo_response,
    load_corpus_chunks,
)
_STRICT_EVAL_PATH = ROOT / 'test_2.7b.py'
_STRICT_EVAL_SPEC = importlib.util.spec_from_file_location('strict_eval_test_2_7b', _STRICT_EVAL_PATH)
assert _STRICT_EVAL_SPEC and _STRICT_EVAL_SPEC.loader
_STRICT_EVAL_MODULE = importlib.util.module_from_spec(_STRICT_EVAL_SPEC)
_STRICT_EVAL_SPEC.loader.exec_module(_STRICT_EVAL_MODULE)
load_model_for_demo = _STRICT_EVAL_MODULE.load_model_for_demo

DEFAULT_CORPUS = Path('/mnt/Storage/Projects/catbelly_studio/sinai_webdev_base_v1')
DEFAULT_MODEL = Path('artifacts/ignorance_1_2.7b_v340_neighborhood_posterior_taxonomy_coverage.pt')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run the Sinai WebDev demo harness over the built official-docs corpus.')
    parser.add_argument('query', nargs='?', help='Question to ask Sinai WebDev Base v1')
    parser.add_argument('--corpus', type=Path, default=DEFAULT_CORPUS, help='Path to Sinai WebDev Base v1 corpus root')
    parser.add_argument('--top-k', type=int, default=3, help='How many citations to include')
    parser.add_argument('--threshold', type=float, default=0.6, help='Support threshold for abstention')
    parser.add_argument('--confidence-threshold', type=float, default=0.35, help='Model confidence threshold for abstention')
    parser.add_argument('--model-path', type=Path, default=DEFAULT_MODEL, help='Path to Sinai checkpoint')
    parser.add_argument('--size', type=int, default=2_700_000_000, help='Nominal model size for config reconstruction')
    parser.add_argument('--force-cpu', action='store_true', help='Load checkpoint on CPU')
    parser.add_argument('--lexical-only', action='store_true', help='Use heuristic retrieval without loading the model')
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if not args.query:
        parser.error('query is required')

    chunks = load_corpus_chunks(args.corpus)
    if args.lexical_only:
        harness = SinaiWebDevDemoHarness(chunks, support_threshold=args.threshold)
    else:
        model, config, tokenizer, has_confidence_head = load_model_for_demo(
            model_path=str(args.model_path),
            size=args.size,
            force_cpu=args.force_cpu,
        )
        device = next(model.parameters()).device.type
        harness = SinaiModelBackedDemoHarness(
            model=model,
            config=config,
            tokenizer=tokenizer,
            chunks=chunks,
            device=device,
            has_confidence_head=has_confidence_head,
            support_threshold=args.threshold,
            confidence_threshold=args.confidence_threshold,
        )
    result = harness.answer(args.query, top_k=args.top_k)
    print(format_demo_response(result))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
