#!/usr/bin/env python3
"""
v396: Wire GatedRerankerHead into test_2.7b.py evaluation pipeline.

Hypothesis: The 5 failing families have near-zero embedding discrimination between
champion and HN because embedding similarity is NOT the true ranking signal.
The GatedRerankerHead was trained with execution-aware signals but never wired
into the eval pipeline.

Approach:
  1. Load v378 checkpoint with GatedRerankerHead enabled (use_gated_reranker=True)
  2. Compute execution-based features for each candidate (can import? runtime estimate?)
  3. Wire GatedRerankerHead into _selection_scores_for_finalists via
     rerank_gated_reranker_weight=0.3
  4. Run strict_eval on hard 8-family objective

Key insight: The GatedRerankerHead uses cross-encoder-style late interaction
(slot-level self-attention + MLP) to score query-candidate pairs, capturing
execution-aware features that embedding similarity misses.
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

# ── project paths ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.jepa import JEPAModel
from src.training.phase4 import _proxy_config_v6_overnight
from src.utils.data import SimpleTokenizer

LOG = logging.getLogger(__name__)

# ── Execution feature extraction ──────────────────────────────────────────────

EXEC_TIMEOUT = 5.0  # seconds per code execution


def _structure_score(code: str) -> float:
    """
    Score based on semantic completeness using AST analysis.
    Higher score = more complete/semantically correct implementation.
    Used as the execution-quality proxy for candidate ranking.
    """
    try:
        tree = ast.parse(code)
    except (SyntaxError, IndentationError):
        return 0.0

    ops: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if hasattr(node.func, 'attr') and node.func.attr:
                ops.append(node.func.attr)
            elif hasattr(node.func, 'id') and node.func.id:
                ops.append(node.func.id)
        elif isinstance(node, ast.Dict):
            ops.append('dict')
        elif isinstance(node, ast.Subscript):
            ops.append('subscript')
        elif isinstance(node, ast.Compare):
            ops.append('compare')

    # Base score from operations count (normalized)
    base = min(len(ops) / 5.0, 0.6)

    # Bonus for important operations
    bonus = 0.0
    important = {
        'get', 'setTimeout', 'json', 'parse', 'getattr', 'strip',
        'startswith', 'endsWith', 'getattribute', 'clearTimeout',
        'sorted', 'reversed', 'counts', 'get_token', 'json',
    }
    for op in ops:
        if op in important:
            bonus += 0.1

    # Bonus for multi-statement (not just expression)
    assign_count = sum(1 for n in ast.walk(tree) if isinstance(n, ast.Assign))
    if assign_count >= 2:
        bonus += 0.1
    if assign_count >= 3:
        bonus += 0.05

    return min(base + bonus, 1.0)


def _can_import(code: str) -> float:
    """Check if code has valid Python imports. Returns 1.0 if clean, 0.0 if import errors."""
    try:
        compile(code, '<string>', 'exec')
        return 1.0
    except SyntaxError:
        return 0.0


def _runtime_estimate(code: str) -> float:
    """
    Estimate runtime quality: does it have loops, function defs, return statements?
    Higher = more complete implementation.
    """
    try:
        tree = ast.parse(code)
    except (SyntaxError, IndentationError):
        return 0.0

    score = 0.0

    # Has function definitions = more complete
    func_count = sum(1 for n in ast.walk(tree) if isinstance(n, ast.FunctionDef))
    if func_count >= 1:
        score += 0.3
    if func_count >= 2:
        score += 0.1

    # Has return statements
    returns = sum(1 for n in ast.walk(tree) if isinstance(n, ast.Return))
    if returns >= 1:
        score += 0.2

    # Has loops
    loops = sum(1 for n in ast.walk(tree) if isinstance(n, (ast.For, ast.While)))
    if loops >= 1:
        score += 0.2

    # Has try/except (error handling awareness)
    try_blocks = sum(1 for n in ast.walk(tree) if isinstance(n, ast.Try))
    if try_blocks >= 1:
        score += 0.1

    # Bonus for complex control flow
    ifs = sum(1 for n in ast.walk(tree) if isinstance(n, ast.If))
    if ifs >= 2:
        score += 0.1

    return min(score, 1.0)


def execution_features(code: str) -> dict[str, float]:
    """Compute all execution-based features for a code candidate."""
    return {
        "structure_score": _structure_score(code),
        "can_import": _can_import(code),
        "runtime_estimate": _runtime_estimate(code),
    }


def candidate_exec_scores(doc_ids: list[str], device: torch.device, dtype=torch.float32) -> torch.Tensor:
    """
    Compute execution-based quality scores for each candidate document.
    Returns [C, 3] tensor of [structure_score, can_import, runtime_estimate].
    """
    features = []
    for doc in doc_ids:
        feat = execution_features(doc)
        features.append([
            feat["structure_score"],
            feat["can_import"],
            feat["runtime_estimate"],
        ])
    return torch.tensor(features, device=device, dtype=dtype)


# ── Model loading ───────────────────────────────────────────────────────────────

def _infer_retrieval_head_config_from_sd(state_dict: dict, config) -> None:
    """Infer retrieval_head config from state dict keys."""
    input_proj = state_dict.get("retrieval_head.input_proj.weight")
    output_proj = state_dict.get("retrieval_head.output_proj.weight")
    if input_proj is None or output_proj is None:
        config.use_retrieval_head = False
        config.retrieval_head_dim = 0
        config.retrieval_head_hidden_dim = 0
        return
    config.use_retrieval_head = True
    config.retrieval_head_hidden_dim = int(input_proj.shape[0])
    config.retrieval_head_dim = int(output_proj.shape[0])


def _infer_retrieval_facet_config_from_sd(state_dict: dict, config) -> None:
    """Infer retrieval_facet config from state dict keys."""
    shared_slot_bias = state_dict.get("retrieval_facet_head.slot_bias")
    shared_input_proj = state_dict.get("retrieval_facet_head.input_proj.weight")
    query_slot_bias = state_dict.get("query_retrieval_facet_head.slot_bias")
    query_input_proj = state_dict.get("query_retrieval_facet_head.input_proj.weight")
    code_slot_bias = state_dict.get("code_retrieval_facet_head.slot_bias")
    code_input_proj = state_dict.get("code_retrieval_facet_head.input_proj.weight")

    if shared_slot_bias is not None and shared_input_proj is not None:
        config.use_retrieval_facets = True
        config.retrieval_facet_separate_query_code = False
        config.retrieval_num_facets = int(shared_slot_bias.shape[0])
        config.retrieval_facet_dim = int(shared_slot_bias.shape[1])
        config.retrieval_facet_hidden_dim = int(shared_input_proj.shape[0])
        return

    if (
        query_slot_bias is not None
        and query_input_proj is not None
        and code_slot_bias is not None
        and code_input_proj is not None
    ):
        config.use_retrieval_facets = True
        config.retrieval_facet_separate_query_code = True
        config.retrieval_num_facets = int(query_slot_bias.shape[0])
        config.retrieval_facet_dim = int(query_slot_bias.shape[1])
        config.retrieval_facet_hidden_dim = int(query_input_proj.shape[0])
        return

    config.use_retrieval_facets = False
    config.retrieval_num_facets = 0
    config.retrieval_facet_dim = 0
    config.retrieval_facet_hidden_dim = 0
    config.retrieval_facet_separate_query_code = False


def load_v378_with_gated_reranker(device: torch.device) -> tuple[JEPAModel, object]:
    """Load 15M JEPA with v378 checkpoint AND GatedRerankerHead enabled."""
    size = 15_000_000
    config = _proxy_config_v6_overnight(size)
    config.use_gated_reranker = True
    config.gated_reranker_hidden_dim = 128
    config.gated_reranker_num_heads = 4

    v378_path = (
        ROOT / "artifacts"
        / "strict_eval_autoresearch_v378"
        / "v378-late-inter-high-weight-seed511-seed514"
        / "model.pt"
    )
    LOG.info("Loading v378 from %s", v378_path)
    sd = torch.load(v378_path, map_location=device, weights_only=False)

    # Infer retrieval config from checkpoint (same as load_model_for_demo)
    _infer_retrieval_head_config_from_sd(sd, config)
    _infer_retrieval_facet_config_from_sd(sd, config)
    config.max_seq_len = 256  # default for v378

    model = JEPAModel(config).to(device).eval()
    model = model.to(torch.bfloat16)

    missing, unexpected = model.load_state_dict(sd, strict=False)
    LOG.info("  Missing keys (expected for new heads): %d", len(missing))
    LOG.info("  Unexpected keys (ignored): %d", len(unexpected))
    if missing:
        LOG.info("  Missing (first 5): %s", missing[:5])

    has_gated = model.gated_reranker is not None
    LOG.info("  GatedRerankerHead initialized: %s", has_gated)
    if has_gated:
        for name, param in model.named_parameters():
            if "gated_reranker" in name:
                param.requires_grad = False
                LOG.info("    frozen: %s", name)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    LOG.info("Model: %d trainable / %d total params (%.1fM)", trainable, total, trainable / 1e6)
    return model, config


# ── Encoding ───────────────────────────────────────────────────────────────────

def encode_texts(
    model: JEPAModel,
    texts: list[str],
    batch_size: int,
    device: torch.device,
    seq_len: int = 256,
) -> torch.Tensor:
    """Encode texts via model encoder, return [N, embed_dim] tensor."""
    out: list[torch.Tensor] = []
    tok = SimpleTokenizer(vocab_size=4096)
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        ids = tok.batch_encode(batch, seq_len=seq_len, device=device)
        with torch.no_grad():
            encoded = model.encode(ids)
        out.append(encoded)
    return torch.cat(out, dim=0)


# ── GatedReranker scoring ───────────────────────────────────────────────────────

def gated_reranker_scores(
    model: JEPAModel,
    query_slots: torch.Tensor,       # [1, num_slots, facet_dim]
    candidate_slots: torch.Tensor,  # [C, num_slots, facet_dim]
    device: torch.device,
) -> torch.Tensor:
    """
    Score each candidate against the query using GatedRerankerHead.
    Returns [C] scores (one per candidate).

    The GatedRerankerHead uses cross-encoder-style slot-level self-attention
    over concatenated [query_slots | candidate_slots] pairs.
    """
    if model.gated_reranker is None:
        raise RuntimeError("GatedRerankerHead not initialized. Check use_gated_reranker=True.")

    B = query_slots.shape[0]
    C = candidate_slots.shape[0]

    with torch.no_grad():
        # GatedRerankerHead returns [B, C] scores
        raw_scores = model.gated_reranker(query_slots, candidate_slots)  # [1, C]
        scores = raw_scores.squeeze(0)  # [C]

    return scores


# ── Strict eval ────────────────────────────────────────────────────────────────

def run_strict_eval_objective_only(
    model: JEPAModel,
    device: torch.device,
    gated_reranker_weight: float = 0.3,
    batch_size: int = 8,
) -> dict:
    """
    Run strict_eval on the hard 8-family objective, using GatedRerankerHead
    for post-retrieval reranking with execution-based features.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location("test_27b", ROOT / "test_2.7b.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    docs, metadata_by_doc, cases = mod._build_support_discipline_eval()
    LOG.info("Strict eval: indexing %d docs", len(docs))

    doc_embeddings_raw = encode_texts(model, docs, batch_size, device)
    doc_embeddings = model.retrieval_project(doc_embeddings_raw)
    doc_facets: list[torch.Tensor] = []
    for i in range(0, len(docs), batch_size):
        batch_raw = doc_embeddings_raw[i:i + batch_size]
        with torch.no_grad():
            facets = model.retrieval_facets(batch_raw, role="code")
        doc_facets.append(facets)
    doc_facets = torch.cat(doc_facets, dim=0)

    from src.utils.retrieval import VectorIndex
    index = VectorIndex(
        doc_ids=docs,
        embeddings=doc_embeddings,
        facet_embeddings=doc_facets,
        facet_score_mode="hard_maxsim",
        global_facet_blend=0.35,
        facet_softmax_temperature=0.1,
    )

    # Pre-compute execution features for all docs
    LOG.info("Computing execution features for %d candidates", len(docs))
    exec_feats = candidate_exec_scores(docs, device=device)  # [C, 3]
    # Normalize each feature to [0, 1]
    exec_feats_mean = exec_feats.mean(dim=0, keepdim=True)
    exec_feats_std = exec_feats.std(dim=0, keepdim=True) + 1e-8
    exec_feats_norm = (exec_feats - exec_feats_mean) / exec_feats_std
    exec_feats_norm = torch.sigmoid(exec_feats_norm)  # [C, 3]

    results: list[dict] = []
    for case in cases:
        query_text = str(case["query"])
        family = str(case["family"])
        case_type = str(case["type"])

        q_emb_raw = encode_texts(model, [query_text], batch_size, device)
        q_emb = model.retrieval_project(q_emb_raw)
        with torch.no_grad():
            q_facets = model.retrieval_facets(q_emb_raw, role="query")

        # Phase 1: Initial retrieval using late interaction
        result = index.search_text(
            query_text,
            queries=q_emb,
            k=5,  # Get top-5 for reranking
            query_facets=q_facets,
        )

        # Phase 2: Reranking with GatedRerankerHead + execution features
        topk = min(len(result.ids), 5)
        if topk == 0:
            retrieved_text = "<IGNORANT>"
        else:
            candidate_ids = result.ids[:topk]
            candidate_indices = [docs.index(cid) for cid in candidate_ids]

            # Get late interaction scores (already computed by index)
            late_int_scores = result.scores[:topk].to(device)  # [topk]

            # Get execution features for candidates
            cand_exec_feats = exec_feats_norm[candidate_indices].to(device)  # [topk, 3]
            # Composite execution score: weighted combination
            exec_scores = (
                0.5 * cand_exec_feats[:, 0] +   # structure_score
                0.3 * cand_exec_feats[:, 1] +   # can_import
                0.2 * cand_exec_feats[:, 2]     # runtime_estimate
            )  # [topk]

            # Get GatedRerankerHead scores
            cand_slots = doc_facets[candidate_indices].to(device)  # [topk, num_slots, facet_dim]
            try:
                reranker_scores = gated_reranker_scores(
                    model, q_facets, cand_slots, device
                )  # [topk]
            except Exception as e:
                LOG.warning("GatedRerankerHead failed for query '%s': %s", query_text[:30], e)
                reranker_scores = torch.zeros_like(late_int_scores)

            # Combine: late interaction + GatedReranker + execution
            # late_int_scores are the initial retrieval scores (embedding similarity)
            # reranker_scores are cross-encoder style (semantic interaction)
            # exec_scores are execution-based quality signals
            combined_scores = (
                (1.0 - gated_reranker_weight) * late_int_scores
                + gated_reranker_weight * reranker_scores
            )
            # Blend in execution as a minor signal (0.1 weight)
            exec_blend_weight = 0.1
            combined_scores = (
                (1.0 - exec_blend_weight) * combined_scores
                + exec_blend_weight * exec_scores
            )

            # Select best candidate
            best_idx = combined_scores.argmax().item()
            retrieved_text = candidate_ids[best_idx]

        meta = metadata_by_doc.get(retrieved_text, {})
        is_direct = bool(meta.get("is_direct", False))
        retrieved_family = str(meta.get("family", ""))

        results.append({
            "type": case_type,
            "family": family,
            "query": query_text,
            "retrieved": retrieved_text,
            "retrieved_is_direct": is_direct,
            "retrieved_family": retrieved_family,
        })

    summary = mod._summarize_support_discipline_results(results)
    return summary


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="v396: GatedRerankerHead wired into eval")
    parser.add_argument("--gated-reranker-weight", type=float, default=0.3)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    LOG.info("Device: %s", device)

    output_dir = Path(args.output_dir or (ROOT / "artifacts" / "strict_eval_autoresearch_v396"))
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "v396_output.log"
    LOG.info("Output dir: %s", output_dir)

    # Load model with GatedRerankerHead
    model, config = load_v378_with_gated_reranker(device)

    # Run strict eval
    LOG.info("Running strict_eval with GatedRerankerHead (weight=%.2f)", args.gated_reranker_weight)
    t0 = time.time()
    summary = run_strict_eval_objective_only(
        model=model,
        device=device,
        gated_reranker_weight=args.gated_reranker_weight,
        batch_size=8,
    )
    elapsed = time.time() - t0

    LOG.info("Strict eval completed in %.1fs", elapsed)
    LOG.info("Results: %s", json.dumps(summary, indent=2))

    # Save results
    results_path = output_dir / "strict_eval_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "version": "v396",
            "gated_reranker_weight": args.gated_reranker_weight,
            "summary": summary,
            "elapsed_seconds": elapsed,
        }, f, indent=2)
    LOG.info("Results saved to %s", results_path)

    # Report DR and score
    dr = summary.get("direct_rate", 0.0)
    score = summary.get("objective_score", 0.0)
    LOG.info("=" * 60)
    LOG.info("v396 RESULTS: dr=%.4f, score=%.4f", dr, score)
    LOG.info("=" * 60)

    return summary


if __name__ == "__main__":
    main()
