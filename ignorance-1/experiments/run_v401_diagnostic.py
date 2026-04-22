"""
v401: Diagnostic — Candidate 0 from the research memo.

Question: Are the missed supported cases already in retriever top-k,
but merely misordered by the scoring interface?

For each eval case:
  1. Encode query with v378 model (retrieval_facets + retrieval_global)
  2. Search the full vector index (all code snippets) using late-interaction scoring
  3. Report: champion_rank@{5,10,20}, top_wrong_family, top_wrong_is_same_family

This tells us whether the bottleneck is:
  (A) Scoring interface: champion IS in top-k but ranked below near-misses
      → Verifier/reranker is the right bet
  (B) Top-k coverage: champion IS NOT in top-k
      → Pivot to query augmentation + retriever distillation
"""
from __future__ import annotations
import sys as _sys
_project_root = "/mnt/Storage/Projects/catbelly_studio/ignorance-1"
if _project_root not in _sys.path:
    _sys.path.insert(0, _project_root)

import json, os, random, torch, torch.nn.functional as F
from pathlib import Path

from src.models.jepa import JEPAModel
from src.training.phase4 import _proxy_config_v6_overnight
from src.utils.data import BenchmarkTokenizer, make_text_code_pairs, set_seed
from src.utils.retrieval import VectorIndex
from src.losses.alignment import late_interaction_score_matrix

ROOT = Path(_project_root)
DEVICE = "cuda"
V378_CKPT = ROOT / "artifacts/strict_eval_autoresearch_v378/v378-late-inter-high-weight-seed511-seed514/model.pt"


# ── Family labeling ────────────────────────────────────────────────────────────
def _query_to_family(query: str) -> str:
    q = query.lower()
    if "sort" in q and "descending" in q: return "sorting"
    if "trailing newline" in q or "whitespace from each line" in q or "remove trailing" in q: return "strip_lines"
    if "json" in q and "serialize" in q: return "json_parse"
    if "debounce" in q or "delay" in q or "user stops typing" in q: return "debounce"
    if "frequency" in q or "count token" in q: return "frequency"
    if "merge" in q and "dict" in q: return "merge_dicts"
    if "post" in q and "json" in q: return "fetch_json"
    if "javascript" in q and ("start" in q or "prefix" in q or "suffix" in q or "end" in q): return "startswith_js"
    if "anagram" in q: return "anagram"
    if "json" in q and "dump" in q and "serialize" not in q: return "json_dump"
    if "convert" in q and ("camel" in q or "snake" in q): return "camel_to_snake"
    if "file" in q and "extension" in q: return "file_ext"
    if "list" in q and "unique" in q: return "parselist"
    if "extract" in q and "number" in q: return "extract_numbers"
    return "unknown"


# ── Build code corpus ─────────────────────────────────────────────────────────
def build_code_corpus():
    """All code snippets from make_text_code_pairs, with family labels."""
    set_seed(0)
    code_pairs = make_text_code_pairs(repeats=10)
    seen = set()
    corpus = []
    for query, code in code_pairs:
        norm = str(code)
        if norm in seen:
            continue
        seen.add(norm)
        corpus.append({"query": query, "code": code, "family": _query_to_family(query)})
    return corpus


# ── Diagnostic ────────────────────────────────────────────────────────────────
def run_diagnostic():
    print("=" * 70)
    print("v401 DIAGNOSTIC: Top-k coverage for missed supported cases")
    print("=" * 70)

    # Load v378 model with verified correct dimensions
    # embed_dim=192, retrieval_head_dim=256, retrieval_facet_dim=256, facets=30, shared head
    print("\nLoading v378 model...")
    config = _proxy_config_v6_overnight(15_000_000)
    config.embed_dim = 192
    config.retrieval_head_dim = 256
    config.retrieval_head_hidden_dim = 512
    config.retrieval_facet_dim = 256
    config.retrieval_facet_hidden_dim = 512
    config.retrieval_num_facets = 30
    config.retrieval_facet_score_mode = "softmax_maxsim"
    config.retrieval_facet_softmax_temperature = 0.1
    config.use_retrieval_facets = True
    config.use_retrieval_head = True
    config.retrieval_facet_separate_query_code = False

    model = JEPAModel(config).to(DEVICE, dtype=torch.bfloat16)
    ckpt = torch.load(V378_CKPT, map_location="cpu", weights_only=False)
    result = model.load_state_dict(ckpt, strict=False)
    assert len(result.missing_keys) == 0, f"Missing keys: {result.missing_keys[:3]}"
    model.eval()
    print("Model loaded and verified (embed_dim=192, rh_dim=256, rf_dim=256, 30 facets).")

    # Build corpus
    corpus = build_code_corpus()
    print(f"Corpus: {len(corpus)} unique code snippets")

    # Encode corpus
    tokenizer = BenchmarkTokenizer(vocab_size=4096)
    codes = [item["code"] for item in corpus]

    print("Encoding corpus (this may take ~30s)...")
    all_global = []
    all_facets = []
    batch_size = 32
    for i in range(0, len(codes), batch_size):
        batch = codes[i:i+batch_size]
        token_ids = tokenizer.batch_encode(batch, seq_len=256, device=DEVICE)
        with torch.no_grad():
            encoded = model.encode(input_ids=token_ids)
            # z_raw is the raw prediction latent
            z_raw = model.predict(encoded, action_id=1)
            # global embedding for index search
            z_global = model.retrieval_project(z_raw)
            # facets for late-interaction scoring
            facets = model.retrieval_facets(z_raw)
        all_global.append(z_global.cpu())
        all_facets.append(facets.cpu())

    all_global = torch.cat(all_global, dim=0)
    all_facets = torch.cat(all_facets, dim=0)
    print(f"Encoded. global={all_global.shape}, facets={all_facets.shape}")

    # Build vector index with v398 best inference config (maxsim mode)
    index = VectorIndex(
        doc_ids=[item["code"] for item in corpus],
        embeddings=all_global.to(DEVICE),
        facet_embeddings=all_facets.to(DEVICE),
        facet_score_mode="maxsim",
        global_facet_blend=0.35,
        facet_softmax_temperature=0.1,
    )
    print("Index built (maxsim mode).")

    # Get 5-shot eval queries (same seed as strict eval)
    set_seed(42)
    eval_pairs = make_text_code_pairs(repeats=5)
    eval_by_family = {}
    for query, code in eval_pairs:
        family = _query_to_family(query)
        if family not in eval_by_family:
            eval_by_family[family] = []
        eval_by_family[family].append({"query": query, "code": code})

    # ── Run retrieval for each eval case ──────────────────────────────────────
    SUPPORTED_FAMILIES = {"anagram", "json_dump", "sorting"}
    ABSTAINING_FAMILIES = {"strip_lines", "debounce", "frequency", "merge_dicts", "startswith_js"}

    results = []

    print("\n" + "=" * 80)
    print(f"{'Family':<16} {'Query (truncated)':<44} {'@5':>4} {'@10':>5} {'@20':>5} {'TopWrongFam':>14} {'Same?':>5}")
    print("-" * 80)

    for family in sorted(eval_by_family):
        is_supported = family in SUPPORTED_FAMILIES
        is_abstaining = family in ABSTAINING_FAMILIES

        for item in eval_by_family[family]:
            query = item["query"]
            champ_code = item["code"].strip()

            # Encode query
            q_ids = tokenizer.batch_encode([query], seq_len=256, device=DEVICE)
            with torch.no_grad():
                q_enc = model.encode(input_ids=q_ids)
                q_raw = model.predict(q_enc, action_id=1)
                q_global = model.retrieval_project(q_raw)
                q_facets = model.retrieval_facets(q_raw)

            # Search top-20 with v398 best config
            result = index.search_text(
                query_text=query,
                queries=q_global,
                k=20,
                lexical_weight=0.4,
                query_facets=q_facets,
            )

            # Find champion rank
            champ_at = {5: None, 10: None, 20: None}
            champ_code_norm = champ_code
            for rank, doc_id in enumerate(result.ids, 1):
                if doc_id.strip() == champ_code_norm:
                    if champ_at[5] is None and rank <= 5: champ_at[5] = rank
                    if champ_at[10] is None and rank <= 10: champ_at[10] = rank
                    if champ_at[20] is None and rank <= 20: champ_at[20] = rank

            # Top wrong family
            champ_family = family
            top_wrong_family = None
            top_wrong_same = False
            for doc_id in result.ids:
                if doc_id.strip() == champ_code_norm:
                    continue
                top_wrong_family = _query_to_family(doc_id)
                top_wrong_same = (top_wrong_family == champ_family)
                break

            # Format rank display
            def fmt_rank(d, k):
                v = d[k]
                return f"{v}" if v is not None else f">>{k}"

            label = "SUPPORTED" if is_supported else ("ABSTAIN" if is_abstaining else "OTHER")
            q_short = query[:42]
            print(f"{family:<16} {q_short:<44} {fmt_rank(champ_at,5):>4} {fmt_rank(champ_at,10):>5} {fmt_rank(champ_at,20):>5} {(top_wrong_family or '?'):>14} {'Y' if top_wrong_same else 'N':>5}")

            results.append({
                "family": family,
                "query": query,
                "is_supported": is_supported,
                "is_abstaining": is_abstaining,
                "champ_rank_5": champ_at[5],
                "champ_rank_10": champ_at[10],
                "champ_rank_20": champ_at[20],
                "top_wrong_family": top_wrong_family,
                "top_wrong_same_family": top_wrong_same,
                "champion_in_top20": champ_at[20] is not None,
            })

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for label, res_list in [
        ("SUPPORTED families", [r for r in results if r["is_supported"]]),
        ("ABSTAINING families", [r for r in results if r["is_abstaining"]]),
    ]:
        n = len(res_list)
        if n == 0:
            continue
        at5 = sum(1 for r in res_list if r["champ_rank_5"] is not None)
        at10 = sum(1 for r in res_list if r["champ_rank_10"] is not None)
        at20 = sum(1 for r in res_list if r["champ_rank_20"] is not None)
        same_wrong = sum(1 for r in res_list if r["top_wrong_same_family"])

        # Average rank
        avg_ranks = []
        for r in res_list:
            for k in [5, 10, 20]:
                v = r[f"champ_rank_{k}"]
                if v is not None:
                    avg_ranks.append(v)
                    break
        avg = sum(avg_ranks) / len(avg_ranks) if avg_ranks else float("inf")

        print(f"\n  {label} ({n} cases)")
        print(f"    Champion in top-5:  {at5}/{n} ({100*at5/n:.0f}%)")
        print(f"    Champion in top-10: {at10}/{n} ({100*at10/n:.0f}%)")
        print(f"    Champion in top-20: {at20}/{n} ({100*at20/n:.0f}%)")
        print(f"    Top wrong is same-family: {same_wrong}/{n}")
        print(f"    Average best rank: {avg:.1f}")

    # Recommendation
    supported = [r for r in results if r["is_supported"]]
    abstaining = [r for r in results if r["is_abstaining"]]
    sup_in_20 = sum(1 for r in supported if r["champion_in_top20"])
    abst_in_20 = sum(1 for r in abstaining if r["champion_in_top20"])

    print(f"\n{'='*80}")
    print("RECOMMENDATION (based on diagnostic data)")
    print(f"{'='*80}")

    if len(supported) > 0 and sup_in_20 >= len(supported) * 0.8:
        print(f"✅ SUPPORTED: {sup_in_20}/{len(supported)} champions in top-20 → scoring interface bottleneck")
        print(f"   → VERIFIER PATH is justified. Train family-local verifier on top-20 slates.")
    elif len(supported) > 0:
        print(f"⚠️  SUPPORTED: only {sup_in_20}/{len(supported)} champions in top-20 → coverage gap")
        print(f"   → QUERY AUGMENTATION PATH. Need paraphrase expansion or retriever distillation.")

    if len(abstaining) > 0:
        if abst_in_20 >= len(abstaining) * 0.8:
            print(f"✅ ABSTAINING: {abst_in_20}/{len(abstaining)} champions in top-20")
        else:
            print(f"⚠️  ABSTAINING: only {abst_in_20}/{len(abstaining)} champions in top-20")

    # Save
    out_dir = ROOT / "artifacts" / "strict_eval_autoresearch_v401"
    os.makedirs(out_dir, exist_ok=True)
    with open(out_dir / "diagnostic_results.json", "w") as f:
        json.dump({
            "results": results,
            "summary": {
                "supported_in_top20": sup_in_20,
                "supported_total": len(supported),
                "abstaining_in_top20": abst_in_20,
                "abstaining_total": len(abstaining),
            }
        }, f, indent=2)
    print(f"\nSaved to {out_dir / 'diagnostic_results.json'}")


if __name__ == "__main__":
    run_diagnostic()
