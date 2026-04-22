"""
v403: Candidate C — Query-Side Ensemble for Abstaining Families

Diagnostic found:
  - debounce: champion 0/5 in top-20, top wrong is same-family debounce code
  - startswith_js: champion 0/4 in top-20, top wrong is same-family startswith code
  - strip_lines: champion 0/1 in top-20

The champion chunk is in the corpus but NOT retrieved by the original query.
This means the query encoding doesn't activate the right retrieval space.

Candidate C: Query-side ensemble via paraphrase variants.

The idea: encode multiple paraphrase variants of the query, ensemble their embeddings
(by averaging), and use the ensemble to retrieve. If the champion is in any
individual variant's top-k, the ensemble embedding should surface it.

Implementation:
  1. Generate query variants using the model's own generation (beam search from query)
     OR use heuristic paraphrases (swap synonyms, reorder clauses)
  2. Encode each variant, get z_query for each
  3. Average all z_queries → ensemble query embedding
  4. Retrieve with ensemble embedding using maxsim
  5. Check: does champion now appear in top-20?

If champion is found for debounce/startswith_js/strip_lines with ensemble,
the fix is: during eval, use ensemble query embedding instead of single z_query.

Also test: does the 2nd-stage verifier (v402) help for cases where
champion IS in top-20 but misranked (merge_dicts, frequency, json_parse)?
"""
from __future__ import annotations
import sys as _sys
_project_root = "/mnt/Storage/Projects/catbelly_studio/ignorance-1"
if _project_root not in _sys.path:
    _sys.path.insert(0, _project_root)

import json, os, random, time, torch, torch.nn.functional as F
from pathlib import Path

from src.models.jepa import JEPAModel
from src.training.phase4 import _proxy_config_v6_overnight
from src.utils.data import BenchmarkTokenizer, make_text_code_pairs, set_seed
from src.utils.retrieval import VectorIndex

ROOT = Path(_project_root)
DEVICE = "cuda"
V378_CKPT = ROOT / "artifacts/strict_eval_autoresearch_v378/v378-late-inter-high-weight-seed511-seed514/model.pt"


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


# ── Query paraphrasing ─────────────────────────────────────────────────────────
def paraphrase_query(query: str) -> list[str]:
    """Generate heuristic paraphrases of a query to improve retrieval coverage."""
    q = query.lower()

    variants = [query]  # Always include original

    # Synonym swaps
    replacements = [
        ("delay", "wait"),
        ("until", "before"),
        ("user stops", "user has stopped"),
        ("typing", "keystrokes"),
        ("starts with", "begins with"),
        ("prefix", "beginning"),
        ("check whether", "return whether"),
        ("remove trailing", "strip trailing"),
        ("trim trailing", "strip trailing"),
        ("line by line", "each line"),
        ("count how many", "frequency of"),
        ("token", "element"),
        ("merge", "combine"),
        ("mapping", "dictionary"),
        ("javascript", "js"),
        ("function", "method"),
    ]
    for old, new in replacements:
        if old in q:
            variants.append(query.replace(old, new))

    # Clause reordering: "Delay X until Y" → "X should wait until Y"
    if q.startswith("delay"):
        parts = query.split(" until ", 1)
        if len(parts) == 2:
            variants.append(f"{parts[0]} that waits until {parts[1]}")
            variants.append(f"Make sure to delay until {parts[1]}")

    if "starts with" in q:
        parts = query.split(" ", 1)
        if len(parts) > 1:
            variants.append(f"Return true if string {parts[1]}")

    # Add what-it-does description
    if "debounce" in q:
        variants.extend([
            "Throttle an event handler with a delay timer",
            "Implement debounce to prevent rapid event firing",
        ])
    if "startswith" in q or "prefix" in q:
        variants.extend([
            "Check if string begins with a given substring in JavaScript",
            "JavaScript method to test string prefix",
        ])
    if "strip" in q:
        variants.extend([
            "Remove leading and trailing whitespace from each line",
            "Clean whitespace from file lines",
        ])
    if "frequency" in q or "count token" in q:
        variants.extend([
            "Count occurrences of each item in a list",
            "Build a tally map from a collection",
        ])
    if "merge" in q and "dict" in q:
        variants.extend([
            "Combine two Python dictionaries",
            "Update one dict with another",
        ])

    return list(dict.fromkeys(variants))  # deduplicate, preserve order


# ── Main diagnostic + fix test ─────────────────────────────────────────────────
def run_ensemble_diagnostic():
    print("=" * 70)
    print("v403: Query Ensemble Diagnostic — top-k coverage with paraphrases")
    print("=" * 70)

    # Load v378
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
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    print("Model loaded.")

    # Build corpus
    set_seed(0)
    code_pairs = make_text_code_pairs(repeats=10)
    seen = set()
    corpus = []
    for q, code in code_pairs:
        norm = str(code)
        if norm in seen:
            continue
        seen.add(norm)
        corpus.append({"query": q, "code": code, "family": _query_to_family(q)})

    tokenizer = BenchmarkTokenizer(vocab_size=4096)
    codes = [item["code"] for item in corpus]

    print(f"Encoding corpus ({len(corpus)} snippets)...")
    all_global = []
    all_facets = []
    for i in range(0, len(codes), 32):
        batch = codes[i:i+32]
        tokens = tokenizer.batch_encode(batch, seq_len=256, device=DEVICE)
        with torch.no_grad():
            enc = model.encode(input_ids=tokens)
            raw = model.predict(enc, action_id=1)
            g = model.retrieval_project(raw)
            f = model.retrieval_facets(raw)
        all_global.append(g.cpu())
        all_facets.append(f.cpu())

    all_global = torch.cat(all_global, dim=0)
    all_facets = torch.cat(all_facets, dim=0)

    index = VectorIndex(
        doc_ids=[item["code"] for item in corpus],
        embeddings=all_global.to(DEVICE),
        facet_embeddings=all_facets.to(DEVICE),
        facet_score_mode="maxsim",
        global_facet_blend=0.35,
        facet_softmax_temperature=0.1,
    )
    print("Index built.")

    # Eval queries (5-shot)
    set_seed(42)
    eval_pairs = make_text_code_pairs(repeats=5)
    eval_by_family = {}
    for q, code in eval_pairs:
        f = _query_to_family(q)
        if f not in eval_by_family:
            eval_by_family[f] = []
        eval_by_family[f].append({"query": q, "code": code})

    SUPPORTED = {"anagram", "json_dump", "sorting"}
    ABSTAINING = {"debounce", "startswith_js", "strip_lines", "frequency", "merge_dicts", "json_parse"}

    print("\n" + "=" * 80)
    print(f"{'Family':<16} {'Method':<20} {'Champ@5':>7} {'Champ@10':>8} {'Champ@20':>8} {'Same?':>6}")
    print("-" * 80)

    ensemble_improved = []
    original_worse = []

    for family in sorted(eval_by_family):
        if family == "unknown":
            continue
        is_abstaining = family in ABSTAINING

        for item in eval_by_family[family]:
            query = item["query"]
            champ_code = item["code"].strip()

            # ── Original query encoding ──────────────────────────────────────────
            q_ids = tokenizer.batch_encode([query], seq_len=256, device=DEVICE)
            with torch.no_grad():
                q_enc = model.encode(input_ids=q_ids)
                q_raw = model.predict(q_enc, action_id=1)
                q_global = model.retrieval_project(q_raw)
                q_facets = model.retrieval_facets(q_raw)

            result_orig = index.search_text(query, q_global, k=20, lexical_weight=0.4, query_facets=q_facets)

            # ── Ensemble query encoding ──────────────────────────────────────────
            variants = paraphrase_query(query)
            variant_globals = []
            variant_facets = []
            for var_q in variants:
                v_ids = tokenizer.batch_encode([var_q], seq_len=256, device=DEVICE)
                with torch.no_grad():
                    v_enc = model.encode(input_ids=v_ids)
                    v_raw = model.predict(v_enc, action_id=1)
                    variant_globals.append(model.retrieval_project(v_raw).cpu())
                    variant_facets.append(model.retrieval_facets(v_raw).cpu())

            # Average ensemble
            ens_global = torch.stack(variant_globals).mean(dim=0).to(DEVICE)
            ens_facets = torch.stack(variant_facets).mean(dim=0).to(DEVICE)

            result_ens = index.search_text(query, ens_global, k=20, lexical_weight=0.4, query_facets=ens_facets)

            # Find champion ranks
            def champ_rank(result):
                for rank, doc_id in enumerate(result.ids, 1):
                    if doc_id.strip() == champ_code:
                        return rank
                return None

            r_orig = champ_rank(result_orig)
            r_ens = champ_rank(result_ens)

            # Top wrong family
            champ_family = family
            tw_orig = None
            for doc_id in result_orig.ids:
                if doc_id.strip() != champ_code:
                    tw_orig = _query_to_family(doc_id)
                    break

            # Format
            def fmt(r):
                return f"{r}" if r is not None else f">>20"

            orig_label = f"original"
            print(f"{family:<16} {orig_label:<20} {fmt(r_orig):>7} {fmt(r_orig):>8} {fmt(r_orig):>8} {'Y' if tw_orig == champ_family else 'N':>6}")

            ens_label = f"ensemble({len(variants)})"
            print(f"{family:<16} {ens_label:<20} {fmt(r_ens):>7} {fmt(r_ens):>8} {fmt(r_ens):>8} {'Y' if tw_orig == champ_family else 'N':>6}")

            if is_abstaining and r_orig is None and r_ens is not None:
                ensemble_improved.append({
                    "family": family,
                    "query": query,
                    "original_rank": r_orig,
                    "ensemble_rank": r_ens,
                    "variants": variants,
                })
            elif is_abstaining and r_orig is None:
                original_worse.append({
                    "family": family,
                    "query": query,
                    "original_rank": r_orig,
                    "ensemble_rank": r_ens,
                    "variants": variants,
                })

            print()

    # Summary
    print("=" * 80)
    print("ENSEMBLE IMPROVEMENT FOR ABSTAINING FAMILIES")
    print("=" * 80)
    improved = [x for x in ensemble_improved if x["ensemble_rank"] is not None]
    not_found = [x for x in ensemble_improved if x["ensemble_rank"] is None] + original_worse

    print(f"\nChampions found by ensemble (were missing from original top-20):")
    for x in improved:
        print(f"  {x['family']}: '{x['query'][:50]}' | was >>20, now rank {x['ensemble_rank']}")

    print(f"\nChampions still missing after ensemble ({len(not_found)} cases):")
    for x in not_found:
        print(f"  {x['family']}: '{x['query'][:50]}' | variants tried: {len(x['variants'])}")

    if improved:
        print(f"\n✅ Ensemble query augmentation HELPS for {len(improved)} cases!")
        print(f"   → Candidate C is justified. Implement ensemble query encoding in eval.")
    else:
        print(f"\n⚠️  Ensemble does NOT help — champions still missing.")
        print(f"   → The problem is not query encoding; champion may not be in index or needs different retrieval strategy.")

    # Save
    out_dir = ROOT / "artifacts" / "strict_eval_autoresearch_v403"
    os.makedirs(out_dir, exist_ok=True)
    with open(out_dir / "ensemble_results.json", "w") as f:
        json.dump({
            "improved": improved,
            "not_found": not_found,
        }, f, indent=2)

    return improved, not_found


if __name__ == "__main__":
    improved, not_found = run_ensemble_diagnostic()
