"""
v402: Candidate A — Family-Local Late-Interaction Verifier

The diagnostic found:
  - merge_dicts: champion at rank 1 in 3/3 cases, but model abstains
  - frequency: champion in 4/6 top-20, avg rank 1.2
  - json_parse: champion in 1/4 top-20, avg rank 2.0

These cases have the champion in the top-k, but the scoring interface
doesn't distinguish champion from same-family near-miss. A token-level
verifier (late interaction over query vs candidate) can do what the
single-vector scoring cannot.

Architecture:
  - First stage (frozen v378): z_query, retrieve top-20 with maxsim
  - Second stage (NEW verifier): late_inter_score(query_facets, candidate_facets)
  - Late interaction operates at TOKEN LEVEL: each query facet attends to
    each candidate facet for fine-grained matching

Training data:
  - Family-local training: for each family, collect top-20 slates from v378
  - Triplets: (query, champion_code, same_family_negative) from same family
  - The verifier learns to score champion > same_family_wrong within the slate
"""
from __future__ import annotations
import sys as _sys
_project_root = "/mnt/Storage/Projects/catbelly_studio/ignorance-1"
if _project_root not in _sys.path:
    _sys.path.insert(0, _project_root)

import json, os, random, time, tempfile, subprocess, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader, Dataset

from src.models.jepa import JEPAModel
from src.training.phase4 import _proxy_config_v6_overnight
from src.utils.data import BenchmarkTokenizer, make_text_code_pairs, set_seed
from src.utils.retrieval import VectorIndex

ROOT = Path(_project_root)
DEVICE = "cuda"
V378_CKPT = ROOT / "artifacts/strict_eval_autoresearch_v378/v378-late-inter-high-weight-seed511-seed514/model.pt"


# ── Family-local slate dataset ────────────────────────────────────────────────
class FamilyLocalSlateDataset(Dataset):
    """
    For each family, collect the top-20 retrieved candidates from v378 for each query.
    Then build triplets: (query, champion, same_family_negative).
    We train the verifier to score champion > negative using late interaction.
    """

    def __init__(self, model, tokenizer, corpus, index, seed=42):
        self.model = model
        self.tokenizer = tokenizer
        self.corpus = corpus
        self.index = index
        self.rng = random.Random(seed)
        self.slates = self._build_slates()
        self.triplets = self._build_triplets()

    def _build_slates(self):
        """Retrieve top-20 slates for each family query using v378."""
        slates = {}
        by_family = {}
        for item in self.corpus:
            f = item["family"]
            if f not in by_family:
                by_family[f] = []
            by_family[f].append(item)

        self.model.eval()
        with torch.no_grad():
            for family, items in by_family.items():
                for item in items:
                    query = item["query"]
                    champ_code = item["code"].strip()

                    q_ids = self.tokenizer.batch_encode([query], seq_len=256, device=DEVICE)
                    q_enc = self.model.encode(input_ids=q_ids)
                    q_raw = self.model.predict(q_enc, action_id=1)
                    q_global = self.model.retrieval_project(q_raw)
                    q_facets = self.model.retrieval_facets(q_raw)

                    result = self.index.search_text(
                        query_text=query,
                        queries=q_global,
                        k=20,
                        lexical_weight=0.4,
                        query_facets=q_facets,
                    )

                    champ_rank = None
                    for rank, doc_id in enumerate(result.ids, 1):
                        if doc_id.strip() == champ_code:
                            champ_rank = rank
                            break

                    slates[query] = {
                        "family": family,
                        "champion": champ_code,
                        "top20": list(result.ids),
                        "champ_rank": champ_rank,
                    }
                    print(f"  {family}: '{query[:50]}' champ_rank={champ_rank}")
        return slates

    def _build_triplets(self):
        """Build (query, champion, negative) triplets from slates grouped by family."""
        by_family = {}
        for sl in self.slates.values():
            f = sl["family"]
            if f == "unknown":
                continue
            if f not in by_family:
                by_family[f] = []
            by_family[f].append(sl)

        triplets = []
        for family, items in by_family.items():
            for slate in items:
                champ = slate["champion"]
                # negative: same family, not champion
                negatives = [
                    doc for doc in slate["top20"]
                    if doc.strip() != champ and self._doc_family(doc) == family
                ]
                for neg in negatives[:3]:  # up to 3 negatives per positive
                    triplets.append((slate["query"], champ, neg))
                # Also: champion vs cross-family wrong (harder negative)
                cross_family = [
                    doc for doc in slate["top20"]
                    if doc.strip() != champ and self._doc_family(doc) != family
                ]
                if cross_family:
                    triplets.append((slate["query"], champ, self.rng.choice(cross_family)))

        self.rng.shuffle(triplets)
        print(f"  Built {len(triplets)} triplets from {len(by_family)} families")
        return triplets

    def _doc_family(self, doc: str) -> str:
        for item in self.corpus:
            if item["code"].strip() == doc.strip():
                return item["family"]
        return "unknown"

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        return self.triplets[idx % len(self.triplets)]


class VerifierModel(nn.Module):
    """
    Late-interaction verifier: query_facets (F, D) attend to candidate_facets (F, D)
    and produce a scalar score. Uses ColBERT-style late interaction.

    Unlike the first-stage retriever which scores (query_vec, doc_vec) as a single dot,
    this verifier attends query facets over candidate facets for fine-grained matching.
    """
    def __init__(self, facet_dim=256, hidden_dim=128):
        super().__init__()
        # Project query facets
        self.query_proj = nn.Linear(facet_dim, hidden_dim)
        # Project candidate facets
        self.cand_proj = nn.Linear(facet_dim, hidden_dim)
        # Attention: query facets attend to candidate facets
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        # Final scoring
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, query_facets, cand_facets):
        """
        query_facets: (B, F, D)
        cand_facets: (B, F, D)
        Returns: (B,) scores
        """
        q = self.query_proj(query_facets)           # (B, F, H)
        c = self.cand_proj(cand_facets)            # (B, F, H)
        # Late interaction: query attends to candidate
        attn_out, _ = self.attn(q, c, c)          # (B, F, H)
        # Max over facets + mean pool
        max_out = attn_out.max(dim=1).values       # (B, H)
        mean_out = attn_out.mean(dim=1)            # (B, H)
        out = F.relu(self.fc1(max_out + mean_out)) # (B, H)
        score = self.fc2(out).squeeze(-1)          # (B,)
        return score


def encode_code_for_verifier(model, tokenizer, codes, seq_len=256):
    """Encode code snippets, return facets for late-interaction scoring."""
    tokens = tokenizer.batch_encode(codes, seq_len=seq_len, device=DEVICE)
    with torch.no_grad():
        enc = model.encode(input_ids=tokens)
        raw = model.predict(enc, action_id=1)
        facets = model.retrieval_facets(raw)   # (B, num_facets, facet_dim)
    return facets


def paired_verifier_loss(query_facets, pos_facets, neg_facets, verifier, margin=0.2):
    """Verifier loss: score(pos) - score(neg) > margin."""
    pos_score = verifier(query_facets, pos_facets)
    neg_score = verifier(query_facets, neg_facets)
    return F.relu(margin - (pos_score - neg_score)).mean()


def train_verifier(seed=514, steps=500, lr=2e-4, margin=0.2, cc_weight=0.5, hidden_dim=128):
    """Train family-local verifier on top-20 slates."""
    print(f"\n{'='*60}")
    print(f"Training v402 verifier: seed={seed}, steps={steps}, lr={lr}")
    print(f"{'='*60}")

    torch.manual_seed(seed)
    random.seed(seed)
    set_seed(seed)

    # Load v378
    print("Loading v378 model...")
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

    # Build corpus + index
    tokenizer = BenchmarkTokenizer(vocab_size=4096)
    code_pairs = make_text_code_pairs(repeats=10)
    seen = set()
    corpus = []
    for q, code in code_pairs:
        norm = str(code)
        if norm in seen:
            continue
        seen.add(norm)
        corpus.append({"query": q, "code": code, "family": _query_to_family(q)})

    print(f"Encoding corpus ({len(corpus)} snippets)...")
    codes = [item["code"] for item in corpus]
    all_global = []
    all_facets = []
    for i in range(0, len(codes), 32):
        batch = codes[i:i+32]
        tokens = tokenizer.batch_encode(batch, seq_len=256, device=DEVICE)
        with torch.no_grad():
            enc = model.encode(input_ids=tokens)
            raw = model.predict(enc, action_id=1)
            global_emb = model.retrieval_project(raw)
            facets = model.retrieval_facets(raw)
        all_global.append(global_emb.cpu())
        all_facets.append(facets.cpu())

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

    # Build training dataset
    print("Building training slates (this takes ~60s)...")
    dataset = FamilyLocalSlateDataset(model, tokenizer, corpus, index, seed=seed)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, drop_last=True)
    print(f"Dataset: {len(dataset)} triplets")

    # Verifier
    verifier = VerifierModel(facet_dim=256, hidden_dim=hidden_dim).to(DEVICE)
    optimizer = torch.optim.AdamW(verifier.parameters(), lr=lr, weight_decay=0.01)

    # Freeze v378, train verifier only
    model.requires_grad_(False)

    t0 = time.time()
    step = 0
    losses = []

    verifier.train()
    while step < steps:
        for query_texts, pos_codes, neg_codes in dataloader:
            if step >= steps:
                break

            # Encode codes
            pos_facets = encode_code_for_verifier(model, tokenizer, list(pos_codes))
            neg_facets = encode_code_for_verifier(model, tokenizer, list(neg_codes))

            # Encode queries
            q_tokens = tokenizer.batch_encode(list(query_texts), seq_len=256, device=DEVICE)
            with torch.no_grad():
                q_enc = model.encode(input_ids=q_tokens)
                q_raw = model.predict(q_enc, action_id=1)
                q_facets = model.retrieval_facets(q_raw)

            li_loss = paired_verifier_loss(q_facets, pos_facets, neg_facets, verifier, margin=margin)
            total_loss = li_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(verifier.parameters(), 1.0)
            optimizer.step()

            losses.append({"step": step, "total": total_loss.item(), "li": li_loss.item()})
            step += 1

            if step % 50 == 0:
                elapsed = time.time() - t0
                recent = losses[-50:]
                print(f"  step {step}/{steps} | loss={sum(l['total'] for l in recent)/50:.4f} | {elapsed:.0f}s")

    print(f"  Done. {steps} steps in {time.time()-t0:.1f}s")
    return model, verifier, tokenizer


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


def evaluate_with_verifier(model, verifier, tokenizer, name):
    """Run strict eval with verifier re-ranking of top-20 slates."""
    out_dir = ROOT / "artifacts" / f"strict_eval_autoresearch_v402"
    os.makedirs(out_dir, exist_ok=True)

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False, dir="/tmp") as f:
        tmp = f.name
    try:
        # Save v378 checkpoint
        torch.save(torch.load(V378_CKPT, map_location="cpu", weights_only=False), tmp)

        # Save verifier
        verifier_path = out_dir / f"{name}_verifier.pt"
        torch.save(verifier.state_dict(), verifier_path)

        cmd = [
            sys.executable, str(ROOT / "test_2.7b.py"), "15000000", tmp, "--json",
            # v398 best inference config
            "--retrieval-facet-score-mode", "maxsim",
            "--confidence-threshold", "0.4",
            "--lexical-weight", "0.4",
            "--rerank-consensus-temperature", "0.05",
            "--rerank-agreement-weight", "0.3",
            "--selective-gate-similarity-floor", "0.6",
            # Full config
            "--rerank-topk", "5", "--rerank-shortlist-mode", "pred_query_union_local",
            "--rerank-query-weight", "0.3", "--rerank-lexical-weight", "0.0",
            "--rerank-support-weight", "0.24", "--rerank-consensus-weight", "0.35",
            "--rerank-consensus-floor", "0.9158", "--rerank-consensus-margin-gate", "0.0092",
            "--rerank-pairwise-mode", "supportspec_citecheck_floor_borda",
            "--rerank-support-floor-margin-gate", "0.014", "--rerank-spec-weight", "0.18",
            "--rerank-answerspec-mode", "code_pref", "--rerank-answerspec-margin-gate", "0.034",
            "--rerank-safe-expand-topk", "6", "--rerank-safe-expand-margin", "0.004",
            "--rerank-parafence-weight", "1.0", "--rerank-parafence-variants", "3",
            "--selective-gate-mode", "margin_mean_gap",
            "--selective-gate-margin-threshold", "0.01", "--selective-gate-mean-gap-threshold", "0.016",
            "--rerank-verifier-uplift-weight", "0.4", "--rerank-verifier-gap-scale", "1.0",
            "--rerank-verifier-support-weight", "1.0", "--rerank-verifier-spec-weight", "0.0",
            "--retrieval-facet-softmax-temperature", "0.1", "--retrieval-global-facet-blend", "0.35",
            "--confidence-mode", "support_feature_calibrator",
            "--confidence-support-topk", "5", "--confidence-support-temperature", "0.1",
            # VERIFIER FLAG (will be picked up by patched test_2.7b.py)
            "--verifier-path", str(verifier_path),
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=str(ROOT))
        summary = _parse_json(r.stdout)

        sys.path.insert(0, str(ROOT))
        from research.strict_eval_search_space import strict_answer_score
        score = strict_answer_score(summary)
        dr = summary.get("objective_supported_direct_rate", 0)
        cg = summary.get("objective_confidence_gap", 0)
        results_list = summary.get("objective_results", [])
        direct = sum(1 for r in results_list if "DIRECT" in r.get("status", ""))
        abstained = sum(1 for r in results_list if "ABSTAINED" in r.get("status", ""))
        fp = sum(1 for r in results_list if "FALSE POSITIVE" in r.get("status", ""))

        print(f"\n  === {name} ===")
        print(f"  score={score:.2f} dr={dr:.3f} conf_gap={cg:.4f}")
        print(f"  direct={direct} abstained={abstained} fp={fp}")

        with open(out_dir / f"{name}_summary.json", "w") as f:
            json.dump({**summary, "answer_score": score}, f, indent=2)

        return {"name": name, "score": score, "dr": dr, "conf_gap": cg,
                "direct": direct, "abstained": abstained, "fp": fp}
    finally:
        try:
            os.unlink(tmp)
        except Exception:
            pass


def _parse_json(stdout):
    for idx in range(len(stdout)):
        if stdout[idx] != "{": continue
        for end in range(idx+20, min(idx+100000, len(stdout)+1)):
            try:
                d = json.loads(stdout[idx:end])
                if isinstance(d, dict) and len(d) > 5:
                    return d
            except:
                pass
    return {}


def run_batch():
    results = []

    # Candidate A1: standard verifier
    model1, v1, tok1 = train_verifier(seed=514, steps=500, lr=2e-4, margin=0.2, cc_weight=0.5, hidden_dim=128)
    r1 = evaluate_with_verifier(model1, v1, tok1, "v402-A1-std")
    results.append(r1)

    # Candidate A2: higher margin
    model2, v2, tok2 = train_verifier(seed=515, steps=500, lr=2e-4, margin=0.5, cc_weight=0.3, hidden_dim=128)
    r2 = evaluate_with_verifier(model2, v2, tok2, "v402-A2-highmargin")
    results.append(r2)

    # Candidate A3: deeper verifier
    model3, v3, tok3 = train_verifier(seed=516, steps=500, lr=1e-4, margin=0.2, cc_weight=0.8, hidden_dim=256)
    r3 = evaluate_with_verifier(model3, v3, tok3, "v402-A3-deep")
    results.append(r3)

    print(f"\n{'='*70}")
    print("v402 VERIFIER RESULTS — ranked by score")
    print(f"{'='*70}")
    results.sort(key=lambda x: x["score"], reverse=True)
    print(f"{'Candidate':<35} {'Score':>7} {'dr':>6} {'Direct':>7} {'Abst':>6} {'FP':>4}")
    print(f"{'-'*35} {'-'*7} {'-'*6} {'-'*7} {'-'*6} {'-'*4}")
    for r in results:
        print(f"{r['name']:<35} {r['score']:>7.2f} {r['dr']:>6.3f} {r['direct']:>7} {r['abstained']:>6} {r['fp']:>4}")
    print(f"\nBaseline v378: score=41.11, dr=3/8, direct=3, abstained=5, fp=1")
    print(f"v398 best inference: score=41.64")

    with open(ROOT / "artifacts" / "strict_eval_autoresearch_v402" / "batch_summary.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    import sys
    run_batch()
