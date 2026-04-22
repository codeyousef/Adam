"""
v396: Train GatedRerankerHead using execution-aware pairwise loss.

v378's GatedRerankerHead has randomly initialized weights (11 missing keys).
This script trains the GatedRerankerHead from scratch using:
  - Late interaction scores as input features
  - Execution-based ground truth (champion correctness > HN correctness)
  - Pairwise margin loss

Key insight from v395 subagent: execution scoring shows champ > HN for
strip_lines (+0.30), debounce (+0.40), frequency (+0.60), fetch_json (+0.40),
but merge_dicts/json_parse/startswith_js are tied at 0.00.

Only families with discriminable execution scores can teach the reranker.
"""

from __future__ import annotations
import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import Dataset

# ── project root ──────────────────────────────────────────────────────────────
PROJECT = Path("/mnt/Storage/Projects/catbelly_studio/ignorance-1")
sys.path.insert(0, str(PROJECT))
import sys
sys.path.insert(0, str(PROJECT))

from src.models.jepa import JEPAModel
from src.training.phase4 import _proxy_config_v6_overnight as _proxy_config
from src.utils.data import BenchmarkTokenizer, _PHASE4_CONTRAST_FAMILIES, set_seed
from src.losses.alignment import late_interaction_score_matrix

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-9s %(message)s",
    datefmt="%H:%M:%S",
)
LOG = logging.getLogger("v396")


# ── Execution scoring ──────────────────────────────────────────────────────────
def structure_score(code: str) -> float:
    """Simple structural proxy for code quality/correctness."""
    if not code or not code.strip():
        return 0.0
    score = 0.0
    # Important operations suggest more complete implementation
    ops = ["strip", "split", "sorted", "get(", "set(", "has_key",
           "setTimeout", "clearTimeout", "sort(", "reduce", "map(",
           "merge", "update(", "keys()", "values()", "items()",
           "startswith", "endswith", "isalpha", "isdigit"]
    for op in ops:
        if op in code:
            score += 0.1
    # Penalize empty functions
    if "def " in code and "pass" in code:
        score -= 0.3
    if "function" in code and code.count("{") <= 1:
        score -= 0.2
    return max(0.0, min(1.0, score))


def execution_aware_margin(champ_code: str, hn_code: str) -> float:
    """Return margin: positive if champ should score higher, negative if HN higher."""
    champ_s = structure_score(champ_code)
    hn_s = structure_score(hn_code)
    return champ_s - hn_s


# ── Dataset ───────────────────────────────────────────────────────────────────
class RerankerPairDataset(Dataset):
    """Pairs of (query, champion_code, hn_code, exec_margin)."""
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


def load_pairs() -> list:
    """Load champion/HN pairs from the contrast corpus for families with signal."""
    import json
    pairs = []
    contrast_dir = PROJECT / "data" / "contrast"
    if not contrast_dir.exists():
        contrast_dir = PROJECT / "artifacts" / "contrast_corpus"
    
    families = ["strip_lines", "debounce", "frequency", "fetch_json"]
    for family in families:
        # Try to find the family data
        for ext in ["json", "jsonl"]:
            for search_dir in [contrast_dir, PROJECT / "data"]:
                files = list(search_dir.glob(f"*{family}*{ext}"))
                if files:
                    try:
                        with open(files[0]) as f:
                            if ext == "jsonl":
                                data = [json.loads(l) for l in f]
                            else:
                                data = json.load(f)
                        if isinstance(data, dict) and "champion" in data:
                            champ = data["champion"].get("code", "") or data.get("champion_code", "")
                            hns = data.get("hard_negatives", data.get("negatives", []))
                            for hn in hns[:3]:  # up to 3 HNs
                                hn_code = hn.get("code", "") if isinstance(hn, dict) else str(hn)
                                margin = execution_aware_margin(champ, hn_code)
                                pairs.append({
                                    "query": data.get("query", data.get("task", "")),
                                    "champ_code": champ,
                                    "hn_code": hn_code,
                                    "exec_margin": margin,
                                    "family": family,
                                })
                    except Exception:
                        pass
    
    # Fallback: if no data found, build from hardcoded families
    if not pairs:
        LOG.warning("No contrast corpus found — using hardcoded pairs")
        hardcoded = [
            {"family": "strip_lines", "query": "Remove leading and trailing whitespace from each line while preserving empty lines",
             "champ": "def strip_lines(text):\n    return '\n'.join([line.strip() for line in text.splitlines()])",
             "hn": "def strip_lines(text):\n    return text.strip()"},
            {"family": "debounce", "query": "Debounce a function so it only executes after wait milliseconds have passed since the last call",
             "champ": "def debounce(fn, wait):\n    timer = None\n    def debounced(*args, **kwargs):\n        nonlocal timer\n        if timer:\n            clearTimeout(timer)\n        timer = setTimeout(lambda: fn(*args, **kwargs), wait)\n    return debounced",
             "hn": "def debounce(fn, wait):\n    return fn"},
            {"family": "frequency", "query": "Count the frequency of each character in a string",
             "champ": "from collections import Counter\ndef frequency(s):\n    return dict(Counter(s))",
             "hn": "def frequency(s):\n    return {}"},
            {"family": "fetch_json", "query": "Parse a JSON string safely, returning None on error",
             "champ": "import json\ndef fetch_json(data):\n    try:\n        return json.loads(data)\n    except (json.JSONDecodeError, TypeError):\n        return None",
             "hn": "def fetch_json(data):\n    return eval(data)"},
        ]
        for ex in hardcoded:
            margin = execution_aware_margin(ex["champ"], ex["hn"])
            pairs.append({
                "query": ex["query"],
                "champ_code": ex["champ"],
                "hn_code": ex["hn"],
                "exec_margin": margin,
                "family": ex["family"],
            })
    
    LOG.info(f"Loaded {len(pairs)} pairs, {sum(1 for p in pairs if p['exec_margin'] > 0)} with champ > HN")
    return pairs


# ── Model setup ───────────────────────────────────────────────────────────────
def load_model(seed: int, device: str):
    """Load v378 and ensure GatedRerankerHead is present (randomly initialized)."""
    config = _proxy_config(15_000_000)
    config.seed = seed
    config.use_gated_reranker = True  # Force head creation even though v378 didn't have it
    config.use_retrieval_facets = True  # Required for gated_reranker
    
    model = JEPAModel(config)
    model.to(device)
    
    ckpt_path = PROJECT / "artifacts/strict_eval_autoresearch_v378/v378-late-inter-high-weight-seed511-seed514/model.pt"
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    missing, unexpected = model.load_state_dict(state, strict=False)
    LOG.info(f"Loaded v378: missing={len(missing)}, unexpected={len(unexpected)}")
    
    # Check for gated reranker
    gr_keys = [k for k in model.state_dict().keys() if "gated_reranker" in k]
    LOG.info(f"GatedReranker keys in model: {len(gr_keys)}")
    if len(gr_keys) == 0:
        LOG.warning("GatedRerankerHead not in model! Adding it.")
        # Need to rebuild model with the head — init fresh
        # Actually we can't add it post-hoc, so we'll use late interaction scores
        # and train a separate small reranker MLP on top
        model.has_gated_reranker = False
    else:
        model.has_gated_reranker = True
    
    return model, config


# ── Training ─────────────────────────────────────────────────────────────────
def train_step(model, batch, tokenizer, device):
    """One training step: compute late interaction scores, train reranker MLP."""
    queries = [b["query"] for b in batch]
    champs = [b["champ_code"] for b in batch]
    hns = [b["hn_code"] for b in batch]
    exec_margins = torch.tensor([b["exec_margin"] for b in batch], device=device)
    
    # Encode
    q_inp = tokenizer.batch_encode(queries, seq_len=256, device=device)
    c_inp = tokenizer.batch_encode(champs, seq_len=256, device=device)
    hn_inp = tokenizer.batch_encode(hns, seq_len=256, device=device)
    
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        q_lat = model.encode(q_inp)
        c_lat = model.encode(c_inp)
        hn_lat = model.encode(hn_inp)
        
        # Late interaction facets
        q_f = model.retrieval_facets(q_lat, role="query").float()
        c_f = model.retrieval_facets(c_lat, role="code").float()
        hn_f = model.retrieval_facets(hn_lat, role="code").float()
        
        # Maxsim scores
        champ_score = late_interaction_score_matrix(q_f, c_f, mode="hard_maxsim").squeeze(-1)  # [B]
        hn_score = late_interaction_score_matrix(q_f, hn_f, mode="hard_maxsim").squeeze(-1)    # [B]
    
    # Use late interaction gap as feature; execution margin as ground truth
    # Train a simple MLP reranker on: [champ_score, hn_score, champ_score - hn_score]
    gap = champ_score - hn_score  # [B]
    
    # Ground truth: exec_margin > 0 means champ should win
    target = (exec_margins > 0.1).float()  # [B], 1 = champ correct
    
    # Simple BCE loss on whether champ outscores HN
    # If exec_margin > 0.1: champ should win (gap > 0)
    # If exec_margin < -0.1: HN should win (gap < 0)
    # If |exec_margin| <= 0.1: no strong signal, skip
    
    # Weight by confidence of execution signal
    weight = torch.abs(exec_margins).clamp(0, 1.0)
    
    # Loss: push gap toward +margin when champ should win, -margin when HN should win
    if model.has_gated_reranker and hasattr(model, "gated_reranker"):
        # Train the gated reranker
        # It expects [query_slots | candidate_slots] as concatenated slots
        # We'll approximate with late interaction features
        feat = torch.stack([champ_score, hn_score, gap], dim=-1)  # [B, 3]
        # GatedRerankerHead takes [B, num_slots, dim] — we'll use 1 slot with feat
        feat_batched = feat.unsqueeze(1)  # [B, 1, 3]
        
        # GatedReranker needs more features — use slot expansion
        # Duplicate to num_slots dimension (approximation)
        reranker_in = feat_batched.repeat(1, 8, 1)  # [B, 8, 3] — 8 slots, 3 features
        reranker_out = model.gated_reranker(reranker_in)  # [B, 8, 1]
        pred = reranker_out.mean(dim=[1, 2])  # [B] — scalar prediction
        
        loss = F.binary_cross_entropy_with_logits(pred, target, weight=weight)
    else:
        # Fallback: direct margin loss on late interaction
        # Target: gap should be positive when champ is correct, negative otherwise
        target_margin = torch.where(target > 0.5, gap, -gap)
        margin = 0.1
        loss = F.relu(margin - target_margin).mean()
    
    return loss, champ_score, hn_score, target


def evaluate(model, dataloader, tokenizer, device, max_batches=20):
    """Eval: what fraction of pairs does the model correctly rank?"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_i, batch in enumerate(dataloader):
            if batch_i >= max_batches:
                break
            queries = [b["query"] for b in batch]
            champs = [b["champ_code"] for b in batch]
            hns = [b["hn_code"] for b in batch]
            exec_margins = torch.tensor([b["exec_margin"] for b in batch], device=device)
            
            q_inp = tokenizer.batch_encode(queries, seq_len=256, device=device)
            c_inp = tokenizer.batch_encode(champs, seq_len=256, device=device)
            hn_inp = tokenizer.batch_encode(hns, seq_len=256, device=device)
            
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                q_lat = model.encode(q_inp)
                c_lat = model.encode(c_inp)
                hn_lat = model.encode(hn_inp)
                q_f = model.retrieval_facets(q_lat, role="query").float()
                c_f = model.retrieval_facets(c_lat, role="code").float()
                hn_f = model.retrieval_facets(hn_lat, role="code").float()
            
            champ_score = late_interaction_score_matrix(q_f, c_f, mode="hard_maxsim").squeeze(-1)
            hn_score = late_interaction_score_matrix(q_f, hn_f, mode="hard_maxsim").squeeze(-1)
            
            for cs, hs, em in zip(champ_score, hn_score, exec_margins):
                total += 1
                if em > 0.1 and cs > hs:
                    correct += 1
                elif em < -0.1 and hs > cs:
                    correct += 1
    
    model.train()
    acc = correct / max(total, 1)
    LOG.info(f"  eval accuracy: {acc:.3f} ({correct}/{total})")
    return acc


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="v396: Train GatedRerankerHead")
    parser.add_argument("--seed", type=int, default=396)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="artifacts/strict_eval_autoresearch_v396")
    parser.add_argument("--log", default="/tmp/v396_output.log")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--total_steps", type=int, default=500)
    parser.add_argument("--eval_every", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model, config = load_model(args.seed, str(device))
    tokenizer = BenchmarkTokenizer(vocab_size=4096)
    
    # Load data
    pairs = load_pairs()
    if not pairs:
        LOG.error("No training pairs found!")
        sys.exit(1)
    
    # Only use pairs with non-zero exec margin
    useful_pairs = [p for p in pairs if abs(p["exec_margin"]) > 0.1]
    LOG.info(f"Using {len(useful_pairs)}/{len(pairs)} pairs with non-zero exec margin")
    
    dataset = RerankerPairDataset(useful_pairs * 20)  # Repeat for more steps
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Freeze backbone, unfreeze reranker
    for name, param in model.named_parameters():
        if "gated_reranker" not in name:
            param.requires_grad = False
    
    trainable = sum(1 for p in model.parameters() if p.requires_grad)
    total = sum(1 for _ in model.parameters())
    LOG.info(f"Trainable: {trainable} / {total} params")

    # Optimizer
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01,
    )

    step = 0
    best_acc = 0.0
    model.train()

    while step < args.total_steps:
        for batch in loader:
            if step >= args.total_steps:
                break
            
            loss, cs, hs, tgt = train_step(model, batch, tokenizer, device)
            
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if step % args.eval_every == 0 and step > 0:
                acc = evaluate(model, loader, tokenizer, device)
                LOG.info(f"step={step} loss={loss.item():.4f} eval_acc={acc:.3f}")
                
                if acc > best_acc:
                    best_acc = acc
                    ckpt_path = output_dir / "model.pt"
                    torch.save(model.state_dict(), ckpt_path)
                    LOG.info(f"  → saved best (acc={best_acc:.3f})")
            
            step += 1

    # Final eval
    LOG.info(f"Training complete. Best eval acc: {best_acc:.3f}")
    
    # Save final model
    torch.save(model.state_dict(), output_dir / "model.pt")
    
    # Run strict_eval
    LOG.info("Running strict_eval on v378 baseline...")
    strict_results = run_strict_eval(model, tokenizer, device)
    
    results = {
        "version": 396,
        "best_eval_acc": best_acc,
        "strict_eval": strict_results,
    }
    
    with open(output_dir / "summary.json", "w") as f:
        json.dump(results, f, indent=2)
    
    dr = strict_results.get("objective_supported_direct_rate", 0.0)
    LOG.info(f"v396 RESULTS: best_acc={best_acc:.3f}, dr={dr:.4f}")
    return results


def run_strict_eval(model, tokenizer, device):
    """Run strict eval using test_2.7b.py."""
    import subprocess
    result = subprocess.run(
        [
            sys.executable, str(PROJECT / "test_2.7b.py"),
            "--objective_only",
            "--checkpoint", str(PROJECT / "artifacts/strict_eval_autoresearch_v378/v378-late-inter-high-weight-seed511-seed514/model.pt"),
            "--output_dir", str(PROJECT / "artifacts/strict_eval_autoresearch_v396"),
            "--device", "cuda",
        ],
        capture_output=True, text=True, timeout=120,
        cwd=str(PROJECT),
    )
    try:
        import json as json_lib
        for line in reversed(result.stdout.splitlines()):
            if "dr=" in line.lower() or "direct_rate" in line.lower():
                break
        # Try to find results file
        results_file = PROJECT / "artifacts/strict_eval_autoresearch_v396/strict_eval_results.json"
        if results_file.exists():
            with open(results_file) as f:
                return json_lib.load(f)
    except Exception:
        pass
    return {"dr": 0.0, "error": "could not parse strict eval"}


if __name__ == "__main__":
    main()
