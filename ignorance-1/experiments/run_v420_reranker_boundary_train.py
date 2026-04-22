#!/usr/bin/env python3
"""
v420: Train GatedRerankerHead on boundary discrimination pairs.

CONFIRMED (from code audit):
  - train_production.py NEVER calls model.gated_rerank() during training
  - z_pred = model.predict() = JEPAPredictor output, NOT GatedRerankerHead
  - GatedRerankerHead is ONLY used at eval time via retrieval_facets
  - gated_reranker_weight is DEAD CODE in phase4.py

Strategy: Standalone script that:
  1. Loads v378 checkpoint (proven encoder geometry)
  2. Freezes ALL backbone params
  3. For each step: encodes query+candidate facets, calls model.gated_rerank(),
     computes margin loss on correct vs wrong same-family chunks
  4. ONLY the GatedRerankerHead receives gradient — encoder stays frozen

The GatedRerankerHead takes [B, num_slots, facet_dim] query and candidate slots
and outputs [B, C] scores via slot-level cross-attention. Training it specifically
on same-family boundary pairs teaches it to discriminate supported vs unsupported
chunks without touching the encoder geometry v378 worked hard to establish.
"""

from __future__ import annotations
import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ── project root ──────────────────────────────────────────────────────────────
PROJECT = Path("/mnt/Storage/Projects/catbelly_studio/ignorance-1")
sys.path.insert(0, str(PROJECT))

from src.models.jepa import JEPAModel, JEPAConfig
from src.training.phase4 import _proxy_config_v6_overnight as _proxy_config
from src.utils.data import SimpleTokenizer, set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-9s %(message)s",
    datefmt="%H:%M:%S",
)
LOG = logging.getLogger("v420")


# ── Hard-coded boundary discrimination pairs ──────────────────────────────────
# These are same-family (supported_query, correct_chunk) vs (supported_query, wrong_chunk)
# pairs. The reranker needs to learn that within a family, some chunks correctly
# satisfy the query while others don't.
#
# Ground truth: champ > hn means champ correctly resolves the query.
# exec_margin: positive = champ better, negative = hn better.

BOUNDARY_PAIRS = [
    # strip_lines family
    {
        "family": "strip_lines",
        "query": "Remove leading and trailing whitespace from each line while preserving empty lines.",
        "champ_code": "def strip_lines(text):\n    return '\\n'.join([line.strip() for line in text.splitlines()])",
        "hn_code": "def strip_lines(text):\n    return text.strip()",
        "exec_margin": 0.3,  # champ handles per-line, hn handles whole text
    },
    {
        "family": "strip_lines",
        "query": "Load the file and strip each line before returning it.",
        "champ_code": "# task: Load the file and strip each line before returning it\nwith open(path) as handle:\n    rows = [line.strip() for line in handle]",
        "hn_code": "# task: Load the file and strip each line before returning it\nwith open(path) as handle:\n    rows = handle.read()",
        "exec_margin": 0.4,  # champ parses lines, hn returns raw string
    },
    {
        "family": "strip_lines",
        "query": "Read each line from a text file and strip whitespace.",
        "champ_code": "# task: Read each line from a text file and strip whitespace\nwith open(path) as handle:\n    rows = [line.strip() for line in handle]",
        "hn_code": "# task: Read each line from a text file and strip whitespace\nwith open(path) as handle:\n    rows = handle.read()",
        "exec_margin": 0.4,
    },
    # debounce family
    {
        "family": "debounce",
        "query": "Debounce a function so it only executes after wait milliseconds have passed since the last call.",
        "champ_code": "def debounce(fn, wait):\n    timer = None\n    def debounced(*args, **kwargs):\n        nonlocal timer\n        if timer:\n            clearTimeout(timer)\n        timer = setTimeout(lambda: fn(*args, **kwargs), wait)\n    return debounced",
        "hn_code": "def debounce(fn, wait):\n    return fn",
        "exec_margin": 0.5,  # champ implements debounce logic, hn is identity
    },
    {
        "family": "debounce",
        "query": "Delay a browser handler until the user stops typing.",
        "champ_code": "def delay_handler(handler, wait):\n    timer = None\n    def delayed(*args):\n        nonlocal timer\n        if timer:\n            clearTimeout(timer)\n        timer = setTimeout(lambda: handler(*args), wait)\n    return delayed",
        "hn_code": "def delay_handler(handler, wait):\n    return handler",
        "exec_margin": 0.5,
    },
    # frequency family
    {
        "family": "frequency",
        "query": "Count the frequency of each character in a string.",
        "champ_code": "from collections import Counter\ndef frequency(s):\n    return dict(Counter(s))",
        "hn_code": "def frequency(s):\n    return {}",
        "exec_margin": 0.6,  # champ implements counting, hn is empty
    },
    {
        "family": "frequency",
        "query": "Build a frequency map from a list of tokens.",
        "champ_code": "from collections import Counter\ndef frequency(items):\n    return dict(Counter(items))",
        "hn_code": "def frequency(items):\n    return {}",
        "exec_margin": 0.6,
    },
    # fetch_json family
    {
        "family": "fetch_json",
        "query": "Make a GET request and parse the response body as JSON.",
        "champ_code": "# task: Make a GET request and parse the response body as JSON\nresponse = requests.get(url)\ndata = response.json()",
        "hn_code": "# task: Make a GET request and parse the response body as JSON\nresponse = requests.get(url)\nreturn response.text",
        "exec_margin": 0.4,  # champ parses JSON, hn returns raw text
    },
    {
        "family": "fetch_json",
        "query": "Parse a JSON string safely, returning None on error.",
        "champ_code": "import json\ndef fetch_json(data):\n    try:\n        return json.loads(data)\n    except (json.JSONDecodeError, TypeError):\n        return None",
        "hn_code": "import json\ndef fetch_json(data):\n    return json.loads(data)",
        "exec_margin": 0.3,  # champ handles errors, hn crashes
    },
    # merge_dicts family
    {
        "family": "merge_dicts",
        "query": "Combine two mapping objects into one result.",
        "champ_code": "def merge_dicts(a, b):\n    result = dict(a)\n    result.update(b)\n    return result",
        "hn_code": "def merge_dicts(a, b):\n    return {}",
        "exec_margin": 0.5,
    },
    {
        "family": "merge_dicts",
        "query": "Combine two mapping objects so earlier keys win on collisions.",
        "champ_code": "def merge_dicts(a, b):\n    result = dict(b)\n    result.update(a)\n    return result",
        "hn_code": "def merge_dicts(a, b):\n    result = dict(a)\n    result.update(b)\n    return result",
        "exec_margin": 0.2,  # flip: hn has wrong collision semantics
    },
    # startswith_js family
    {
        "family": "startswith_js",
        "query": "Return whether an input string starts with a given prefix.",
        "champ_code": "function startsWith(text, prefix):\n    return text.startsWith(prefix);",
        "hn_code": "function startsWith(text, prefix):\n    return text.endsWith(prefix);",
        "exec_margin": 0.3,  # hn has wrong direction
    },
    {
        "family": "startswith_js",
        "query": "Return whether a string has the given prefix.",
        "champ_code": "const hasPrefix = text.startsWith(prefix);",
        "hn_code": "const hasPrefix = text.endsWith(prefix);",
        "exec_margin": 0.3,
    },
    # json_parse family — note: parse vs serialize is a real equivalence boundary
    {
        "family": "json_parse",
        "query": "Parse a json string into a javascript object.",
        "champ_code": "# task: Parse a json string into a javascript object\nconst parsed = JSON.parse(payload);",
        "hn_code": "# task: Parse a json string into a javascript object\nconst serialized = JSON.stringify(payload);",
        "exec_margin": 0.2,  # parse vs serialize — wrong direction
    },
    # sorting family
    {
        "family": "sorting",
        "query": "Return a sorted copy of the provided list in ascending order.",
        "champ_code": "def sort_list(items):\n    return sorted(items)",
        "hn_code": "def sort_list(items):\n    return list(reversed(items))",
        "exec_margin": 0.4,  # sort vs reverse
    },
    {
        "family": "sorting",
        "query": "Sort a list of numbers from smallest to largest.",
        "champ_code": "def sort_numbers(nums):\n    return sorted(nums)",
        "hn_code": "def sort_numbers(nums):\n    return nums.sort()",
        "exec_margin": 0.3,  # returns None vs returns sorted list
    },
]


# ── Dataset ───────────────────────────────────────────────────────────────────
class BoundaryPairDataset(Dataset):
    """Each item: (query, champ_code, hn_code, exec_margin, family)"""
    def __init__(self, pairs, repeats=1):
        self.pairs = pairs * repeats

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p = self.pairs[idx % len(self.pairs)]
        return {
            "query": p["query"],
            "champ_code": p["champ_code"],
            "hn_code": p["hn_code"],
            "exec_margin": p["exec_margin"],
            "family": p["family"],
        }


# ── Model setup ───────────────────────────────────────────────────────────────
def load_v378_with_reranker(seed: int, device: str):
    """Load v378 checkpoint, ensure GatedRerankerHead is present."""
    config = _proxy_config(15_000_000)
    config.seed = seed
    # Activate retrieval head first so retrieval_dim = 256 (not embed_dim=192)
    # This is required to match v378's retrieval_facet_head which uses 256-dim input
    config.use_retrieval_head = True
    config.retrieval_head_dim = 256
    config.retrieval_head_hidden_dim = 512
    # Activate facets + reranker (required for GatedRerankerHead)
    config.use_retrieval_facets = True
    config.retrieval_num_facets = 30
    config.retrieval_facet_dim = 256          # MUST match v378 checkpoint (input to input_norm)
    config.retrieval_facet_hidden_dim = 512
    config.use_gated_reranker = True
    config.gated_reranker_hidden_dim = 256
    config.gated_reranker_num_heads = 4

    model = JEPAModel(config).to(device)

    # Load v378 checkpoint — has retrieval facets
    v378_path = PROJECT / "artifacts/strict_eval_autoresearch_v378/v378-late-inter-high-weight-seed511-seed514/model.pt"
    state = torch.load(v378_path, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(state, strict=False)

    gr_keys = [k for k in model.state_dict().keys() if "gated_reranker" in k]
    LOG.info(f"v378 loaded: missing={len(missing)}, unexpected={len(unexpected)}, "
             f"gated_reranker_keys={len(gr_keys)}")

    if len(gr_keys) == 0:
        LOG.error("GatedRerankerHead NOT in model state! Check v378 checkpoint.")
        sys.exit(1)

    return model, config


# ── Training step ─────────────────────────────────────────────────────────────
def train_step(model, batch, tokenizer, device, margin=0.15, temperature=0.07):
    """
    Compute gated reranker scores and margin loss on boundary pairs.

    GatedRerankerHead.forward(query_slots, candidate_slots):
      - query_slots: [B, num_slots, facet_dim]
      - candidate_slots: [B, num_slots, facet_dim]  (one champion + one HN per query)
      - Returns: [B, 2] scores (score for each candidate)

    We want: score[champ] > score[hn] + margin  (when exec_margin > 0)
    """
    queries = [b["query"] for b in batch]
    champs = [b["champ_code"] for b in batch]
    hns = [b["hn_code"] for b in batch]
    exec_margins = torch.tensor([b["exec_margin"] for b in batch], device=device)

    # Tokenize
    q_inp = tokenizer.batch_encode(queries, seq_len=256, device=device)
    c_inp = tokenizer.batch_encode(champs, seq_len=256, device=device)
    hn_inp = tokenizer.batch_encode(hns, seq_len=256, device=device)
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        # Encode all texts
        q_lat = model.encode(q_inp)           # [B, D]
        c_lat = model.encode(c_inp)            # [B, D]
        hn_lat = model.encode(hn_inp)         # [B, D]

        # Retrieval facets: [B, num_slots, facet_dim]
        q_f = model.retrieval_facets(q_lat, role="query").float()
        c_f = model.retrieval_facets(c_lat, role="code").float()
        hn_f = model.retrieval_facets(hn_lat, role="code").float()

        B = q_f.shape[0]

        # Score champ and hn per-query using raw (un-centered) reranker scores.
        # With forward_raw: each query[i] gets scores[i,0]=score(query[i],champ[i])
        # and scores[i,1]=score(query[i],hn[i]). The gap is preserved since there's
        # no centering to cancel it.
        champ_scores_list = []
        hn_scores_list = []
        for i in range(B):
            q_s = q_f[i].unsqueeze(0)           # [1, S, D]
            c_s = c_f[i].unsqueeze(0)           # [1, S, D]
            h_s = hn_f[i].unsqueeze(0)          # [1, S, D]
            c_score = model.gated_rerank_raw(q_s, c_s)   # [1, 1] raw
            h_score = model.gated_rerank_raw(q_s, h_s)   # [1, 1] raw
            champ_scores_list.append(c_score.squeeze(-1))
            hn_scores_list.append(h_score.squeeze(-1))

        champ_scores = torch.stack(champ_scores_list, dim=0)   # [B]
        hn_scores = torch.stack(hn_scores_list, dim=0)         # [B]

    # Margin loss: when exec_margin > 0, champ should score higher
    # Loss = relu(margin - (champ_score - hn_score) * sign(exec_margin))
    # i.e., push champ_score - hn_score in the direction of exec_margin sign
    # with magnitude at least `margin`

    # Binary target: exec_margin > 0.1 -> champ should win (target=1)
    #               exec_margin < -0.1 -> hn should win (target=0)
    #               |exec_margin| <= 0.1 -> skip (weight=0)
    target = (exec_margins > 0.1).float()  # [B], 1 = champ correct
    weight = torch.abs(exec_margins).clamp(0.0, 1.0)  # confidence of signal
    skip_mask = torch.abs(exec_margins) <= 0.1

    if skip_mask.all():
        # All pairs have weak signal — use symmetric margin loss
        gap = champ_scores - hn_scores  # [B]
        loss = F.relu(margin - gap).mean() + F.relu(margin + gap).mean()
        return loss, champ_scores, hn_scores, target, weight

    # Weighted margin loss: push gap in correct direction
    # When target=1: want champ > hn + margin -> loss = relu(margin - gap)
    # When target=0: want hn > champ + margin -> loss = relu(margin + gap)
    gap = champ_scores - hn_scores  # [B]

    # For pairs where target=1: loss = relu(margin - gap)
    # For pairs where target=0: loss = relu(margin + gap)
    target_margin = torch.where(target > 0.5, margin - gap, margin + gap)
    active_weight = weight.clone()
    active_weight[skip_mask] = 0.0

    loss = (F.relu(target_margin) * active_weight).sum() / (active_weight.sum() + 1e-6)

    return loss, champ_scores, hn_scores, target, weight


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate_boundary_acc(model, dataloader, tokenizer, device, max_batches=30):
    """Fraction of pairs where model correctly ranks champ > hn when exec_margin > 0.1."""
    model.eval()
    correct = 0
    total = 0
    champ_wins = 0
    hn_wins = 0

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

            q_lat = model.encode(q_inp)
            c_lat = model.encode(c_inp)
            hn_lat = model.encode(hn_inp)
            q_f = model.retrieval_facets(q_lat, role="query").float()
            c_f = model.retrieval_facets(c_lat, role="code").float()
            hn_f = model.retrieval_facets(hn_lat, role="code").float()

            B = q_f.shape[0]
            champ_scores_list = []
            hn_scores_list = []

            for i in range(B):
                with torch.amp.autocast(device_type="cuda", dtype=torch.float32):
                    q_s = q_f[i].unsqueeze(0)
                    c_score = model.gated_rerank_raw(q_s, c_f[i].unsqueeze(0))
                    h_score = model.gated_rerank_raw(q_s, hn_f[i].unsqueeze(0))
                champ_scores_list.append(c_score.squeeze(-1))
                hn_scores_list.append(h_score.squeeze(-1))

            champ_scores = torch.stack(champ_scores_list, dim=0).squeeze(-1)
            hn_scores = torch.stack(hn_scores_list, dim=0).squeeze(-1)

            for cs, hs, em in zip(champ_scores, hn_scores, exec_margins):
                if abs(em) <= 0.1:
                    continue
                total += 1
                if em > 0.1 and cs > hs:
                    correct += 1
                    champ_wins += 1
                elif em < -0.1 and hs > cs:
                    correct += 1
                    hn_wins += 1
                elif em > 0.1:
                    champ_wins += 1
                elif em < -0.1:
                    hn_wins += 1

    model.train()
    acc = correct / max(total, 1)
    LOG.info(f"  boundary_acc={acc:.3f} ({correct}/{total}), champ_wins={champ_wins}, hn_wins={hn_wins}")
    return acc


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="v420: Train GatedRerankerHead on boundary pairs")
    parser.add_argument("--seed", type=int, default=710)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="artifacts/strict_eval_autoresearch_v420")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--total_steps", type=int, default=800)
    parser.add_argument("--eval_every", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--margin", type=float, default=0.15)
    parser.add_argument("--warmup_steps", type=int, default=40)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    output_dir = Path(args.output) / "v420-reranker-boundary-seed710"
    output_dir.mkdir(parents=True, exist_ok=True)

    LOG.info("Loading v378 with GatedRerankerHead...")
    model, config = load_v378_with_reranker(args.seed, str(device))
    tokenizer = SimpleTokenizer(vocab_size=4096)

    # Freeze entire backbone — only GatedRerankerHead trains
    for name, param in model.named_parameters():
        if "gated_reranker" not in name:
            param.requires_grad = False

    trainable = sum(1 for p in model.parameters() if p.requires_grad)
    total = sum(1 for _ in model.parameters())
    LOG.info(f"Frozen backbone, trainable params: {trainable}/{total}")

    # Dataset: repeat pairs for multiple epochs
    dataset = BoundaryPairDataset(BOUNDARY_PAIRS, repeats=max(1, args.total_steps // len(BOUNDARY_PAIRS)))
    def collate_fn(batch):
        """Default collate stacks dicts into dict of lists — keep as list of dicts."""
        return list(batch)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        drop_last=True, collate_fn=collate_fn)

    # Optimizer with warmup + cosine decay
    reranker_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(reranker_params, lr=args.lr, weight_decay=0.01)

    def lr_lambda(step):
        if step < args.warmup_steps:
            return (step + 1) / args.warmup_steps
        progress = (step - args.warmup_steps) / max(args.total_steps - args.warmup_steps, 1)
        return max(args.min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    step = 0
    best_acc = 0.0
    model.train()

    LOG.info(f"Training GatedRerankerHead for {args.total_steps} steps...")
    LOG.info(f"BOUNDARY_PAIRS: {len(BOUNDARY_PAIRS)} pairs across families: "
             f"{set(p['family'] for p in BOUNDARY_PAIRS)}")

    while step < args.total_steps:
        for batch in loader:
            if step >= args.total_steps:
                break

            loss, cs, hs, tgt, wt = train_step(
                model, batch, tokenizer, device, margin=args.margin
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            if step % 20 == 0:
                iso_score = 0.0  # skip iso for reranker-only training
                LOG.info(
                    f"step={step:4d} loss={loss.item():.4f} "
                    f"lr={optimizer.param_groups[0]['lr']:.2e} "
                    f"gap_mean={(cs - hs).mean().item():+.3f}"
                )

            if step % args.eval_every == 0 and step > 0:
                acc = evaluate_boundary_acc(model, loader, tokenizer, device)
                LOG.info(f"  → step={step} boundary_acc={acc:.3f}")

                if acc > best_acc:
                    best_acc = acc
                    ckpt_path = output_dir / "model.pt"
                    torch.save(model.state_dict(), ckpt_path)
                    LOG.info(f"  → saved best (acc={best_acc:.3f})")

            step += 1

    LOG.info(f"Training complete. Best boundary acc: {best_acc:.3f}")

    # Save final model
    torch.save(model.state_dict(), output_dir / "model.pt")

    # ── Run strict eval ────────────────────────────────────────────────────────
    LOG.info("Running strict_eval on trained model...")
    strict_results = run_strict_eval(output_dir / "model.pt", device)

    score = strict_results.get("score", 0.0)
    strict_status = strict_results.get("strict_status", "UNKNOWN")

    LOG.info(f"v420 RESULTS: best_boundary_acc={best_acc:.3f}, score={score:.2f}, status={strict_status}")
    LOG.info(f"Results saved to {output_dir}")

    with open(output_dir / "summary.json", "w") as f:
        json.dump({
            "version": 420,
            "best_boundary_acc": best_acc,
            "score": score,
            "strict_status": strict_status,
            "strict_results": strict_results,
            "pairs": len(BOUNDARY_PAIRS),
            "families": list(set(p["family"] for p in BOUNDARY_PAIRS)),
        }, f, indent=2)

    return strict_results


def run_strict_eval(model_path: Path, device: str):
    """Run test_2.7b.py on the trained model."""
    import subprocess

    eval_cmd = [
        sys.executable,
        str(PROJECT / "test_2.7b.py"),
        "15000000",
        str(model_path),
        "--json",
        "--confidence-threshold", "0.312",
    ]

    LOG.info(f"Running: {' '.join(eval_cmd)}")
    result = subprocess.run(
        eval_cmd,
        capture_output=True,
        text=True,
        cwd=str(PROJECT),
        timeout=600,
    )

    output = result.stdout + result.stderr

    with open(model_path.parent / "eval_output.log", "w") as f:
        f.write(output)

    # Parse JSON output
    start = output.find("{")
    if start < 0:
        LOG.error(f"No JSON found in eval output:\n{output[-2000:]}")
        return {"score": 0.0, "strict_status": "FAIL", "error": "no JSON"}

    try:
        data = json.loads(output[start:])
    except json.JSONDecodeError as e:
        LOG.error(f"JSON parse error: {e}\n{output[start:start+500]}")
        return {"score": 0.0, "strict_status": "FAIL", "error": str(e)}

    obj = data.get("objective_results", [])
    direct = sum(1 for r in obj if "✅ DIRECT SUPPORT" in r.get("status", ""))
    fp = sum(1 for r in obj if "❌ FALSE POSITIVE" in r.get("status", ""))
    ci = sum(1 for r in obj if "CORRECTLY IGNORANT" in r.get("status", ""))
    abst = sum(1 for r in obj if "❌ ABSTAINED" in r.get("status", ""))
    sf = sum(1 for r in obj if "SAME-FAMILY" in r.get("status", ""))

    from research.strict_eval_search_space import strict_answer_score
    score = strict_answer_score(data)

    LOG.info(f"Strict eval: D={direct} FP={fp} CI={ci} A={abst} SF={sf} | score={score:.2f}")
    LOG.info(f"strict_status: {data.get('strict_status')}")

    return {
        "score": score,
        "strict_status": data.get("strict_status", "UNKNOWN"),
        "direct": direct,
        "fp": fp,
        "ci": ci,
        "abstain": abst,
        "sf": sf,
        "objective_results": obj,
    }


if __name__ == "__main__":
    main()
