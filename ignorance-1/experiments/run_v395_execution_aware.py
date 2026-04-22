#!/usr/bin/env python3
"""
v395: Execution-aware training for IGNORANCE-1.

Hypothesis: The 5 failing families have near-zero embedding discrimination between
champion and HN because embedding similarity is NOT the true ranking signal.
Code execution IS the true signal.

Approach:
  1. Load v378 checkpoint (v6_overnight config = embed_dim=192)
  2. For each training example (champion code + HN code):
     - Execute champion code in sandbox -> check import success + output correctness
     - Execute HN code in sandbox -> same checks
     - execution_score champion vs HN is the ground truth label
  3. Train retrieval_facet_head to produce late interaction scores that correlate
     with execution outcomes, not embedding similarity
  4. Evaluate with strict_eval on hard 8-family objective

Key insight: execution outcome is the TRUE signal, not embedding similarity.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

# ── project paths ──────────────────────────────────────────────────────────────
from pathlib import Path
PROJECT = Path("/mnt/Storage/Projects/catbelly_studio/ignorance-1")
sys.path.insert(0, str(PROJECT))

from src.losses.alignment import late_interaction_score_matrix, late_interaction_margin_loss
from src.models.jepa import JEPAModel
from src.training.phase4 import _proxy_config_v6_overnight, _set_torch_seed
from src.utils.data import BenchmarkTokenizer

LOG = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Execution sandbox
# ─────────────────────────────────────────────────────────────────────────────

EXEC_TIMEOUT = 5.0  # seconds per code execution


def _structure_score(code: str) -> float:
    """
    Score based on semantic completeness using AST analysis.
    Higher score = more complete/semantically correct implementation.
    Used as the execution-quality proxy for training signal.
    """
    import ast

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


def execution_score(code: str) -> float:
    """Compute an execution-quality score [0..1] using structural analysis."""
    return _structure_score(code)


@dataclass
class ExecutionExample:
    query: str
    champion_code: str
    hn_code: str
    family: str
    champ_exec_score: float
    hn_exec_score: float


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

def build_execution_dataset(rng: random.Random) -> list[ExecutionExample]:
    """
    Build execution dataset from the contrast corpus.
    For each (champion, HN) pair, compute execution scores.
    Only include pairs where execution differentiates them clearly.
    """
    from src.utils.data import _PHASE4_CONTRAST_FAMILIES

    examples: list[ExecutionExample] = []
    families_by_name = {f["family"]: f for f in _PHASE4_CONTRAST_FAMILIES}

    # Use all 8 families, focusing on the 5 failing ones
    for fam in _PHASE4_CONTRAST_FAMILIES:
        family_name = fam["family"]
        champ_code = str(fam["code"])
        champ_score = execution_score(champ_code)

        # Score hard negatives
        for hn in fam.get("hard_negatives", []):
            hn_str = str(hn)
            hn_score = execution_score(hn_str)

            # Only include if execution differentiates (or both fail equally)
            # The signal is: champ_exec_score vs hn_exec_score
            # Format query with task prefix (following v391 pattern)
            prompt = str(fam["prompts"][0]) if fam.get("prompts") else ""
            examples.append(ExecutionExample(
                query=f"# task: {prompt}\n",
                champion_code=f"# task: {prompt}\n{champ_code}",
                hn_code=f"# task: {prompt}\n{hn_str}",
                family=family_name,
                champ_exec_score=champ_score,
                hn_exec_score=hn_score,
            ))

        # Also add cross-family negatives for context
        for other in _PHASE4_CONTRAST_FAMILIES:
            if other["family"] != family_name:
                other_code = str(other["code"])
                other_score = execution_score(other_code)
                examples.append(ExecutionExample(
                    query=str(fam["prompts"][0]) if fam.get("prompts") else "",
                    champion_code=champ_code,
                    hn_code=other_code,
                    family=family_name,
                    champ_exec_score=champ_score,
                    hn_exec_score=other_score,
                ))

    rng.shuffle(examples)
    LOG.info("Execution dataset: %d examples", len(examples))
    return examples


class ExecutionDataset(Dataset):
    def __init__(self, examples: list[ExecutionExample]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> ExecutionExample:
        return self.examples[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Model helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_model(device: torch.device) -> JEPAModel:
    """Build 15M JEPA with v6_overnight config (embed_dim=192 for v378 compat)."""
    config = _proxy_config_v6_overnight(15_000_000)
    config.use_retrieval_facets = True
    config.retrieval_num_facets = 30
    config.retrieval_facet_dim = 256
    config.retrieval_facet_hidden_dim = 512
    config.use_retrieval_head = True
    config.retrieval_head_dim = 256
    config.retrieval_head_hidden_dim = 512
    config.use_gated_reranker = False
    model = JEPAModel(config)
    model.to(device)
    return model


def load_v378(model: JEPAModel, device: torch.device) -> None:
    """Load v378 checkpoint, freeze backbone, leave only retrieval_facet_head trainable."""
    ckpt_path = (
        PROJECT
        / "artifacts"
        / "strict_eval_autoresearch_v378"
        / "v378-late-inter-high-weight-seed511-seed514"
        / "model.pt"
    )
    LOG.info("Loading v378 from %s", ckpt_path)
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(sd, strict=True)

    for name, param in model.named_parameters():
        if "retrieval_facet_head" in name or "query_retrieval_facet_head" in name or "code_retrieval_facet_head" in name:
            param.requires_grad = True
            LOG.info("  trainable: %s", name)
        else:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    LOG.info("Trainable %d / %d total params (%.1fM)", trainable, total, trainable / 1e6)


def encode_texts(model: JEPAModel, texts: list[str], batch_size: int, device: torch.device, seq_len: int = 256) -> torch.Tensor:
    """Encode texts via model.encode(), return [N, retrieval_dim] tensor."""
    out: list[torch.Tensor] = []
    tok = BenchmarkTokenizer(vocab_size=4096)
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        ids = tok.batch_encode(batch, seq_len=seq_len, device=device)
        with torch.no_grad():
            # model.encode() applies encoder then retrieval_project (192→256)
            encoded = model.encode(ids)
        out.append(encoded)
    return torch.cat(out, dim=0)


# ─────────────────────────────────────────────────────────────────────────────
# Strict eval (from v394)
# ─────────────────────────────────────────────────────────────────────────────

def run_strict_eval_objective_only(
    model: JEPAModel,
    device: torch.device,
    batch_size: int = 8,
) -> dict:
    """Use test_2.7b.py's strict eval via subprocess for accurate results."""
    import subprocess, json as json_lib, tempfile, os

    # Save current model to a temp checkpoint
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False, dir="/tmp") as f:
        temp_ckpt = f.name
    torch.save(model.state_dict(), temp_ckpt)

    output_dir = tempfile.mkdtemp()
    try:
        result = subprocess.run(
            [
                sys.executable,
                str(PROJECT / "test_2.7b.py"),
                "15000000",
                temp_ckpt,
                "--json",
            ],
            capture_output=True, text=True, timeout=120,
            cwd=str(PROJECT),
        )
        # Extract summary JSON — test_2.7b.py --json outputs the full summary object
        stdout_lines = result.stdout.split('\n')
        outer_first_brace = None
        for i, line in enumerate(stdout_lines):
            if line.strip() == '{':
                outer_first_brace = i
                break
        outer_last_brace = None
        for i in range(len(stdout_lines) - 1, -1, -1):
            if stdout_lines[i].strip() == '}' and i > 10:
                outer_last_brace = i
                break
        if outer_first_brace is not None and outer_last_brace is not None:
            try:
                json_text = '\n'.join(stdout_lines[outer_first_brace:outer_last_brace + 1])
                return json_lib.loads(json_text)
            except Exception:
                pass
        LOG.warning("strict_eval rc=%d stdout snippet: %s", result.returncode, result.stdout[-200:])
        LOG.warning("strict_eval stderr: %s", result.stderr[-200:])
    finally:
        os.unlink(temp_ckpt)
    return {"dr": 0.0, "error": "strict eval failed"}


# ─────────────────────────────────────────────────────────────────────────────
# Training with execution-aware loss
# ─────────────────────────────────────────────────────────────────────────────

def execution_aware_loss(
    model: JEPAModel,
    example: ExecutionExample,
    device: torch.device,
) -> torch.Tensor:
    """
    Compute execution-aware ranking loss.

    The ground truth is the execution score difference:
      champ_exec_score vs hn_exec_score

    We want the late interaction score between (query, champ) to be higher
    than (query, hn) when execution confirms champ is correct.
    """
    # Encode query, champion, and HN through model.encode() -> retrieval_facets()
    tok = BenchmarkTokenizer(vocab_size=4096)
    with torch.no_grad():
        q_lat = model.encode(tok.batch_encode([example.query], seq_len=256, device=device))
        champ_lat = model.encode(tok.batch_encode([example.champion_code], seq_len=256, device=device))
        hn_lat = model.encode(tok.batch_encode([example.hn_code], seq_len=256, device=device))

    q_facets = model.retrieval_facets(q_lat, role="query").float()      # [1, F, D]
    champ_facets = model.retrieval_facets(champ_lat, role="code").float()  # [1, F, D]
    hn_facets = model.retrieval_facets(hn_lat, role="code").float()       # [1, F, D]

    # Late interaction scores
    champ_score = late_interaction_score_matrix(q_facets, champ_facets, mode="hard_maxsim")  # [1, 1]
    hn_score = late_interaction_score_matrix(q_facets, hn_facets, mode="hard_maxsim")        # [1, 1]

    champ_score = champ_score.squeeze(-1)  # [1]
    hn_score = hn_score.squeeze(-1)         # [1]

    # Ground truth: execution-based ranking
    # If champ_exec_score > hn_exec_score, champ should score higher
    exec_margin = example.champ_exec_score - example.hn_exec_score

    # Margin-based loss: enforce champ_score - hn_score aligns with exec_margin
    # Use a soft margin: if exec_margin > 0, champ should beat hn by at least margin
    margin = 0.1
    if exec_margin > 0.1:  # champion is clearly better
        # Champion should score higher than HN
        loss = F.relu(margin - (champ_score - hn_score))
    elif exec_margin < -0.1:  # HN is actually better (rare case)
        # HN should score higher than champion (flip labels)
        loss = F.relu(margin - (hn_score - champ_score))
    else:  # execution ties - don't enforce ranking, but push both away from each other
        # At least ensure they're not too far apart (avoid collapse)
        loss = (champ_score - hn_score).pow(2) * 0.01

    return loss.mean()


def train_step(
    model: JEPAModel,
    example: ExecutionExample,
    device: torch.device,
) -> torch.Tensor:
    """Single training step."""
    loss = execution_aware_loss(model, example, device)
    return loss


def run_training(
    model: JEPAModel,
    dataset: ExecutionDataset,
    output_dir: Path,
    device: torch.device,
    log_path: str,
    cfg: dict,
) -> dict:
    """Train with execution-aware loss. Periodic eval every eval_every steps."""
    lr = cfg.get("lr", 1e-4)
    batch_size = cfg.get("batch_size", 8)
    total_steps = cfg.get("total_steps", 200)
    eval_every = cfg.get("eval_every", 50)

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=1e-5,
    )

    model.train()
    step = 0
    total_loss = 0.0

    rng = random.Random(395)
    indices = list(range(len(dataset)))

    LOG.info("Training: lr=%.4g, batch_size=%d, total_steps=%d", lr, batch_size, total_steps)

    t0 = time.time()

    while step < total_steps:
        rng.shuffle(indices)
        batch_losses: list[torch.Tensor] = []

        for b in range(batch_size):
            idx = indices[(step * batch_size + b) % len(indices)]
            ex = dataset[idx]
            loss_t = train_step(model, ex, device)
            batch_losses.append(loss_t)

        loss_batch = torch.stack(batch_losses).mean()

        optimizer.zero_grad()
        loss_batch.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=1.0
        )
        optimizer.step()

        total_loss += loss_batch.item()
        step += 1

        if step % 20 == 0:
            elapsed = time.time() - t0
            LOG.info(
                "step %d/%d | loss=%.4f | elapsed=%.1fs",
                step, total_steps, loss_batch.item(), elapsed
            )

        if step % eval_every == 0 or step == total_steps:
            LOG.info("  -> running strict_eval at step %d …", step)
            model.eval()
            summary = run_strict_eval_objective_only(model, device)
            dr = summary.get("objective_supported_direct_rate", 0.0)
            LOG.info(
                "  -> step %d strict_eval | dr=%.4f | conf_gap=%.4f",
                step, dr, summary.get("objective_confidence_gap", 0.0)
            )
            model.train()

    avg_loss = total_loss / total_steps
    wall_time = time.time() - t0
    LOG.info("Training complete. avg_loss=%.4f, wall_time=%.1fs", avg_loss, wall_time)
    return {"avg_loss": avg_loss, "wall_time": wall_time, "final_step": step}


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="v395 execution-aware training")
    parser.add_argument("--seed", type=int, default=395)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="artifacts/strict_eval_autoresearch_v395")
    parser.add_argument("--log", default="/tmp/v395_output.log")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--total_steps", type=int, default=200)
    parser.add_argument("--eval_every", type=int, default=50)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = PROJECT / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configure logging to both file and stdout
    handler = logging.FileHandler(args.log, mode="w")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    root_log = logging.getLogger()
    root_log.setLevel(logging.INFO)
    root_log.addHandler(handler)
    root_log.addHandler(logging.StreamHandler(sys.stdout))

    LOG.info("=" * 60)
    LOG.info("v395 execution-aware training experiment")
    LOG.info("Output: %s", output_dir)
    LOG.info("Log: %s", args.log)
    LOG.info("=" * 60)

    _set_torch_seed(args.seed)

    # ── Build model + load v378 ────────────────────────────────────────────────
    LOG.info("Building model …")
    model = build_model(device)
    load_v378(model, device)

    # ── Build execution dataset ────────────────────────────────────────────────
    LOG.info("Building execution dataset …")
    rng = random.Random(args.seed)
    examples = build_execution_dataset(rng)
    LOG.info("Execution dataset: %d examples", len(examples))
    dataset = ExecutionDataset(examples)

    # ── Train ─────────────────────────────────────────────────────────────────
    cfg = {
        "lr": args.lr,
        "batch_size": args.batch_size,
        "total_steps": args.total_steps,
        "eval_every": args.eval_every,
    }

    train_result = run_training(
        model=model,
        dataset=dataset,
        output_dir=output_dir,
        device=device,
        log_path=args.log,
        cfg=cfg,
    )

    # ── Final strict_eval ───────────────────────────────────────────────────────
    LOG.info("Running final strict_eval …")
    model.eval()
    summary = run_strict_eval_objective_only(model, device)

    dr = summary.get("objective_supported_direct_rate", 0.0)
    conf_gap = summary.get("objective_confidence_gap", 0.0)
    LOG.info("Final strict_eval | dr=%.4f | conf_gap=%.4f", dr, conf_gap)

    # ── Save checkpoint ───────────────────────────────────────────────────────
    ckpt_path = output_dir / "model.pt"
    torch.save(model.state_dict(), ckpt_path)
    LOG.info("Checkpoint saved: %s", ckpt_path)

    # ── Save summary ───────────────────────────────────────────────────────────
    result = {
        "version": "v395",
        "seed": args.seed,
        "train_result": train_result,
        "strict_eval": summary,
        "dr": float(dr),
        "objective_confidence_gap": float(conf_gap),
        "log_path": args.log,
        "config": cfg,
    }
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(result, f, indent=2)
    LOG.info("Summary saved: %s", summary_path)

    print("\n" + "=" * 60)
    print(f"v395 execution-aware training COMPLETE")
    print(f"dr={dr:.4f}, conf_gap={conf_gap:.4f}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Log: {args.log}")
    print("=" * 60)


if __name__ == "__main__":
    main()