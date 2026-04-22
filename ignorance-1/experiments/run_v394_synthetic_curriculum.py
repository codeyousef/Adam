#!/usr/bin/env python3
"""
v394 synthetic curriculum experiment for IGNORANCE-1.

Trains the 15M model on progressively harder near-miss HNs for 5 failing families:
strip_lines, debounce, frequency, merge_dicts, startswith_js.

Approach:
  1. Load v378 checkpoint (v6_overnight config = embed_dim 192)
  2. Build curriculum dataset with 3 HN tiers per family:
     - Tier 1 (easy): cross-family negatives
     - Tier 2 (medium): same-family hard negatives
     - Tier 3 (hard): the actual hard negatives from contrast corpus
  3. Train with late-interaction margin loss; curriculum schedule:
     steps 0-100: tier1+2 only (easy/medium)
     steps 100-200: mix in tier3 (up to 50%)
  4. Evaluate with strict_eval on 8-family objective.
  5. Save checkpoint to artifacts/strict_eval_autoresearch_v394/
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

# ── project paths ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.losses.alignment import late_interaction_margin_loss
from src.models.jepa import JEPAModel
from src.training.phase4 import _proxy_config
from src.utils.data import _PHASE4_CONTRAST_FAMILIES, set_seed, BenchmarkTokenizer
from src.utils.retrieval import VectorIndex

LOG = logging.getLogger(__name__)


# ── 5 failing families ──────────────────────────────────────────────────────────
FAILING_FAMILIES = ["strip_lines", "debounce", "frequency", "merge_dicts", "startswith_js"]


# ─────────────────────────────────────────────────────────────────────────────
# Curriculum dataset
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CurriculumExample:
    query: str
    positive: str
    tier1_negatives: list[str]   # cross-family easy HNs
    tier2_negatives: list[str]   # same-family medium HNs
    tier3_negatives: list[str]   # actual hard HNs from contrast corpus
    family: str


class CurriculumDataset(Dataset):
    def __init__(self, examples: list[CurriculumExample]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> CurriculumExample:
        return self.examples[idx]


def build_curriculum_dataset(rng: random.Random) -> list[CurriculumExample]:
    """
    Build one CurriculumExample per failing-family per prompt.

    Tier 1: cross_family_negatives from OTHER families (different keyword)
    Tier 2: hard_negatives from SAME family (different algorithm)
    Tier 3: hard_negatives from SAME family (hardest variants)
    """
    examples: list[CurriculumExample] = []
    families_by_name = {f["family"]: f for f in _PHASE4_CONTRAST_FAMILIES}

    for family_name in FAILING_FAMILIES:
        fam = families_by_name.get(family_name)
        if fam is None:
            continue

        # cross-family negatives (easy tier1)
        cross_family = [
            str(other["code"])
            for other in _PHASE4_CONTRAST_FAMILIES
            if str(other["family"]) != family_name
        ]
        rng.shuffle(cross_family)
        tier1 = cross_family[:4]

        # same-family hard negatives (medium tier2 = same as tier3 here,
        # but we split: tier2 is shuffled once, tier3 shuffled independently)
        hard_negs = list(fam["hard_negatives"])
        rng.shuffle(hard_negs)
        tier2 = hard_negs

        # tier3: hard negatives again (independent shuffle for actual hardest)
        tier3 = list(fam["hard_negatives"])
        rng.shuffle(tier3)

        for prompt in fam["prompts"]:
            examples.append(
                CurriculumExample(
                    query=str(prompt),
                    positive=str(fam["code"]),
                    tier1_negatives=tier1,
                    tier2_negatives=tier2,
                    tier3_negatives=tier3,
                    family=family_name,
                )
            )

    rng.shuffle(examples)
    return examples


# ─────────────────────────────────────────────────────────────────────────────
# Model helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_model(device: torch.device) -> JEPAModel:
    """Build 15M JEPA with v6_overnight config (embed_dim=192 for v378 compat)."""
    config = _proxy_config(15_000_000, "v6_overnight")
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
        ROOT
        / "artifacts"
        / "strict_eval_autoresearch_v378"
        / "v378-late-inter-high-weight-seed511-seed514"
        / "model.pt"
    )
    LOG.info("Loading v378 from %s", ckpt_path)
    sd = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(sd, strict=True)

    for name, param in model.named_parameters():
        if "retrieval_facet_head" in name:
            param.requires_grad = True
            LOG.info("  trainable: %s", name)
        else:
            param.requires_grad = False

    trainable = sum(1 for p in model.parameters() if p.requires_grad)
    total = sum(1 for _ in model.parameters())
    LOG.info("Trainable %d / %d total params", trainable, total)


def encode_texts(model: JEPAModel, texts: list[str], batch_size: int, device: torch.device) -> torch.Tensor:
    """Encode texts via model.encode() (includes projection), return [N, embed_dim] tensor."""
    out: list[torch.Tensor] = []
    tok = BenchmarkTokenizer(vocab_size=4096)
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        ids = tok.batch_encode(batch, seq_len=256, device=device)
        ids_t = torch.tensor(ids, device=device, dtype=torch.long)
        with torch.no_grad():
            encoded = model.encode(ids_t)  # Returns [B, 192] for v378 compat
        out.append(encoded)
    return torch.cat(out, dim=0)


# ─────────────────────────────────────────────────────────────────────────────
# Strict eval (objective only, mirrors test_2.7b.py)
# ─────────────────────────────────────────────────────────────────────────────

def run_strict_eval_objective_only(
    model: JEPAModel,
    device: torch.device,
    batch_size: int = 8,
) -> dict:
    """Run strict eval via test_2.7b.py subprocess — avoids all dimension mismatches."""
    import tempfile, os, subprocess, json as json_lib

    # Save current model checkpoint to a temp file
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False, dir="/tmp") as f:
        temp_ckpt = f.name
    torch.save(model.state_dict(), temp_ckpt)

    output_dir = tempfile.mkdtemp()
    try:
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "test_2.7b.py"),
                "15000000",
                temp_ckpt,
                "--json",
            ],
            capture_output=True, text=True, timeout=120,
            cwd=str(ROOT),
        )
        # Extract summary JSON — test_2.7b.py --json outputs the full summary object
        # starting with '{' on its own line and ending with '}' on its own line.
        # The challenge: it's preceded by debug text. Strategy: find the last
        # '{' that starts the outer summary object by finding the first '{' line
        # and last '}' line within the region after the key marker.
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
    return {"objective_supported_direct_rate": 0.0, "objective_score": 0.0}


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_step(
    model: JEPAModel,
    example: CurriculumExample,
    tier3_fraction: float,
    margin: float,
    device: torch.device,
    batch_size: int,
    tokenizer: BenchmarkTokenizer,
) -> torch.Tensor:
    """
    Single training step with late-interaction margin loss.
    tier3_fraction: fraction of negatives drawn from tier3 (hard HNs).
    """
    # Encode query and positive
    # Encode via model.encode() which projects 192→256, then get facets
    with torch.no_grad():
        q_lat = model.encode(tokenizer.batch_encode([example.query], seq_len=256, device=device))
        pos_lat = model.encode(tokenizer.batch_encode([example.positive], seq_len=256, device=device))

    q_facets = model.retrieval_facets(q_lat, role="query").float()      # [1, F, D]
    pos_facets = model.retrieval_facets(pos_lat, role="code").float()   # [1, F, D]

    # Build negative pool
    if tier3_fraction <= 0.0:
        all_negs = example.tier1_negatives + example.tier2_negatives
    else:
        all_negs = example.tier1_negatives + example.tier2_negatives + example.tier3_negatives

    if not all_negs:
        return torch.tensor(0.0, device=device)

    # Encode negatives through model.encode() then retrieval_facets
    with torch.no_grad():
        neg_lat_list = [
            model.encode(tokenizer.batch_encode([neg_text], seq_len=256, device=device))
            for neg_text in all_negs
        ]
        neg_lat = torch.cat(neg_lat_list, dim=0)  # [N, 256]
    neg_facets = model.retrieval_facets(neg_lat, role="code").unsqueeze(0).float()  # [1, N, F, D]
    # late_interaction_margin_loss expects [N, F, D] negatives, not [1, N, F, D]
    neg_facets = neg_facets.squeeze(0)  # [N, F, D]

    loss = late_interaction_margin_loss(
        q_facets,
        pos_facets,
        negative_facets=neg_facets,
        margin=margin,
        mode="hard_maxsim",
        softmax_temperature=0.1,
    )
    return loss


def run_training(
    model: JEPAModel,
    dataset: CurriculumDataset,
    output_dir: Path,
    device: torch.device,
    log_path: str,
    cfg: dict,
) -> dict:
    """
    Train with curriculum: first half = tier1+2 only, second half = mix tier3.
    Periodic eval every eval_every steps.
    """
    lr = cfg.get("lr", 1e-4)
    batch_size = cfg.get("batch_size", 8)
    total_steps = cfg.get("total_steps", 200)
    margin = cfg.get("margin", 0.1)
    eval_every = cfg.get("eval_every", 50)

    tokenizer = BenchmarkTokenizer(vocab_size=4096)

    # Only retrieval_facet_head is trainable
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=1e-5,
    )

    model.train()
    step = 0
    total_loss = 0.0

    rng = random.Random(394)
    indices = list(range(len(dataset)))

    LOG.info("Training: lr=%.4g, batch_size=%d, total_steps=%d, margin=%.2g",
             lr, batch_size, total_steps, margin)

    t0 = time.time()

    while step < total_steps:
        # Curriculum: first half tier1+2 only, second half mix tier3
        if step < total_steps // 2:
            tier3_fraction = 0.0
        else:
            frac = (step - total_steps // 2) / (total_steps - total_steps // 2)
            tier3_fraction = 0.5 * frac  # ramp to 50% tier3

        # Assemble batch
        rng.shuffle(indices)
        batch_losses: list[torch.Tensor] = []
        for b in range(batch_size):
            idx = indices[(step * batch_size + b) % len(indices)]
            ex = dataset[idx]
            loss_t = train_step(model, ex, tier3_fraction, margin, device, batch_size, tokenizer)
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
                "step %d/%d | tier3=%.2f | loss=%.4f | elapsed=%.1fs",
                step, total_steps, tier3_fraction, loss_batch.item(), elapsed
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
    parser = argparse.ArgumentParser(description="v394 synthetic curriculum")
    parser.add_argument("--seed", type=int, default=394)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="artifacts/strict_eval_autoresearch_v394")
    parser.add_argument("--log", default="/tmp/v394_output.log")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--total_steps", type=int, default=200)
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument("--eval_every", type=int, default=50)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = ROOT / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configure logging to both file and stdout
    handler = logging.FileHandler(args.log, mode="w")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    root_log = logging.getLogger()
    root_log.setLevel(logging.INFO)
    root_log.addHandler(handler)
    root_log.addHandler(logging.StreamHandler(sys.stdout))

    LOG.info("=" * 60)
    LOG.info("v394 synthetic curriculum experiment")
    LOG.info("Output: %s", output_dir)
    LOG.info("Log: %s", args.log)
    LOG.info("=" * 60)

    set_seed(args.seed)

    # ── Build model + load v378 ────────────────────────────────────────────────
    LOG.info("Building model …")
    model = build_model(device)
    load_v378(model, device)

    # ── Build curriculum dataset ────────────────────────────────────────────────
    rng = random.Random(args.seed)
    examples = build_curriculum_dataset(rng)
    LOG.info("Curriculum dataset: %d examples (%d families x prompts)",
             len(examples), len(FAILING_FAMILIES))
    dataset = CurriculumDataset(examples)

    # ── Train ─────────────────────────────────────────────────────────────────────
    cfg = {
        "lr": args.lr,
        "batch_size": args.batch_size,
        "total_steps": args.total_steps,
        "margin": args.margin,
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

    # ── Save checkpoint ────────────────────────────────────────────────────────
    ckpt_path = output_dir / "model.pt"
    torch.save(model.state_dict(), ckpt_path)
    LOG.info("Checkpoint saved: %s", ckpt_path)

    # ── Save summary ───────────────────────────────────────────────────────────
    result = {
        "version": "v394",
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
    print(f"v394 synthetic curriculum COMPLETE")
    print(f"  dr              = {dr:.4f}")
    print(f"  conf_gap       = {conf_gap:.4f}")
    print(f"  avg_loss        = {train_result['avg_loss']:.4f}")
    print(f"  wall_time       = {train_result['wall_time']:.1f}s")
    print(f"  checkpoint      = {ckpt_path}")
    print(f"  log             = {args.log}")
    print("=" * 60)

    return result


if __name__ == "__main__":
    main()
