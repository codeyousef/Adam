#!/usr/bin/env python3
"""
v418 Launch Script — NCA Loss (k-NN Margin Optimization)

AIM: Replace InfoNCE (contrastive alignment) with Neighborhood Components Analysis (NCA)
loss. NCA directly optimizes the k-NN classification margin in embedding space —
better suited for the fine-grained boundary problem than contrastive discrimination.

Key insight: InfoNCE pulls positive pairs together and pushes all negatives away equally.
NCA learns which dimensions matter for the k-NN decision boundary. For debounce vs
throttle, the key dimensions are timing-semantics tokens — NCA can focus on those
rather than averaging gradients across all negatives.

Safe approach: Only modifies alignment.py (adds nca_loss function) and
phase4.py (adds config param and loss call). Does NOT touch data.py.

v378 baseline: score=41.64 (D=3, A=5, FP=1, SF=0)
"""
import sys, os, subprocess, shutil, yaml
from pathlib import Path

ROOT = Path("/mnt/Storage/Projects/catbelly_studio/ignorance-1")
PY = str(ROOT / "../.venv/bin/python")
RUN_DIR = ROOT / "artifacts/strict_eval_autoresearch_v4/v418-nca-loss-seed706"
RUN_DIR.mkdir(parents=True, exist_ok=True)

V378_CKPT = ROOT / "artifacts/strict_eval_autoresearch_v378/v378-late-inter-high-weight-seed511-seed514/model.pt"
V338_CKPT = ROOT / "artifacts/strict_eval_autoresearch_v338/v338-promoted-earlier-onset-tiny-mixed-bridge-seed504/model.pt"

# === STEP 1: Add NCA loss to alignment.py ===
alignment_py = ROOT / "src/losses/alignment.py"
alignment_content = alignment_py.read_text()

NCA_FUNC = '''
def nca_loss(
    anchors: torch.Tensor,
    positives: torch.Tensor,
    negative_pool: torch.Tensor | None = None,
    *,
    temperature: float = 0.07,
    num_negative_samples: int = 31,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Neighborhood Components Analysis (NCA) loss.

    NCA learns a Mahalanobis-style metric by maximizing the softness of the
    k-NN classification boundary, directly optimizing classification accuracy
    rather than contrastive separation.

    Unlike InfoNCE which pushes all negatives equally, NCA focuses gradient
    on the negatives that are closest to the decision boundary.

    Args:
        anchors: [B, D] query embeddings
        positives: [B, D] positive code embeddings
        negative_pool: [N, D] negative code embeddings for sampling
        temperature: softmax temperature for computing neighbor similarities
        num_negative_samples: number of negatives to sample per anchor
        reduction: "mean" or "none"

    Returns:
        Scalar loss
    """
    B = anchors.shape[0]
    device = anchors.device

    anchors_norm = F.normalize(anchors.float(), dim=-1)
    positives_norm = F.normalize(positives.float(), dim=-1)

    # Positive similarities (diagonal)
    pos_sim = (anchors_norm * positives_norm).sum(dim=-1)  # [B]

    # Sample negatives from the pool
    if negative_pool is not None and negative_pool.numel() > 0:
        neg_norm = F.normalize(negative_pool.float(), dim=-1)
        N = neg_norm.shape[0]
        if N >= num_negative_samples:
            indices = torch.randperm(N, device=device)[:num_negative_samples]
        else:
            indices = torch.arange(N, device=device)
        neg_samples = neg_norm[indices]  # [K, D]
        neg_sim = anchors_norm @ neg_samples.T  # [B, K]
    else:
        neg_sim = anchors_norm.new_empty(B, 0)

    # Also include other positives in the batch as implicit negatives
    other_pos_sim = anchors_norm @ positives_norm.T  # [B, B]
    mask = ~torch.eye(B, device=device, dtype=torch.bool)
    other_pos_sim = other_pos_sim.masked_fill(~mask, float("-inf"))

    # Concatenate negatives
    all_neg_sim = torch.cat([neg_sim, other_pos_sim], dim=-1)

    # Softmax: p_pos = exp(pos_sim) / (exp(pos_sim) + sum(exp(neg_sim)))
    pos_exp = torch.exp(pos_sim / temperature)
    neg_exp = torch.exp(all_neg_sim / temperature).sum(dim=-1)
    denom = pos_exp + neg_exp.clamp(min=1e-8)
    p_pos = pos_exp / denom

    loss = -torch.log(p_pos.clamp(min=1e-8))

    if reduction == "mean":
        return loss.mean().to(anchors.dtype)
    return loss.to(anchors.dtype)

'''

# Append to alignment.py
alignment_content = alignment_content.rstrip() + NCA_FUNC
alignment_py.write_text(alignment_content)
print(f"Added nca_loss() to {alignment_py}")

# === STEP 2: Add NCA to phase4.py imports and config ===
phase4_py = ROOT / "src/training/phase4.py"
phase4_content = phase4_py.read_text()

# Add nca_loss to imports if not present
if 'nca_loss' not in phase4_content:
    phase4_content = phase4_content.replace(
        'from src.losses.alignment import (',
        'from src.losses.alignment import (\n    nca_loss,'
    )

# Add nca params after answerability_distilled params
nca_params = '''
    # === Research9 v418: NCA Loss for k-NN margin optimization ===
    nca_loss_weight = float(getattr(config, "nca_loss_weight", 0.0))
    nca_temperature = float(getattr(config, "nca_temperature", 0.07))
    nca_num_negative_samples = int(getattr(config, "nca_num_negative_samples", 31))
'''
phase4_content = phase4_content.replace(
    '    phase4_dataset = str(getattr(config, "phase4_dataset", "benchmark_v1"))',
    nca_params + '\n    phase4_dataset = str(getattr(config, "phase4_dataset", "benchmark_v1"))'
)

# Add NCA to total_loss after answerability_distilled loss block
nca_loss_call = '''
            # research9 v418: NCA loss for fine-grained k-NN boundary
            if nca_loss_weight > 0.0 and negative_pool.numel() > 0:
                nca_loss_val = nca_loss(
                    anchors=z_query.detach(),
                    positives=z_code.detach(),
                    negative_pool=negative_pool,
                    temperature=nca_temperature,
                    num_negative_samples=nca_num_negative_samples,
                    reduction="mean",
                )
                total_loss = total_loss + nca_loss_weight * nca_loss_val
'''
phase4_content = phase4_content.replace(
    '                    total_loss = total_loss + answerability_distilled_weight * ans_loss',
    '                    total_loss = total_loss + answerability_distilled_weight * ans_loss' + nca_loss_call
)

phase4_py.write_text(phase4_content)
print(f"Patched {phase4_py}")

# === STEP 3: Build v418 config ===
config = yaml.safe_load((V378_CKPT.parent / "config.yaml").read_text())
phase4 = config.setdefault("phase4", {})

config["seed"] = 706
config["profile"] = "strict-eval-autoresearch-v4-v418-nca-loss"
config["warm_start_phase3_only"] = False
config["warm_start_model_path"] = str(V378_CKPT)
config["base_model_path"] = str(V338_CKPT)

phase4["seed"] = 706
phase4["steps"] = 300
phase4["phase4_steps"] = 300
phase4["classifier_weight"] = 0.09
phase4["clf_weight"] = 0.09
phase4["query_multiview_weight"] = 1.0
phase4["warm_start_phase3_only"] = False
phase4["warm_start_model_path"] = str(V378_CKPT)
phase4["production_mode"] = False
phase4["production_steps"] = 0
phase4["production_phase4_repeats"] = 0
phase4["phase4_dataset"] = "behavioral_constraints_v2_taxonomy_support_discipline_v1"

# === v418 key change ===
phase4["nca_loss_weight"] = 1.0
phase4["nca_temperature"] = 0.07
phase4["nca_num_negative_samples"] = 31

config_path = RUN_DIR / "config.yaml"
config_path.write_text(yaml.safe_dump(config, sort_keys=False))
print(f"Config saved to {config_path}")

# === STEP 4: Train v418 ===
model_path = RUN_DIR / "model.pt"
tmp_ckpt = str(model_path) + ".tmp"
shutil.copy2(V378_CKPT, tmp_ckpt)

train_cmd = [
    PY, str(ROOT / "train_production.py"),
    "--config", str(config_path),
    "--size", str(int((config.get("sizes") or config.get("phase4", {}).get("sizes", [15_000_000]))[0])),
    "--output", str(model_path),
    "--device", str(config.get("device", "cuda")),
]

print("\nTraining command:", " ".join(train_cmd))
result = subprocess.run(train_cmd, cwd=ROOT, timeout=36000)
print(f"\nReturn code: {result.returncode}")
if result.returncode != 0:
    print("STDERR:", result.stderr[-3000:] if result.stderr else "(none)")
    sys.exit(result.returncode)

print(f"\nModel saved to {model_path}")
print("Run strict eval:")
print(f"  python test_2.7b.py 15000000 {model_path} 2>&1 | grep -E 'Objective|D=|A=|FP=|SF=|score='")
