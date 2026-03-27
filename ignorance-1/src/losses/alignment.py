from __future__ import annotations

import torch
import torch.nn.functional as F


def contrastive_alignment_loss(
    source: torch.Tensor,
    target: torch.Tensor,
    negative_pool: torch.Tensor | None = None,
    temperature: float = 0.07,
    symmetric: bool = True,
) -> torch.Tensor:
    if source.ndim != 2 or target.ndim != 2:
        raise ValueError(f"Expected [B, D] tensors, got {tuple(source.shape)} and {tuple(target.shape)}")
    if source.shape != target.shape:
        raise ValueError(f"Expected matching shapes, got {tuple(source.shape)} and {tuple(target.shape)}")

    source_norm = F.normalize(source.float(), dim=-1)
    target_norm = F.normalize(target.float(), dim=-1)
    if negative_pool is not None and negative_pool.numel() > 0:
        negative_norm = F.normalize(negative_pool.float(), dim=-1)
        candidate_norm = torch.cat([target_norm, negative_norm], dim=0)
    else:
        candidate_norm = target_norm

    logits = source_norm @ candidate_norm.T
    logits = logits / max(temperature, 1e-4)
    labels = torch.arange(logits.shape[0], device=logits.device)
    loss = F.cross_entropy(logits, labels)
    if symmetric:
        reverse_logits = (target_norm @ source_norm.T) / max(temperature, 1e-4)
        loss = 0.5 * (loss + F.cross_entropy(reverse_logits, labels))
    return loss.to(source.dtype)


def normalized_mse_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_norm = F.normalize(prediction, dim=-1)
    target_norm = F.normalize(target, dim=-1)
    return F.mse_loss(pred_norm, target_norm)


def ignorance_penalty(
    query: torch.Tensor,
    candidate_pool: torch.Tensor,
    target_max_similarity: float = 0.1,
) -> torch.Tensor:
    if query.ndim != 2 or candidate_pool.ndim != 2:
        raise ValueError(f"Expected [B, D] tensors, got {tuple(query.shape)} and {tuple(candidate_pool.shape)}")
    if candidate_pool.shape[0] == 0:
        return query.new_tensor(0.0)

    query_norm = F.normalize(query.float(), dim=-1)
    candidate_norm = F.normalize(candidate_pool.float(), dim=-1)
    max_similarity = (query_norm @ candidate_norm.T).amax(dim=-1)
    penalty = F.relu(max_similarity - target_max_similarity)
    return penalty.mean().to(query.dtype)


def paired_alignment_loss(
    z_text: torch.Tensor,
    z_code: torch.Tensor,
    z_pred: torch.Tensor,
    negative_pool: torch.Tensor | None = None,
    temperature: float = 0.07,
    prediction_weight: float = 1.0,
    embedding_weight: float = 0.5,
    mse_weight: float = 0.25,
) -> tuple[torch.Tensor, dict[str, float]]:
    code_target = z_code.detach()
    pred_contrastive = contrastive_alignment_loss(
        z_pred,
        code_target,
        negative_pool=negative_pool,
        temperature=temperature,
    )
    embed_contrastive = contrastive_alignment_loss(
        z_text,
        code_target,
        negative_pool=negative_pool,
        temperature=temperature,
    )
    mse = normalized_mse_loss(z_pred, code_target)
    loss = prediction_weight * pred_contrastive + embedding_weight * embed_contrastive + mse_weight * mse
    metrics = {
        "pred_contrastive": float(pred_contrastive.detach().cpu().item()),
        "embed_contrastive": float(embed_contrastive.detach().cpu().item()),
        "mse": float(mse.detach().cpu().item()),
    }
    return loss, metrics