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


def momentum_queue_contrastive_loss(
    z_query: torch.Tensor,
    z_positive: torch.Tensor,
    negative_queue: torch.Tensor | None = None,
    temperature: float = 0.07,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Contrastive loss using a queue of negatives (momentum queue)."""
    q = F.normalize(z_query.float(), dim=-1)
    k = F.normalize(z_positive.float(), dim=-1)
    if negative_queue is not None and negative_queue.numel() > 0:
        neg = F.normalize(negative_queue.float(), dim=-1)
        candidate = torch.cat([k, neg], dim=0)
    else:
        candidate = k
    logits = q @ candidate.T / max(temperature, 1e-4)
    labels = torch.zeros(logits.shape[0], device=logits.device, dtype=torch.long)
    loss = F.cross_entropy(logits, labels)
    metrics = {"momentum_queue_loss": float(loss.detach().cpu().item())}
    return loss.to(z_query.dtype), metrics


def prototype_alignment_loss(
    z_query: torch.Tensor,
    prototypes: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Prototype alignment loss: align queries to their class prototypes."""
    q = F.normalize(z_query.float(), dim=-1)
    p = F.normalize(prototypes.float(), dim=-1)
    logits = q @ p.T / max(temperature, 1e-4)
    loss = F.cross_entropy(logits, labels)
    metrics = {"prototype_loss": float(loss.detach().cpu().item())}
    return loss.to(z_query.dtype), metrics


def pairwise_similarity_penalty(z: torch.Tensor, max_samples: int = 256) -> torch.Tensor:
    """Penalize high pairwise similarity within a batch to encourage spread."""
    if z.shape[0] < 2:
        return z.new_tensor(0.0)
    indices = torch.randperm(z.shape[0], device=z.device)[:min(max_samples, z.shape[0])]
    z_sub = F.normalize(z[indices].float(), dim=-1)
    sim = z_sub @ z_sub.T
    sim = sim - torch.eye(sim.shape[0], device=sim.device) * 1e4
    if sim.numel() == 0:
        return z.new_tensor(0.0)
    return F.relu(sim.max()).to(z.dtype)


def retrieval_margin_loss(
    z_query: torch.Tensor,
    z_positive: torch.Tensor,
    negative_pool: torch.Tensor | None = None,
    margin: float = 0.2,
    mode: str = "hard_maxsim",
) -> torch.Tensor:
    """Margin-based retrieval loss: push positive similarity above negatives by margin."""
    q = F.normalize(z_query.float(), dim=-1)
    p = F.normalize(z_positive.float(), dim=-1)
    pos_sim = (q * p).sum(dim=-1)
    if negative_pool is None or negative_pool.numel() == 0:
        return -pos_sim.mean()
    negs = F.normalize(negative_pool.float(), dim=-1)
    if mode == "hard_maxsim" and negs.ndim == 2:
        neg_sim = (q.unsqueeze(1) * negs).amax(dim=1).squeeze(1)
    else:
        neg_sim = (q.unsqueeze(1) * negs).mean(dim=1).squeeze(1)
    loss = F.relu(neg_sim - pos_sim + margin)
    return loss.mean().to(z_query.dtype)


def retrieval_vicreg_loss(
    z_query: torch.Tensor,
    z_positive: torch.Tensor,
    query_queue: torch.Tensor | None = None,
    positive_queue: torch.Tensor | None = None,
    invariance_weight: float = 1.0,
    variance_weight: float = 1.0,
    covariance_weight: float = 0.05,
    variance_target: float = 0.75,
) -> tuple[torch.Tensor, dict[str, float]]:
    """VICReg-style loss for retrieval: invariance + variance + covariance terms.

    z_query and z_positive are the two representations to align.
    query_queue and positive_queue are optional queues for contrastive negatives.
    """
    # Invariance: align query to positive
    q = F.normalize(z_query.float(), dim=-1)
    p = F.normalize(z_positive.float(), dim=-1)

    if positive_queue is not None and positive_queue.numel() > 0:
        pos_q = F.normalize(positive_queue.float(), dim=-1)
        neg_q = F.normalize(query_queue.float(), dim=-1) if query_queue is not None and query_queue.numel() > 0 else None
        if neg_q is not None:
            logits = q @ torch.cat([pos_q, neg_q], dim=0).T / 0.07
        else:
            logits = (q * pos_q).sum(dim=-1, keepdim=True) / 0.07
        labels = torch.zeros(z_query.shape[0], device=z_query.device, dtype=torch.long)
        invariance = F.cross_entropy(logits, labels)
    else:
        invariance = (1 - (q * p).sum(dim=-1)).mean()

    # Variance: encourage std(z) ~ sqrt(batch_size)
    def variance_loss(z):
        if z.shape[0] < 2:
            return z.new_tensor(0.0)
        z_std = z.float().std(dim=0)
        return F.relu(variance_target - z_std).mean()

    var_loss = variance_loss(z_query) + variance_loss(z_positive)

    # Covariance: decorrelate dimensions
    def covariance_loss(z):
        if z.shape[0] < 2 or z.shape[1] < 2:
            return z.new_tensor(0.0)
        z_centered = z.float() - z.float().mean(dim=0, keepdim=True)
        cov = (z_centered.T @ z_centered) / max(z.shape[0] - 1, 1)
        d = cov.shape[0]
        off_diag = cov.flatten()[1:].reshape(d, d - 1).T.flatten()
        return (off_diag ** 2).mean()

    cov_loss = covariance_loss(z_query) + covariance_loss(z_positive)

    loss = invariance_weight * invariance + variance_weight * var_loss + covariance_weight * cov_loss
    metrics = {
        "vicreg_invariance": float(invariance.detach().cpu().item()),
        "vicreg_variance": float(var_loss.detach().cpu().item()),
        "vicreg_covariance": float(cov_loss.detach().cpu().item()),
    }
    return loss.to(z_query.dtype), metrics