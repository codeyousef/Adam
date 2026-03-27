from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def sigreg_loss(
    z: torch.Tensor,
    m: int = 1024,
    num_knots: int = 32,
    lambda_reg: float = 0.1,
) -> torch.Tensor:
    if z.ndim != 3:
        raise ValueError(f"Expected z with shape [B, T, D], got {tuple(z.shape)}")

    _, _, dim = z.shape
    z_flat = z.reshape(-1, dim)
    if z_flat.shape[0] < 2:
        return z_flat.new_tensor(0.0)

    directions = torch.randn(m, dim, device=z.device, dtype=torch.float32)
    directions = F.normalize(directions, dim=-1)
    
    # Cast to float32 for stable CF calculation
    z_flat = z_flat.float()
    projections = z_flat @ directions.T

    t_vals = torch.linspace(0.2, 4.0, num_knots, device=z.device, dtype=torch.float32)
    weights = torch.exp(-(t_vals**2) / (2.0 * (lambda_reg**2)))
    target_cf = torch.exp(-(t_vals**2) / 2.0)

    projected = projections.T.unsqueeze(1)
    char_fn = torch.exp(1j * t_vals.view(1, -1, 1) * projected).mean(dim=-1)
    diff = torch.abs(char_fn - target_cf.view(1, -1)) ** 2
    integrand = weights.view(1, -1) * diff
    loss = torch.trapezoid(integrand, t_vals, dim=-1).mean()
    return loss.real.to(z.dtype)


def isotropic_score(z: torch.Tensor) -> float:
    if z.ndim == 3:
        z = z.reshape(-1, z.shape[-1])
    z = z.float()
    z = z - z.mean(dim=0, keepdim=True)
    cov = torch.cov(z.T)
    diag = torch.diag(cov)
    denom = diag.mean().clamp_min(1e-6)
    off_diag = cov - torch.diag(diag)
    score = 1.0 - (off_diag.abs().mean() / denom)
    return float(score.clamp(0.0, 1.0).item())


def gaussian_projection_p_value(z: torch.Tensor, num_projections: int = 32) -> float:
    if z.ndim == 3:
        z = z.reshape(-1, z.shape[-1])
    z = z.float()
    dim = z.shape[-1]
    directions = torch.randn(num_projections, dim, device=z.device)
    directions = F.normalize(directions, dim=-1)
    projections = (z @ directions.T).cpu()

    try:
        from scipy import stats
    except Exception:
        return 0.0

    p_values: list[float] = []
    for idx in range(projections.shape[1]):
        sample = projections[:, idx].numpy()
        std = max(sample.std(), 1e-6)
        standardized = (sample - sample.mean()) / std
        _, p_value = stats.kstest(standardized, "norm")
        p_values.append(float(p_value))
    return float(sum(p_values) / max(len(p_values), 1))


def collapse_detected(z: torch.Tensor, threshold: float = 0.95) -> bool:
    if z.ndim == 3:
        z = z.reshape(-1, z.shape[-1])
    if z.shape[0] < 4:
        return False
    z = F.normalize(z.float(), dim=-1)
    sim = z @ z.T
    mask = ~torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)
    high_sim = (sim[mask] > threshold).float().mean()
    return bool(high_sim.item() > 0.5)


def epps_pulley_proxy(z: torch.Tensor, num_projections: int = 128) -> float:
    return float(sigreg_loss(z, m=num_projections).detach().cpu().item())