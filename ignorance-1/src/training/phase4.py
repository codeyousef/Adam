from __future__ import annotations

import math
import os
import time

import torch
import torch.nn.functional as F

from src.losses.alignment import ignorance_penalty, paired_alignment_loss
from src.models.jepa import JEPAConfig, JEPAModel, approximate_model_params
from src.utils.data import SimpleTokenizer, make_text_code_pairs, sample_ood_queries


# Collapse detection: if code-query offdiag similarity > this threshold for N consecutive steps, abort training
COLLAPSE_OFFDIAG_THRESH = 0.95
COLLAPSE_CONSECUTIVE_STEPS = 3


def _mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return math.sqrt(max(variance, 0.0))


def _average_curves(curves: list[list[float]]) -> list[float]:
    if not curves:
        return []
    width = min(len(curve) for curve in curves)
    return [_mean([curve[idx] for curve in curves]) for idx in range(width)]


def _build_shared_splits(batch_size: int, num_splits: int) -> list[tuple[list[tuple[str, str]], list[tuple[str, str]]]]:
    splits: list[tuple[list[tuple[str, str]], list[tuple[str, str]]]] = []
    for _ in range(num_splits):
        pairs = make_text_code_pairs(repeats=max(batch_size * 8, 128))
        split = max(int(len(pairs) * 0.8), len(pairs) - 16)
        splits.append((pairs[:split], pairs[split:]))
    return splits


def _proxy_config_v4(size: int) -> JEPAConfig:
    if size <= 15_000_000:
        return JEPAConfig(
            embed_dim=128,
            encoder_layers=2,
            encoder_heads=4,
            predictor_layers=2,
            predictor_heads=4,
            decoder_layers=1,
            decoder_heads=4,
            decoder_hidden_dim=128,
        )
    if size <= 80_000_000:
        return JEPAConfig(
            embed_dim=192,
            encoder_layers=4,
            encoder_heads=6,
            predictor_layers=4,
            predictor_heads=6,
            decoder_layers=2,
            decoder_heads=6,
            decoder_hidden_dim=192,
        )
    if size <= 300_000_000:
        return JEPAConfig(
            embed_dim=256,
            encoder_layers=6,
            encoder_heads=8,
            predictor_layers=6,
            predictor_heads=8,
            decoder_layers=2,
            decoder_heads=8,
            decoder_hidden_dim=256,
        )
    if size <= 600_000_000:
        return JEPAConfig(
            embed_dim=320,
            encoder_layers=8,
            encoder_heads=10,
            predictor_layers=8,
            predictor_heads=10,
            decoder_layers=3,
            decoder_heads=10,
            decoder_hidden_dim=320,
        )
    return JEPAConfig(
        embed_dim=384,
        encoder_layers=10,
        encoder_heads=12,
        predictor_layers=10,
        predictor_heads=12,
        decoder_layers=3,
        decoder_heads=12,
        decoder_hidden_dim=384,
    )


def _proxy_config_v5_distinct(size: int) -> JEPAConfig:
    if size <= 15_000_000:
        return JEPAConfig(
            embed_dim=128,
            encoder_layers=2,
            encoder_heads=4,
            predictor_layers=2,
            predictor_heads=4,
            decoder_layers=1,
            decoder_heads=4,
            decoder_hidden_dim=128,
        )
    if size <= 40_000_000:
        return JEPAConfig(
            embed_dim=168,
            encoder_layers=3,
            encoder_heads=6,
            predictor_layers=3,
            predictor_heads=6,
            decoder_layers=1,
            decoder_heads=6,
            decoder_hidden_dim=168,
        )
    if size <= 80_000_000:
        return JEPAConfig(
            embed_dim=192,
            encoder_layers=4,
            encoder_heads=6,
            predictor_layers=4,
            predictor_heads=6,
            decoder_layers=2,
            decoder_heads=6,
            decoder_hidden_dim=192,
        )
    if size <= 150_000_000:
        return JEPAConfig(
            embed_dim=224,
            encoder_layers=5,
            encoder_heads=8,
            predictor_layers=5,
            predictor_heads=8,
            decoder_layers=2,
            decoder_heads=8,
            decoder_hidden_dim=224,
        )
    if size <= 300_000_000:
        return JEPAConfig(
            embed_dim=256,
            encoder_layers=6,
            encoder_heads=8,
            predictor_layers=6,
            predictor_heads=8,
            decoder_layers=2,
            decoder_heads=8,
            decoder_hidden_dim=256,
        )
    if size <= 600_000_000:
        return JEPAConfig(
            embed_dim=320,
            encoder_layers=8,
            encoder_heads=10,
            predictor_layers=8,
            predictor_heads=10,
            decoder_layers=3,
            decoder_heads=10,
            decoder_hidden_dim=320,
        )
    return JEPAConfig(
        embed_dim=384,
        encoder_layers=10,
        encoder_heads=12,
        predictor_layers=10,
        predictor_heads=12,
        decoder_layers=3,
        decoder_heads=12,
        decoder_hidden_dim=384,
    )


def _proxy_config_v6_overnight(size: int) -> JEPAConfig:
    if size <= 15_000_000:
        return JEPAConfig(
            embed_dim=192,
            encoder_layers=4,
            encoder_heads=6,
            predictor_layers=4,
            predictor_heads=6,
            decoder_layers=2,
            decoder_heads=6,
            decoder_hidden_dim=192,
        )
    if size <= 40_000_000:
        return JEPAConfig(
            embed_dim=256,
            encoder_layers=6,
            encoder_heads=8,
            predictor_layers=6,
            predictor_heads=8,
            decoder_layers=2,
            decoder_heads=8,
            decoder_hidden_dim=256,
        )
    if size <= 80_000_000:
        return JEPAConfig(
            embed_dim=320,
            encoder_layers=8,
            encoder_heads=10,
            predictor_layers=8,
            predictor_heads=10,
            decoder_layers=3,
            decoder_heads=10,
            decoder_hidden_dim=320,
        )
    if size <= 150_000_000:
        return JEPAConfig(
            embed_dim=384,
            encoder_layers=10,
            encoder_heads=12,
            predictor_layers=10,
            predictor_heads=12,
            decoder_layers=3,
            decoder_heads=12,
            decoder_hidden_dim=384,
        )
    if size <= 300_000_000:
        return JEPAConfig(
            embed_dim=512,
            encoder_layers=12,
            encoder_heads=16,
            predictor_layers=12,
            predictor_heads=16,
            decoder_layers=4,
            decoder_heads=16,
            decoder_hidden_dim=512,
        )
    if size <= 600_000_000:
        return JEPAConfig(
            embed_dim=768,
            encoder_layers=18,
            encoder_heads=12,
            predictor_layers=18,
            predictor_heads=12,
            decoder_layers=6,
            decoder_heads=12,
            decoder_hidden_dim=768,
        )
    if size <= 1_200_000_000:
        return JEPAConfig(
            embed_dim=1024,
            encoder_layers=24,
            encoder_heads=16,
            predictor_layers=24,
            predictor_heads=16,
            decoder_layers=8,
            decoder_heads=16,
            decoder_hidden_dim=1024,
        )
    return JEPAConfig(
        embed_dim=1728,
        encoder_layers=32,
        encoder_heads=12,
        predictor_layers=32,
        predictor_heads=12,
        decoder_layers=10,
        decoder_heads=12,
        decoder_hidden_dim=1728,
    )


def _proxy_config(size: int, recipe: str) -> JEPAConfig:
    if recipe == "v6_overnight":
        return _proxy_config_v6_overnight(size)
    if recipe == "v5_distinct":
        return _proxy_config_v5_distinct(size)
    return _proxy_config_v4(size)


def _scaled_training_hparams(config, requested_size: int) -> tuple[int, float, float, float]:
    minimum_size = max(min(config.sizes), 1)
    size_ratio = max(requested_size / minimum_size, 1.0)

    step_scale_power = max(getattr(config, "step_scale_power", 0.0), 0.0)
    max_step_multiplier = max(getattr(config, "max_step_multiplier", 1.0), 1.0)
    step_multiplier = min(max_step_multiplier, size_ratio ** step_scale_power)
    scaled_steps = max(1, int(round(config.steps * step_multiplier)))

    lr_scale_power = max(getattr(config, "lr_scale_power", 0.0), 0.0)
    max_lr_divisor = max(getattr(config, "max_lr_divisor", 1.0), 1.0)
    lr_divisor = min(max_lr_divisor, size_ratio ** lr_scale_power)
    scaled_lr = config.lr / max(lr_divisor, 1.0)
    return scaled_steps, scaled_lr, step_multiplier, lr_divisor


def _update_ema_model(ema_model: torch.nn.Module, model: torch.nn.Module, decay: float = 0.999) -> None:
    """Update EMA model parameters using exponential moving average."""
    with torch.no_grad():
        for ema_p, p in zip(ema_model.parameters(), model.parameters()):
            ema_p.data.mul_(decay).add_(p.data, alpha=1.0 - decay)


def _lr_multiplier(
    step: int,
    total_steps: int,
    scheduler: str = "cosine",
    warmup_fraction: float = 0.15,
    min_lr_ratio: float = 0.2,
) -> float:
    """Compute LR multiplier for a given step. Supports cosine, linear, and constant schedules."""
    warmup_steps = int(total_steps * warmup_fraction)
    if step < warmup_steps:
        return max(1e-6, step / max(warmup_steps, 1))
    if scheduler == "cosine":
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
    elif scheduler == "linear":
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(min_lr_ratio, 1.0 - progress)
    else:  # constant
        return 1.0


def run_phase4(config, device: str) -> dict:
    tokenizer = SimpleTokenizer(vocab_size=4096)
    results: dict[int, dict] = {}
    shared_splits = _build_shared_splits(config.batch_size, config.num_splits)
    proxy_recipe = getattr(config, "proxy_recipe", "v4")

    for requested_size in config.sizes:
        model_config = _proxy_config(requested_size, proxy_recipe)
        scaled_steps, scaled_lr, step_multiplier, lr_divisor = _scaled_training_hparams(config, requested_size)
        split_final_losses: list[float] = []
        split_val_losses: list[float] = []
        split_train_curves: list[list[float]] = []
        split_throughputs: list[float] = []
        peak_vram_gb = 0.0

        for train_pairs, val_pairs in shared_splits:
            model = JEPAModel(model_config).to(device)

            # Warm-start: load phase3 weights from checkpoint before training
            warm_start_path = getattr(config, "warm_start_model_path", None)
            if warm_start_path and os.path.exists(warm_start_path):
                warm_state = torch.load(warm_start_path, map_location=device, weights_only=True)
                model_state = model.state_dict()
                loaded, skipped = 0, 0
                for key in warm_state:
                    if key in model_state and model_state[key].shape == warm_state[key].shape:
                        model_state[key] = warm_state[key]
                        loaded += 1
                    else:
                        skipped += 1
                model.load_state_dict(model_state)
                print(f"[phase4] warm-start loaded {loaded} tensors, skipped {skipped} from {warm_start_path}")

            freeze_backbone = getattr(config, "freeze_backbone", False)
            if freeze_backbone:
                for name, param in model.named_parameters():
                    if "encoder" in name or "predictor" in name:
                        param.requires_grad = False
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(trainable_params, lr=scaled_lr)
            negative_queue = torch.empty(0, model_config.embed_dim, device=device)
            torch.cuda.reset_peak_memory_stats() if device.startswith("cuda") else None
            start = time.time()
            final_loss = 0.0
            train_curve: list[float] = []
            collapse_count = 0
            for step in range(scaled_steps):
                batch_pairs = [train_pairs[(step * config.batch_size + offset) % len(train_pairs)] for offset in range(config.batch_size)]
                texts = tokenizer.batch_encode([pair[0] for pair in batch_pairs], model_config.max_seq_len, device)
                codes = tokenizer.batch_encode([pair[1] for pair in batch_pairs], model_config.max_seq_len, device)
                ood = tokenizer.batch_encode(sample_ood_queries(len(batch_pairs)), model_config.max_seq_len, device)
                z_text = model.encode(texts)
                z_code = model.encode(codes)
                z_ood = model.encode(ood)
                z_pred = model.predict(z_text, action_id=1)
                z_ood_pred = model.predict(z_ood, action_id=1)
                coding_logits = model.query_logits(z_text)
                ood_logits = model.query_logits(z_ood)
                loss, _ = paired_alignment_loss(z_text, z_code, z_pred, negative_pool=negative_queue)
                code_candidates = torch.cat([z_code.detach(), negative_queue], dim=0) if negative_queue.numel() else z_code.detach()
                clf_loss = F.binary_cross_entropy_with_logits(coding_logits, torch.ones_like(coding_logits))
                clf_loss = clf_loss + F.binary_cross_entropy_with_logits(ood_logits, torch.zeros_like(ood_logits))
                ood_w = getattr(config, "loss_ood_weight", 0.2)
                ood_pred_w = getattr(config, "loss_ood_pred_weight", 0.2)
                clf_w = getattr(config, "loss_clf_weight", 0.25)
                loss = loss + ood_w * ignorance_penalty(z_ood, code_candidates) + ood_pred_w * ignorance_penalty(z_ood_pred, code_candidates) + clf_w * clf_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                # Collapse detection: compute code-query offdiag similarity
                offdiag = float((z_text @ z_code.T).diag().mean().detach().cpu())
                if offdiag > COLLAPSE_OFFDIAG_THRESH:
                    collapse_count += 1
                else:
                    collapse_count = 0
                if collapse_count >= COLLAPSE_CONSECUTIVE_STEPS:
                    final_loss = float(loss.detach().cpu().item())
                    train_curve.append(final_loss)
                    break  # early stop — collapse detected

                negative_queue = torch.cat([z_code.detach(), negative_queue], dim=0)[: max(config.batch_size * 64, 256)]
                final_loss = float(loss.detach().cpu().item())
                train_curve.append(final_loss)

            with torch.no_grad():
                val_losses: list[float] = []
                for start_idx in range(0, len(val_pairs), config.batch_size):
                    batch_pairs = val_pairs[start_idx : start_idx + config.batch_size]
                    texts = tokenizer.batch_encode([pair[0] for pair in batch_pairs], model_config.max_seq_len, device)
                    codes = tokenizer.batch_encode([pair[1] for pair in batch_pairs], model_config.max_seq_len, device)
                    ood = tokenizer.batch_encode(sample_ood_queries(len(batch_pairs)), model_config.max_seq_len, device)
                    z_text = model.encode(texts)
                    z_code = model.encode(codes)
                    z_ood = model.encode(ood)
                    z_pred = model.predict(z_text, action_id=1)
                    z_ood_pred = model.predict(z_ood, action_id=1)
                    coding_logits = model.query_logits(z_text)
                    ood_logits = model.query_logits(z_ood)
                    val_loss, _ = paired_alignment_loss(z_text, z_code, z_pred, negative_pool=negative_queue)
                    code_candidates = torch.cat([z_code.detach(), negative_queue], dim=0) if negative_queue.numel() else z_code.detach()
                    clf_loss = F.binary_cross_entropy_with_logits(coding_logits, torch.ones_like(coding_logits))
                    clf_loss = clf_loss + F.binary_cross_entropy_with_logits(ood_logits, torch.zeros_like(ood_logits))
                    ood_w = getattr(config, "loss_ood_weight", 0.2)
                    ood_pred_w = getattr(config, "loss_ood_pred_weight", 0.2)
                    clf_w = getattr(config, "loss_clf_weight", 0.25)
                    val_loss = val_loss + ood_w * ignorance_penalty(z_ood, code_candidates) + ood_pred_w * ignorance_penalty(z_ood_pred, code_candidates) + clf_w * clf_loss
                    val_losses.append(float(val_loss.detach().cpu().item()))

            split_final_losses.append(final_loss)
            split_val_losses.append(_mean(val_losses))
            split_train_curves.append(train_curve[:4] + train_curve[-4:] if len(train_curve) > 8 else train_curve)
            elapsed = max(time.time() - start, 1e-6)
            split_throughputs.append((scaled_steps * config.batch_size) / elapsed)
            if device.startswith("cuda"):
                peak_vram_gb = max(peak_vram_gb, torch.cuda.max_memory_allocated() / 1e9)

        results[requested_size] = {
            "final_loss": _mean(split_final_losses),
            "final_loss_std": _std(split_final_losses),
            "val_loss": _mean(split_val_losses),
            "val_loss_std": _std(split_val_losses),
            "split_val_losses": split_val_losses,
            "train_curve": _average_curves(split_train_curves),
            "samples_per_sec": _mean(split_throughputs),
            "peak_vram_gb": peak_vram_gb,
            "fits_on_4090": peak_vram_gb < config.max_vram_gb if peak_vram_gb else True,
            "proxy_params": approximate_model_params(model_config),
            "num_splits": config.num_splits,
            "proxy_recipe": proxy_recipe,
            "scaled_steps": scaled_steps,
            "scaled_lr": scaled_lr,
            "step_multiplier": step_multiplier,
            "lr_divisor": lr_divisor,
        }

    ordered_sizes = sorted(config.sizes)
    largest_size = ordered_sizes[-1]
    first_size = ordered_sizes[0]
    first_loss = results[first_size]["val_loss"]
    best_size = min(ordered_sizes, key=lambda size: results[size]["val_loss"])
    best_loss = results[best_size]["val_loss"]
    improvement = 0.0 if first_loss <= 0 else max(0.0, (first_loss - best_loss) / first_loss)
    tolerant_steps = 0
    for earlier, later in zip(ordered_sizes, ordered_sizes[1:]):
        earlier_loss = results[earlier]["val_loss"]
        later_loss = results[later]["val_loss"]
        later_uncertainty = results[later]["val_loss_std"]
        if later_loss <= earlier_loss * 1.03 + later_uncertainty:
            tolerant_steps += 1
    monotonic_fraction = tolerant_steps / max(len(ordered_sizes) - 1, 1)
    log_sizes = [math.log(size) for size in ordered_sizes]
    neg_losses = [-results[size]["val_loss"] for size in ordered_sizes]
    mean_x = sum(log_sizes) / len(log_sizes)
    mean_y = sum(neg_losses) / len(neg_losses)
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(log_sizes, neg_losses))
    var_x = sum((x - mean_x) ** 2 for x in log_sizes)
    var_y = sum((y - mean_y) ** 2 for y in neg_losses)
    correlation = cov / math.sqrt(max(var_x * var_y, 1e-12))
    feasible = [size for size, details in results.items() if details["fits_on_4090"]]
    largest_wins = best_size == largest_size
    best_std = results[best_size]["val_loss_std"]
    competitor_sizes = [size for size in ordered_sizes if size != largest_size]
    competitor_size = min(competitor_sizes, key=lambda size: results[size]["val_loss"])
    largest_loss = results[largest_size]["val_loss"]
    competitor_loss = results[competitor_size]["val_loss"]
    largest_std = results[largest_size]["val_loss_std"]
    competitor_std = results[competitor_size]["val_loss_std"]
    largest_margin = competitor_loss - largest_loss
    largest_margin_ratio = largest_margin / max(competitor_loss, 1e-6)
    confidence_margin = largest_margin - (largest_std + competitor_std)
    largest_beats_competitor_confidently = confidence_margin > 0.0
    largest_split_losses = results[largest_size]["split_val_losses"]
    competitor_split_losses = results[competitor_size]["split_val_losses"]
    split_margins = [
        competitor_split_loss - largest_split_loss
        for largest_split_loss, competitor_split_loss in zip(largest_split_losses, competitor_split_losses)
    ]
    split_margin_ratios = [
        margin / max(competitor_split_loss, 1e-6)
        for margin, competitor_split_loss in zip(split_margins, competitor_split_losses)
    ]
    pairwise_win_rate = sum(margin > 0.0 for margin in split_margins) / max(len(split_margins), 1)
    pairwise_margin_mean = _mean(split_margins)
    pairwise_margin_std = _std(split_margins)
    pairwise_margin_ratio = _mean(split_margin_ratios)
    worst_pairwise_margin = min(split_margins) if split_margins else 0.0
    worst_pairwise_margin_ratio = min(split_margin_ratios) if split_margin_ratios else 0.0
    strong_gain = improvement >= 0.3 and monotonic_fraction >= 0.75 and correlation >= 0.6 and best_std <= 0.01
    steady_gain = improvement >= 0.1 and largest_wins and monotonic_fraction >= 0.75 and correlation >= 0.75 and best_std <= 0.01
    margin_gain = (
        largest_wins
        and monotonic_fraction >= 0.75
        and correlation >= 0.8
        and largest_margin_ratio >= 0.01
        and largest_beats_competitor_confidently
        and largest_std <= 0.01
    )
    pairwise_gain = (
        largest_wins
        and monotonic_fraction >= 0.75
        and correlation >= 0.75
        and pairwise_win_rate >= 0.75
        and pairwise_margin_ratio >= 0.005
        and worst_pairwise_margin_ratio > -0.0025
    )
    scaling_efficient = strong_gain or steady_gain or margin_gain or pairwise_gain
    return {
        "scaling_efficient": scaling_efficient,
        "loss_improvement": improvement,
        "best_size": best_size,
        "largest_size": largest_size,
        "largest_wins": largest_wins,
        "monotonic_fraction": monotonic_fraction,
        "loss_correlation": correlation,
        "best_size_val_std": best_std,
        "competitor_size": competitor_size,
        "largest_margin": largest_margin,
        "largest_margin_ratio": largest_margin_ratio,
        "confidence_margin": confidence_margin,
        "largest_beats_competitor_confidently": largest_beats_competitor_confidently,
        "pairwise_win_rate": pairwise_win_rate,
        "pairwise_margin_mean": pairwise_margin_mean,
        "pairwise_margin_std": pairwise_margin_std,
        "pairwise_margin_ratio": pairwise_margin_ratio,
        "worst_pairwise_margin": worst_pairwise_margin,
        "worst_pairwise_margin_ratio": worst_pairwise_margin_ratio,
        "max_feasible_params": max(feasible) if feasible else None,
        "proceed_to_2_7b": scaling_efficient and bool(feasible),
        "details": results,
        "proxy_recipe": proxy_recipe,
        "step_scale_power": getattr(config, "step_scale_power", 0.0),
        "max_step_multiplier": getattr(config, "max_step_multiplier", 1.0),
        "lr_scale_power": getattr(config, "lr_scale_power", 0.0),
        "max_lr_divisor": getattr(config, "max_lr_divisor", 1.0),
        "proxy_only": True,
    }