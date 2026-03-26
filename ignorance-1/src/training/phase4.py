from __future__ import annotations

import math
import time

import torch
import torch.nn.functional as F

from src.models.jepa import JEPAConfig, JEPAModel, approximate_model_params
from src.utils.data import SimpleTokenizer, make_text_code_pairs


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


def _proxy_config(size: int, recipe: str) -> JEPAConfig:
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
            optimizer = torch.optim.AdamW(model.parameters(), lr=scaled_lr)
            torch.cuda.reset_peak_memory_stats() if device.startswith("cuda") else None
            start = time.time()
            final_loss = 0.0
            train_curve: list[float] = []

            for step in range(scaled_steps):
                batch_pairs = [train_pairs[(step * config.batch_size + offset) % len(train_pairs)] for offset in range(config.batch_size)]
                texts = tokenizer.batch_encode([pair[0] for pair in batch_pairs], model_config.max_seq_len, device)
                codes = tokenizer.batch_encode([pair[1] for pair in batch_pairs], model_config.max_seq_len, device)
                z_text = model.encode(texts)
                z_code = model.encode(codes)
                z_pred = model.predict(z_text, action_id=1)
                loss = F.mse_loss(z_pred, z_code)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                final_loss = float(loss.detach().cpu().item())
                train_curve.append(final_loss)

            with torch.no_grad():
                val_losses: list[float] = []
                for start_idx in range(0, len(val_pairs), config.batch_size):
                    batch_pairs = val_pairs[start_idx : start_idx + config.batch_size]
                    texts = tokenizer.batch_encode([pair[0] for pair in batch_pairs], model_config.max_seq_len, device)
                    codes = tokenizer.batch_encode([pair[1] for pair in batch_pairs], model_config.max_seq_len, device)
                    z_text = model.encode(texts)
                    z_code = model.encode(codes)
                    z_pred = model.predict(z_text, action_id=1)
                    val_losses.append(float(F.mse_loss(z_pred, z_code).detach().cpu().item()))

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