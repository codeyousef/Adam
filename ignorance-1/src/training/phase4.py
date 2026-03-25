from __future__ import annotations

import math
import time

import torch
import torch.nn.functional as F

from src.models.jepa import JEPAConfig, JEPAModel, approximate_model_params
from src.utils.data import SimpleTokenizer, make_text_code_pairs


def _proxy_config(size: int) -> JEPAConfig:
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


def run_phase4(config, device: str) -> dict:
    tokenizer = SimpleTokenizer(vocab_size=4096)
    pairs = make_text_code_pairs(repeats=max(config.batch_size * 8, 128))
    split = max(int(len(pairs) * 0.8), len(pairs) - 16)
    train_pairs = pairs[:split]
    val_pairs = pairs[split:]
    results: dict[int, dict] = {}

    for requested_size in config.sizes:
        model_config = _proxy_config(requested_size)
        model = JEPAModel(model_config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
        torch.cuda.reset_peak_memory_stats() if device.startswith("cuda") else None
        start = time.time()
        final_loss = 0.0
        train_curve: list[float] = []

        for step in range(config.steps):
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
        val_loss = sum(val_losses) / max(len(val_losses), 1)

        elapsed = max(time.time() - start, 1e-6)
        throughput = (config.steps * config.batch_size) / elapsed
        peak_vram_gb = 0.0
        if device.startswith("cuda"):
            peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9
        results[requested_size] = {
            "final_loss": final_loss,
            "val_loss": val_loss,
            "train_curve": train_curve[:4] + train_curve[-4:] if len(train_curve) > 8 else train_curve,
            "samples_per_sec": throughput,
            "peak_vram_gb": peak_vram_gb,
            "fits_on_4090": peak_vram_gb < config.max_vram_gb if peak_vram_gb else True,
            "proxy_params": approximate_model_params(model_config),
        }

    ordered_sizes = sorted(config.sizes)
    first_size = ordered_sizes[0]
    first_loss = results[first_size]["val_loss"]
    best_size = min(ordered_sizes, key=lambda size: results[size]["val_loss"])
    best_loss = results[best_size]["val_loss"]
    improvement = 0.0 if first_loss <= 0 else max(0.0, (first_loss - best_loss) / first_loss)
    tolerant_steps = 0
    for earlier, later in zip(ordered_sizes, ordered_sizes[1:]):
        earlier_loss = results[earlier]["val_loss"]
        later_loss = results[later]["val_loss"]
        if later_loss <= earlier_loss * 1.03:
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
    largest_wins = best_size == ordered_sizes[-1]
    strong_gain = improvement >= 0.3 and monotonic_fraction >= 0.75 and correlation >= 0.6
    steady_gain = improvement >= 0.1 and largest_wins and monotonic_fraction >= 0.75 and correlation >= 0.75
    scaling_efficient = strong_gain or steady_gain
    return {
        "scaling_efficient": scaling_efficient,
        "loss_improvement": improvement,
        "best_size": best_size,
        "largest_wins": largest_wins,
        "monotonic_fraction": monotonic_fraction,
        "loss_correlation": correlation,
        "max_feasible_params": max(feasible) if feasible else None,
        "proceed_to_2_7b": scaling_efficient and bool(feasible),
        "details": results,
        "proxy_only": True,
    }