from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from src.models.jepa import JEPAConfig, JEPAModel, approximate_model_params
from src.utils.data import SimpleTokenizer, make_text_code_pairs


def _proxy_config(size: int) -> JEPAConfig:
    if size <= 15_000_000:
        return JEPAConfig(embed_dim=128, encoder_layers=2, encoder_heads=4, predictor_layers=2, predictor_heads=4)
    if size <= 150_000_000:
        return JEPAConfig(embed_dim=192, encoder_layers=4, encoder_heads=6, predictor_layers=6, predictor_heads=6)
    return JEPAConfig(embed_dim=256, encoder_layers=6, encoder_heads=8, predictor_layers=8, predictor_heads=8)


def run_phase4(config, device: str) -> dict:
    tokenizer = SimpleTokenizer(vocab_size=4096)
    pairs = make_text_code_pairs(repeats=max(config.batch_size, 16))
    results: dict[int, dict] = {}

    for requested_size in config.sizes:
        model_config = _proxy_config(requested_size)
        model = JEPAModel(model_config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
        torch.cuda.reset_peak_memory_stats() if device.startswith("cuda") else None
        start = time.time()
        final_loss = 0.0

        for step in range(config.steps):
            batch_pairs = [pairs[(step * config.batch_size + offset) % len(pairs)] for offset in range(config.batch_size)]
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

        elapsed = max(time.time() - start, 1e-6)
        throughput = (config.steps * config.batch_size) / elapsed
        peak_vram_gb = 0.0
        if device.startswith("cuda"):
            peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9
        results[requested_size] = {
            "final_loss": final_loss,
            "samples_per_sec": throughput,
            "peak_vram_gb": peak_vram_gb,
            "fits_on_4090": peak_vram_gb < config.max_vram_gb if peak_vram_gb else True,
            "proxy_params": approximate_model_params(model_config),
        }

    first_size = config.sizes[0]
    last_size = config.sizes[-1]
    loss_first = results[first_size]["final_loss"]
    loss_last = results[last_size]["final_loss"]
    improvement = 0.0 if loss_first <= 0 else max(0.0, (loss_first - loss_last) / loss_first)
    feasible = [size for size, details in results.items() if details["fits_on_4090"]]
    return {
        "scaling_efficient": improvement > 0.5,
        "loss_improvement": improvement,
        "max_feasible_params": max(feasible) if feasible else None,
        "proceed_to_2_7b": improvement > 0.5 and bool(feasible),
        "details": results,
        "proxy_only": True,
    }