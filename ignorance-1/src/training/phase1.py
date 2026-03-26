from __future__ import annotations

from dataclasses import asdict

import torch
import torch.nn.functional as F

from src.losses.sigreg import collapse_detected, gaussian_projection_p_value, isotropic_score, sigreg_loss
from src.models.jepa import JEPAConfig, JEPAModel
from src.utils.data import SimpleTokenizer, make_text_code_pairs


def run_phase1(config, device: str) -> dict:
    tokenizer = SimpleTokenizer(vocab_size=config.vocab_size)
    pairs = make_text_code_pairs(repeats=max(config.batch_size * 4, 64))
    results: dict[float, dict] = {}

    for lambda_reg in config.lambdas:
        model = JEPAModel(
            JEPAConfig(
                vocab_size=config.vocab_size,
                patch_size=config.patch_size,
                max_seq_len=config.seq_len,
                embed_dim=config.embed_dim,
                encoder_layers=config.encoder_layers,
                encoder_heads=config.encoder_heads,
                predictor_layers=config.predictor_layers,
                predictor_heads=config.predictor_heads,
            )
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
        losses: list[float] = []

        for step in range(config.steps):
            batch_pairs = [pairs[(step * config.batch_size + offset) % len(pairs)] for offset in range(config.batch_size)]
            texts = [pair[0] for pair in batch_pairs]
            codes = [pair[1] for pair in batch_pairs]

            text_ids = tokenizer.batch_encode(texts, config.seq_len, device)
            code_ids = tokenizer.batch_encode(codes, config.seq_len, device)

            z_text = model.encode(text_ids)
            z_code = model.encode(code_ids)
            z_pred = model.predict(z_text, action_id=1)

            pred_loss = F.mse_loss(z_pred, z_code)
            reg_loss = sigreg_loss(torch.stack([z_text, z_code], dim=1), m=config.projections, lambda_reg=lambda_reg)
            loss = pred_loss + lambda_reg * reg_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))

        with torch.no_grad():
            eval_count = min(len(pairs), max(config.batch_size * 4, 64))
            eval_pairs = pairs[:eval_count]
            encoded_chunks: list[torch.Tensor] = []
            for start in range(0, len(eval_pairs), config.batch_size):
                chunk = eval_pairs[start : start + config.batch_size]
                eval_text = tokenizer.batch_encode([pair[0] for pair in chunk], config.seq_len, device)
                eval_code = tokenizer.batch_encode([pair[1] for pair in chunk], config.seq_len, device)
                encoded_chunks.append(torch.stack([model.encode(eval_text), model.encode(eval_code)], dim=1))
            z = torch.cat(encoded_chunks, dim=0)

        p_value = gaussian_projection_p_value(z, num_projections=24)
        iso = isotropic_score(z)
        collapsed = collapse_detected(z)
        well_conditioned = iso >= 0.89 and p_value > 0.05 and not collapsed
        results[lambda_reg] = {
            "converged": losses[-1] < losses[0],
            "final_loss": losses[-1],
            "isotropic": well_conditioned,
            "isotropic_score": iso,
            "gaussian_p": p_value,
            "gaussian_pass": p_value > 0.05,
            "collapse_detected": collapsed,
        }

    valid = [value for value, details in results.items() if details["converged"] and details["isotropic"] and not details["collapse_detected"]]
    return {
        "optimal_lambda": valid[0] if valid else None,
        "details": results,
    }