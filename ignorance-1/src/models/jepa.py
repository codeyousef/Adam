from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class JEPAConfig:
    vocab_size: int = 4096
    patch_size: int = 32
    max_seq_len: int = 256
    embed_dim: int = 192
    encoder_layers: int = 4
    encoder_heads: int = 3
    predictor_layers: int = 6
    predictor_heads: int = 6
    predictor_dropout: float = 0.1
    decoder_layers: int = 2
    decoder_heads: int = 3
    decoder_hidden_dim: int = 192


class PatchTextEncoder(nn.Module):
    def __init__(self, config: JEPAConfig):
        super().__init__()
        self.config = config
        self.token_embed = nn.Embedding(config.vocab_size, config.embed_dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 1 + config.max_seq_len // config.patch_size, config.embed_dim)
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.encoder_heads,
            dim_feedforward=config.embed_dim * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=config.encoder_layers)
        self.final_ln = nn.LayerNorm(config.embed_dim)
        self.final_bn = nn.BatchNorm1d(config.embed_dim)
        self.proj = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim),
            nn.GELU(),
            nn.Linear(config.embed_dim, config.embed_dim),
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(self.token_embed.weight, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.proj:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _patchify(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch, seq_len = input_ids.shape
        patch = self.config.patch_size
        pad = (patch - (seq_len % patch)) % patch
        if pad:
            input_ids = F.pad(input_ids, (0, pad), value=0)
        embeds = self.token_embed(input_ids)
        embeds = embeds.view(batch, -1, patch, self.config.embed_dim).mean(dim=2)
        return embeds

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        patches = self._patchify(input_ids)
        batch = patches.shape[0]
        cls = self.cls.expand(batch, -1, -1)
        tokens = torch.cat([cls, patches], dim=1)
        tokens = tokens + self.pos_embed[:, : tokens.shape[1]]
        encoded = self.transformer(tokens)
        pooled = self.final_ln(encoded[:, 0])
        # Custom BatchNorm logic to handle bfloat16 + small batches
        bn_weight = self.final_bn.weight.float() if self.final_bn.weight is not None else None
        bn_bias = self.final_bn.bias.float() if self.final_bn.bias is not None else None
        if self.final_bn.training:
            pooled_float = pooled.float()
            if pooled.shape[0] > 1:
                batch_mean = pooled_float.mean(dim=0)
                batch_var = pooled_float.var(dim=0, unbiased=False)
                if self.final_bn.running_mean is not None and self.final_bn.running_var is not None:
                    momentum = self.final_bn.momentum if self.final_bn.momentum is not None else 0.1
                    self.final_bn.running_mean.mul_(1.0 - momentum).add_(momentum * batch_mean.to(self.final_bn.running_mean.dtype))
                    self.final_bn.running_var.mul_(1.0 - momentum).add_(momentum * batch_var.to(self.final_bn.running_var.dtype))
                normalized = (pooled_float - batch_mean) / torch.sqrt(batch_var + self.final_bn.eps)
            else:
                running_mean = self.final_bn.running_mean.float() if self.final_bn.running_mean is not None else pooled_float.new_zeros(pooled_float.shape[-1])
                running_var = self.final_bn.running_var.float() if self.final_bn.running_var is not None else pooled_float.new_ones(pooled_float.shape[-1])
                normalized = (pooled_float - running_mean) / torch.sqrt(running_var + self.final_bn.eps)
            if bn_weight is not None:
                normalized = normalized * bn_weight
            if bn_bias is not None:
                normalized = normalized + bn_bias
            pooled = normalized.to(pooled.dtype)
        else:
            # Eval mode - also use float32 for stability if in bfloat16 context
            pooled_float = pooled.float()
            running_mean = self.final_bn.running_mean.float() if self.final_bn.running_mean is not None else pooled_float.new_zeros(pooled_float.shape[-1])
            running_var = self.final_bn.running_var.float() if self.final_bn.running_var is not None else pooled_float.new_ones(pooled_float.shape[-1])
            normalized = (pooled_float - running_mean) / torch.sqrt(running_var + self.final_bn.eps)
            if bn_weight is not None:
                normalized = normalized * bn_weight
            if bn_bias is not None:
                normalized = normalized + bn_bias
            pooled = normalized.to(pooled.dtype)
        return self.proj(pooled)


class AdaLN(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.scale = nn.Linear(dim, dim)
        self.shift = nn.Linear(dim, dim)

    def forward(self, hidden: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(hidden)
        scale = self.scale(cond).unsqueeze(1)
        shift = self.shift(cond).unsqueeze(1)
        return normalized * (1.0 + scale) + shift


class JEPAPredictor(nn.Module):
    def __init__(self, config: JEPAConfig):
        super().__init__()
        self.config = config
        self.action_embed = nn.Embedding(4, config.embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, 2, config.embed_dim))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.predictor_heads,
            dim_feedforward=config.embed_dim * 4,
            dropout=config.predictor_dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=config.predictor_layers)
        self.adaln = AdaLN(config.embed_dim)
        self.head = nn.Sequential(
            nn.LayerNorm(config.embed_dim),
            nn.Linear(config.embed_dim, config.embed_dim),
            nn.GELU(),
            nn.Linear(config.embed_dim, config.embed_dim),
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(self.action_embed.weight, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(
        self,
        z_t: torch.Tensor,
        action_embed: torch.Tensor | None = None,
        action_id: int = 0,
    ) -> torch.Tensor:
        batch = z_t.shape[0]
        if action_embed is None:
            action_embed = self.action_embed.weight[action_id].unsqueeze(0).expand(batch, -1)
        tokens = torch.stack([z_t, action_embed], dim=1)
        tokens = tokens + self.pos_embed
        tokens = self.adaln(tokens, action_embed)
        hidden = self.transformer(tokens)
        return self.head(hidden[:, 0])

    def generate_query(self, z_t: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.head(z_t), dim=-1)


class LightweightDecoder(nn.Module):
    def __init__(self, config: JEPAConfig):
        super().__init__()
        dec_layer = nn.TransformerEncoderLayer(
            d_model=config.decoder_hidden_dim,
            nhead=config.decoder_heads,
            dim_feedforward=config.decoder_hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(dec_layer, num_layers=config.decoder_layers)
        self.head = nn.Linear(config.decoder_hidden_dim, config.vocab_size)

    def forward(self, latent: torch.Tensor, steps: int = 4) -> torch.Tensor:
        seq = latent.unsqueeze(1).repeat(1, steps, 1)
        hidden = self.transformer(seq)
        return self.head(hidden)


class JEPAModel(nn.Module):
    def __init__(self, config: JEPAConfig):
        super().__init__()
        self.config = config
        self.encoder = PatchTextEncoder(config)
        self.predictor = JEPAPredictor(config)
        self.decoder = LightweightDecoder(config)
        self.query_head = nn.Sequential(
            nn.LayerNorm(config.embed_dim),
            nn.Linear(config.embed_dim, config.embed_dim),
            nn.GELU(),
            nn.Linear(config.embed_dim, 1),
        )
        self._reset_query_head()

    def _reset_query_head(self) -> None:
        for module in self.query_head:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.encoder(input_ids)

    def predict(self, z_t: torch.Tensor, action_embed: torch.Tensor | None = None, action_id: int = 0) -> torch.Tensor:
        return self.predictor(z_t, action_embed=action_embed, action_id=action_id)

    def query_logits(self, z_t: torch.Tensor) -> torch.Tensor:
        return self.query_head(z_t).squeeze(-1)

    def query_confidence(self, z_t: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.query_logits(z_t))

    def generate(self, latent: torch.Tensor, steps: int = 4) -> torch.Tensor:
        return self.decoder(latent, steps=steps)


def approximate_model_params(config: JEPAConfig) -> int:
    model = JEPAModel(config)
    return sum(param.numel() for param in model.parameters())