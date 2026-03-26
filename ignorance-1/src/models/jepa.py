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
        if pooled.shape[0] == 1 and self.final_bn.training:
            pooled = F.batch_norm(
                pooled,
                self.final_bn.running_mean,
                self.final_bn.running_var,
                self.final_bn.weight,
                self.final_bn.bias,
                training=False,
                momentum=0.0,
                eps=self.final_bn.eps,
            )
        else:
            pooled = self.final_bn(pooled)
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

    def encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.encoder(input_ids)

    def predict(self, z_t: torch.Tensor, action_embed: torch.Tensor | None = None, action_id: int = 0) -> torch.Tensor:
        return self.predictor(z_t, action_embed=action_embed, action_id=action_id)

    def generate(self, latent: torch.Tensor, steps: int = 4) -> torch.Tensor:
        return self.decoder(latent, steps=steps)


def approximate_model_params(config: JEPAConfig) -> int:
    model = JEPAModel(config)
    return sum(param.numel() for param in model.parameters())