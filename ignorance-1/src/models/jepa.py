from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


MODEL_BRAND_NAME = "Sinai"


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
    use_final_batch_norm: bool = True
    use_retrieval_head: bool = False
    retrieval_head_dim: int = 0
    retrieval_head_hidden_dim: int = 0
    use_retrieval_facets: bool = False
    retrieval_num_facets: int = 0
    retrieval_facet_dim: int = 0
    retrieval_facet_hidden_dim: int = 0
    retrieval_facet_separate_query_code: bool = False
    use_gated_reranker: bool = False
    gated_reranker_hidden_dim: int = 128
    gated_reranker_num_heads: int = 4
    gated_reranker_score_mode: str = "cross_attention"


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
        if not self.config.use_final_batch_norm:
            return self.proj(pooled)
        bn_requires_float_path = (
            pooled.dtype in (torch.float16, torch.bfloat16)
            or (self.final_bn.running_mean is not None and pooled.dtype != self.final_bn.running_mean.dtype)
            or (self.final_bn.running_var is not None and pooled.dtype != self.final_bn.running_var.dtype)
            or (self.final_bn.weight is not None and pooled.dtype != self.final_bn.weight.dtype)
            or (self.final_bn.bias is not None and pooled.dtype != self.final_bn.bias.dtype)
        )
        if bn_requires_float_path:
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
                pooled_float = pooled.float()
                running_mean = self.final_bn.running_mean.float() if self.final_bn.running_mean is not None else pooled_float.new_zeros(pooled_float.shape[-1])
                running_var = self.final_bn.running_var.float() if self.final_bn.running_var is not None else pooled_float.new_ones(pooled_float.shape[-1])
                normalized = (pooled_float - running_mean) / torch.sqrt(running_var + self.final_bn.eps)
                if bn_weight is not None:
                    normalized = normalized * bn_weight
                if bn_bias is not None:
                    normalized = normalized + bn_bias
                pooled = normalized.to(pooled.dtype)
        elif pooled.shape[0] == 1 and self.final_bn.training:
            pooled_float = pooled.float()
            running_mean = self.final_bn.running_mean.float() if self.final_bn.running_mean is not None else pooled_float.new_zeros(pooled_float.shape[-1])
            running_var = self.final_bn.running_var.float() if self.final_bn.running_var is not None else pooled_float.new_ones(pooled_float.shape[-1])
            normalized = (pooled_float - running_mean) / torch.sqrt(running_var + self.final_bn.eps)
            if self.final_bn.weight is not None:
                normalized = normalized * self.final_bn.weight.float()
            if self.final_bn.bias is not None:
                normalized = normalized + self.final_bn.bias.float()
            pooled = normalized.to(pooled.dtype)
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


class RetrievalProjectionHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 0):
        super().__init__()
        resolved_output_dim = output_dim if output_dim > 0 else input_dim
        resolved_hidden_dim = hidden_dim if hidden_dim > 0 else max(input_dim, resolved_output_dim)
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Linear(input_dim, resolved_hidden_dim)
        self.output_proj = nn.Linear(resolved_hidden_dim, resolved_output_dim)
        self.residual_proj = nn.Identity() if input_dim == resolved_output_dim else nn.Linear(input_dim, resolved_output_dim)
        self.output_norm = nn.LayerNorm(resolved_output_dim)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        hidden = self.input_proj(self.input_norm(latent))
        hidden = F.gelu(hidden)
        projected = self.output_proj(hidden)
        residual = self.residual_proj(latent)
        return self.output_norm(projected + residual)


class RetrievalFacetHead(nn.Module):
    def __init__(self, input_dim: int, facet_dim: int, num_facets: int, hidden_dim: int = 0):
        super().__init__()
        resolved_hidden_dim = hidden_dim if hidden_dim > 0 else max(input_dim, facet_dim)
        self.num_facets = max(int(num_facets), 1)
        self.facet_dim = facet_dim
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Linear(input_dim, resolved_hidden_dim)
        self.output_proj = nn.Linear(resolved_hidden_dim, self.num_facets * facet_dim)
        self.slot_bias = nn.Parameter(torch.zeros(self.num_facets, facet_dim))
        self.output_norm = nn.LayerNorm(facet_dim)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        hidden = self.input_proj(self.input_norm(latent))
        hidden = F.gelu(hidden)
        projected = self.output_proj(hidden).view(latent.shape[0], self.num_facets, self.facet_dim)
        return self.output_norm(projected + self.slot_bias.unsqueeze(0))


class GatedRerankerHead(nn.Module):
    """Cross-encoder-style late-interaction reranker.

    Takes [query_slots | candidate_slots] (concatenated along feature dim) and
    uses slot-level self-attention + MLP to produce a scalar score per candidate.

    Architecture:
    - Concatenate [B, S, 2*D] and [C, S, 2*D]
    - Self-attention over slots (interaction)
    - Mean pooling over slots
    - MLP -> scalar score per candidate

    This is a proper cross-encoder that allows full query-candidate interaction,
    unlike the bi-encoder which only does cosine similarity.
    """

    def __init__(
        self,
        *,
        num_slots: int,
        facet_dim: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.facet_dim = facet_dim
        pair_dim = 2 * facet_dim

        # Self-attention over slots for interaction
        self.attn = nn.MultiheadAttention(
            embed_dim=pair_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(pair_dim)

        # Project to hidden dim
        self.proj = nn.Sequential(
            nn.LayerNorm(pair_dim),
            nn.Linear(pair_dim, hidden_dim),
            nn.GELU(),
        )

        # Score head: mean pool over slots -> scalar
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim, 1, bias=False),
        )

    def forward(
        self,
        query_slots: torch.Tensor,      # [B, num_slots, facet_dim]
        candidate_slots: torch.Tensor,  # [C, num_slots, facet_dim]
    ) -> torch.Tensor:
        """Score each candidate against each query.
        Returns: [B, C] scores
        """
        B, S, D = query_slots.shape
        C = candidate_slots.shape[0]

        # Build pair tensor: [B*C, S, 2*D]
        # For each pair (i,j): concat query_slots[i] and candidate_slots[j]
        q_expanded = query_slots.view(B, 1, S, D).repeat(1, C, 1, 1)  # [B, C, S, D]
        c_expanded = candidate_slots.view(1, C, S, D).repeat(B, 1, 1, 1)  # [B, C, S, D]
        pairs = torch.cat([q_expanded, c_expanded], dim=-1)  # [B, C, S, 2*D]
        pairs = pairs.reshape(B * C, S, 2 * D)                     # [B*C, S, 2*D]

        # Self-attention over slots
        attn_out, _ = self.attn(pairs, pairs, pairs)  # [B*C, S, 2*D]
        attn_out = self.attn_norm(attn_out + pairs)    # residual
        pooled = attn_out.mean(dim=1)                  # [B*C, 2*D]

        # Project and score
        hidden = self.proj(pooled)                      # [B*C, hidden_dim]
        raw_scores = self.score_head(hidden).squeeze(-1)  # [B*C]
        scores = raw_scores.view(B, C)                  # [B, C]

        # Per-query centering (subtract mean) for numerical stability
        scores = scores - scores.mean(dim=1, keepdim=True)
        return scores

    def forward_raw(
        self,
        query_slots: torch.Tensor,      # [B, num_slots, facet_dim]
        candidate_slots: torch.Tensor,  # [C, num_slots, facet_dim]
    ) -> torch.Tensor:
        """Score without centering — returns [B, C] raw scores for training."""
        B, S, D = query_slots.shape
        C = candidate_slots.shape[0]
        q_expanded = query_slots.view(B, 1, S, D).repeat(1, C, 1, 1)
        c_expanded = candidate_slots.view(1, C, S, D).repeat(B, 1, 1, 1)
        pairs = torch.cat([q_expanded, c_expanded], dim=-1)
        pairs = pairs.reshape(B * C, S, 2 * D)
        attn_out, _ = self.attn(pairs, pairs, pairs)
        attn_out = self.attn_norm(attn_out + pairs)
        pooled = attn_out.mean(dim=1)
        hidden = self.proj(pooled)
        raw_scores = self.score_head(hidden).squeeze(-1)
        return raw_scores.view(B, C)


class JEPAModel(nn.Module):
    def __init__(self, config: JEPAConfig):
        super().__init__()
        self.config = config
        self.encoder = PatchTextEncoder(config)
        self.predictor = JEPAPredictor(config)
        self.decoder = LightweightDecoder(config)
        if config.use_retrieval_head:
            self.retrieval_head = RetrievalProjectionHead(
                input_dim=config.embed_dim,
                output_dim=config.retrieval_head_dim,
                hidden_dim=config.retrieval_head_hidden_dim,
            )
            self.retrieval_dim = config.retrieval_head_dim if config.retrieval_head_dim > 0 else config.embed_dim
        else:
            self.retrieval_head = None
            self.retrieval_dim = config.embed_dim
        self.use_retrieval_facets = bool(config.use_retrieval_facets and int(config.retrieval_num_facets or 0) > 0)
        self.retrieval_num_facets = int(config.retrieval_num_facets or 0) if self.use_retrieval_facets else 1
        self.retrieval_facet_dim = int(config.retrieval_facet_dim or 0) if int(config.retrieval_facet_dim or 0) > 0 else self.retrieval_dim
        self.retrieval_facet_hidden_dim = int(config.retrieval_facet_hidden_dim or 0)
        self.retrieval_facet_separate_query_code = bool(config.retrieval_facet_separate_query_code)
        if self.use_retrieval_facets:
            if self.retrieval_facet_separate_query_code:
                self.query_retrieval_facet_head = RetrievalFacetHead(
                    input_dim=self.retrieval_dim,
                    facet_dim=self.retrieval_facet_dim,
                    num_facets=self.retrieval_num_facets,
                    hidden_dim=self.retrieval_facet_hidden_dim,
                )
                self.code_retrieval_facet_head = RetrievalFacetHead(
                    input_dim=self.retrieval_dim,
                    facet_dim=self.retrieval_facet_dim,
                    num_facets=self.retrieval_num_facets,
                    hidden_dim=self.retrieval_facet_hidden_dim,
                )
                self.retrieval_facet_head = None
            else:
                self.retrieval_facet_head = RetrievalFacetHead(
                    input_dim=self.retrieval_dim,
                    facet_dim=self.retrieval_facet_dim,
                    num_facets=self.retrieval_num_facets,
                    hidden_dim=self.retrieval_facet_hidden_dim,
                )
                self.query_retrieval_facet_head = None
                self.code_retrieval_facet_head = None
        else:
            self.retrieval_facet_head = None
            self.query_retrieval_facet_head = None
            self.code_retrieval_facet_head = None
        self.query_head = nn.Sequential(
            nn.LayerNorm(config.embed_dim),
            nn.Linear(config.embed_dim, config.embed_dim),
            nn.GELU(),
            nn.Linear(config.embed_dim, 1),
        )
        self._reset_query_head()

        # Gated late-interaction reranker head (two-tower)
        self.use_gated_reranker = bool(config.use_gated_reranker and self.use_retrieval_facets)
        if self.use_gated_reranker:
            self.gated_reranker = GatedRerankerHead(
                num_slots=self.retrieval_num_facets,
                facet_dim=self.retrieval_facet_dim,
                hidden_dim=config.gated_reranker_hidden_dim,
                num_heads=config.gated_reranker_num_heads,
            )
        else:
            self.gated_reranker = None

    def _reset_query_head(self) -> None:
        for module in self.query_head:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.encoder(input_ids)

    def retrieval_global(self, latent: torch.Tensor) -> torch.Tensor:
        if self.retrieval_head is None:
            return latent
        return self.retrieval_head(latent)

    def retrieval_project(self, latent: torch.Tensor) -> torch.Tensor:
        return self.retrieval_global(latent)

    def retrieval_facets(self, latent: torch.Tensor, role: str = "query") -> torch.Tensor:
        global_latent = self.retrieval_global(latent)
        if not self.use_retrieval_facets:
            return global_latent.unsqueeze(1)
        normalized_role = str(role or "query").strip().lower()
        if self.retrieval_facet_head is not None:
            return self.retrieval_facet_head(global_latent)
        if normalized_role == "query":
            return self.query_retrieval_facet_head(global_latent)
        return self.code_retrieval_facet_head(global_latent)

    def retrieval_encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.retrieval_project(self.encode(input_ids))

    def gated_rerank(
        self,
        query_slots: torch.Tensor,      # [B, num_slots, facet_dim]
        candidate_slots: torch.Tensor,  # [B, num_slots, facet_dim]
    ) -> torch.Tensor:
        """Score each candidate against each query using the gated reranker.

        Returns: [B, C] scores
        """
        if self.gated_reranker is None:
            raise RuntimeError("Gated reranker not initialized. Set use_gated_reranker=True.")
        return self.gated_reranker(query_slots, candidate_slots)

    def gated_rerank_raw(
        self,
        query_slots: torch.Tensor,
        candidate_slots: torch.Tensor,
    ) -> torch.Tensor:
        """Raw un-centered scores for training — gap is meaningful across candidates."""
        if self.gated_reranker is None:
            raise RuntimeError("Gated reranker not initialized. Set use_gated_reranker=True.")
        return self.gated_reranker.forward_raw(query_slots, candidate_slots)

    def predict(self, z_t: torch.Tensor, action_embed: torch.Tensor | None = None, action_id: int = 0) -> torch.Tensor:
        return self.predictor(z_t, action_embed=action_embed, action_id=action_id)

    def retrieval_predict(self, z_t: torch.Tensor, action_embed: torch.Tensor | None = None, action_id: int = 0) -> torch.Tensor:
        return self.retrieval_project(self.predict(z_t, action_embed=action_embed, action_id=action_id))

    def query_logits(self, z_t: torch.Tensor) -> torch.Tensor:
        return self.query_head(z_t).squeeze(-1)

    def query_confidence(self, z_t: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.query_logits(z_t))

    def generate(self, latent: torch.Tensor, steps: int = 4) -> torch.Tensor:
        return self.decoder(latent, steps=steps)


def approximate_model_params(config: JEPAConfig) -> int:
    model = JEPAModel(config)
    return sum(param.numel() for param in model.parameters())