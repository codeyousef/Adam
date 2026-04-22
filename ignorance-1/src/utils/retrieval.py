from __future__ import annotations

import re
from dataclasses import dataclass

import torch
import torch.nn.functional as F



@dataclass
class RetrievalResult:
    ids: list[str]
    embeddings: torch.Tensor
    scores: torch.Tensor


class VectorIndex:
    def __init__(
        self,
        doc_ids: list[str],
        embeddings: torch.Tensor,
        *,
        facet_embeddings: torch.Tensor | None = None,
        facet_score_mode: str = "hard_maxsim",
        global_facet_blend: float = 1.0,
        facet_softmax_temperature: float = 0.1,
    ):
        self.doc_ids = doc_ids
        self.embeddings = F.normalize(embeddings.float(), dim=-1)
        self.facet_embeddings = (
            F.normalize(facet_embeddings.float(), dim=-1)
            if facet_embeddings is not None and facet_embeddings.numel() > 0
            else None
        )
        self.facet_score_mode = str(facet_score_mode or "hard_maxsim").strip().lower()
        self.global_facet_blend = float(global_facet_blend)
        self.facet_softmax_temperature = float(facet_softmax_temperature)
        self.doc_tokens = [set(re.findall(r"[a-z0-9_]+", doc.lower())) for doc in doc_ids]
        self.backend = "torch"
        self._faiss_index = None
        try:
            import faiss  # type: ignore

            cpu_embeddings = self.embeddings.cpu().numpy()
            index = faiss.IndexFlatIP(cpu_embeddings.shape[1])
            index.add(cpu_embeddings)
            self._faiss_index = index
            self.backend = "faiss"
        except Exception:
            self._faiss_index = None

    def search(
        self,
        queries: torch.Tensor,
        k: int = 1,
        *,
        query_facets: torch.Tensor | None = None,
    ) -> RetrievalResult:
        """
        First-stage retrieval. Uses facets when query_facets is provided and
        global_facet_blend < 1.0 (i.e., when multi-vector ColBERT-style scoring is desired).
        Falls back to global embedding only when facets are unavailable or when
        global_facet_blend=1.0 (legacy single-vector mode).
        """
        queries = F.normalize(queries.float(), dim=-1)
        embedding_scores = (queries @ self.embeddings.T).squeeze(0)

        # Research9 v416: Use facets for first-stage retrieval when available.
        # This is the ColBERT-style multi-vector first-stage: max-sim over facets
        # instead of single-vector dot product. Activated by global_facet_blend < 1.0.
        facet_scores: torch.Tensor | None = None
        if query_facets is not None and self.facet_embeddings is not None and self.global_facet_blend < 1.0:
                query_facets.to(queries.device),
        blend = min(max(float(self.global_facet_blend), 0.0), 1.0)
        if facet_scores is not None and blend < 1.0:
            # Blend global and facet scores
            embedding_scores = blend * embedding_scores + (1.0 - blend) * facet_scores.to(dtype=embedding_scores.dtype)
        # else: use global embedding scores as-is (legacy path)

        if self._faiss_index is not None and blend >= 1.0 and query_facets is None:
            # Pure global retrieval via FAISS (no facet contribution)
            import numpy as np

            scores_np = queries.squeeze(0).cpu().numpy().astype(np.float32).reshape(1, -1)
            scores_np = scores_np / (np.linalg.norm(scores_np) + 1e-8)
            raw_scores, indices = self._faiss_index.search(scores_np, k)
            idx_tensor = torch.tensor(indices, dtype=torch.long)
            score_tensor = torch.tensor(raw_scores, dtype=torch.float32)
            gathered = self.embeddings[idx_tensor]
            flat_ids = [self.doc_ids[idx] for idx in idx_tensor[:, 0].tolist()]
            return RetrievalResult(ids=flat_ids, embeddings=gathered[:, 0], scores=score_tensor[:, 0])

        # Torch backend or facet-aware scoring path
        top_scores, top_idx = torch.topk(embedding_scores.unsqueeze(0), k=k, dim=-1)
        gathered = self.embeddings[top_idx]
        flat_ids = [self.doc_ids[idx] for idx in top_idx[0].tolist()]
        return RetrievalResult(ids=flat_ids, embeddings=gathered[0], scores=top_scores[0])

    def search_text(
        self,
        query_text: str,
        queries: torch.Tensor,
        k: int = 1,
        lexical_weight: float = 0.7,
        *,
        query_facets: torch.Tensor | None = None,
    ) -> RetrievalResult:
        if queries.ndim != 2 or queries.shape[0] != 1:
            raise ValueError(f"Expected a single query embedding with shape [1, D], got {tuple(queries.shape)}")

        embeddings = self.embeddings.to(queries.device)
        embedding_scores = (F.normalize(queries.float(), dim=-1) @ embeddings.T).squeeze(0)
        # Facet-based scoring disabled — late_interaction_score_matrix not available
        # Re-enable by restoring the function to src/losses/alignment.py

        query_tokens = set(re.findall(r"[a-z0-9_]+", query_text.lower()))
        lexical_scores = []
        for doc_tokens in self.doc_tokens:
            if not query_tokens or not doc_tokens:
                lexical_scores.append(0.0)
                continue
            overlap = len(query_tokens & doc_tokens)
            lexical_scores.append(overlap / max(len(query_tokens | doc_tokens), 1))
        lexical_tensor = torch.tensor(lexical_scores, device=queries.device, dtype=embedding_scores.dtype)
        combined = (1.0 - lexical_weight) * embedding_scores + lexical_weight * lexical_tensor
        top_scores, top_idx = torch.topk(combined.unsqueeze(0), k=k, dim=-1)
        gathered = embeddings[top_idx]
        flat_ids = [self.doc_ids[idx] for idx in top_idx[0].tolist()]
        return RetrievalResult(ids=flat_ids, embeddings=gathered[0], scores=top_scores[0])
