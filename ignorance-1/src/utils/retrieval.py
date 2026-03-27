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
    def __init__(self, doc_ids: list[str], embeddings: torch.Tensor):
        self.doc_ids = doc_ids
        self.embeddings = F.normalize(embeddings.float(), dim=-1)
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

    def search(self, queries: torch.Tensor, k: int = 1) -> RetrievalResult:
        queries = F.normalize(queries.float(), dim=-1)
        if self._faiss_index is not None:
            import numpy as np

            scores, indices = self._faiss_index.search(queries.cpu().numpy().astype(np.float32), k)
            idx_tensor = torch.tensor(indices, dtype=torch.long)
            score_tensor = torch.tensor(scores, dtype=torch.float32)
            gathered = self.embeddings[idx_tensor]
            flat_ids = [self.doc_ids[idx] for idx in idx_tensor[:, 0].tolist()]
            return RetrievalResult(ids=flat_ids, embeddings=gathered[:, 0], scores=score_tensor[:, 0])

        embeddings = self.embeddings.to(queries.device)
        scores = queries @ embeddings.T
        top_scores, top_idx = torch.topk(scores, k=k, dim=-1)
        gathered = embeddings[top_idx]
        flat_ids = [self.doc_ids[idx] for idx in top_idx[:, 0].tolist()]
        return RetrievalResult(ids=flat_ids, embeddings=gathered[:, 0], scores=top_scores[:, 0])

    def search_text(self, query_text: str, queries: torch.Tensor, k: int = 1, lexical_weight: float = 0.7) -> RetrievalResult:
        if queries.ndim != 2 or queries.shape[0] != 1:
            raise ValueError(f"Expected a single query embedding with shape [1, D], got {tuple(queries.shape)}")

        embeddings = self.embeddings.to(queries.device)
        embedding_scores = (F.normalize(queries.float(), dim=-1) @ embeddings.T).squeeze(0)
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
        flat_ids = [self.doc_ids[idx] for idx in top_idx[:, 0].tolist()]
        return RetrievalResult(ids=flat_ids, embeddings=gathered[:, 0], scores=top_scores[:, 0])