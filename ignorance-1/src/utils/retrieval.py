from __future__ import annotations

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