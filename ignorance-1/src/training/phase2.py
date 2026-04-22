from __future__ import annotations

import torch
import torch.nn.functional as F

from src.models.jepa import JEPAConfig, JEPAModel
from src.utils.data import BenchmarkTokenizer, coding_facts
from src.utils.retrieval import VectorIndex


def _phase2_log(message: str) -> None:
    print(message, flush=True)


def _progress_points(total_steps: int) -> set[int]:
    if total_steps <= 1:
        return {0}
    fractions = [0.25, 0.5, 0.75, 1.0]
    return {min(total_steps - 1, max(0, int(total_steps * fraction) - 1)) for fraction in fractions}


def _build_index(model: JEPAModel, tokenizer: BenchmarkTokenizer, facts, seq_len: int, device: str) -> VectorIndex:
    doc_ids = []
    embeddings = []
    with torch.no_grad():
        for fact in facts:
            doc_ids.extend(fact.solution_docs)
            doc_tensor = tokenizer.batch_encode([fact.doc], seq_len, device)
            embeddings.append(model.encode(doc_tensor).squeeze(0).cpu())
    return VectorIndex(doc_ids, torch.stack(embeddings, dim=0))


def run_phase2(config, phase1_result: dict, device: str) -> dict:
    seq_len = 128
    vocab_size = 4096
    tokenizer = BenchmarkTokenizer(vocab_size=vocab_size)
    model = JEPAModel(
        JEPAConfig(
            vocab_size=vocab_size,
            patch_size=32,
            max_seq_len=seq_len,
            embed_dim=192,
            encoder_layers=4,
            encoder_heads=3,
            predictor_layers=4,
            predictor_heads=6,
        )
    ).to(device)
    facts = coding_facts()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    progress_points = _progress_points(config.epochs)

    _phase2_log(
        f"[phase2] epochs={config.epochs} lr={config.lr} facts={len(facts)} batch_size={config.batch_size}"
    )

    answer_ids = tokenizer.batch_encode([fact.answer for fact in facts], seq_len, device)
    question_ids = tokenizer.batch_encode([fact.question for fact in facts], seq_len, device)
    doc_ids = tokenizer.batch_encode([fact.doc for fact in facts], seq_len, device)

    for epoch in range(config.epochs):
        z_question = model.encode(question_ids)
        z_answer = model.encode(answer_ids)
        z_doc = model.encode(doc_ids)
        z_pred_with = model.predict(z_question, action_embed=z_doc, action_id=2)
        z_pred_without = model.predict(z_question, action_embed=None, action_id=0)

        retrieve_loss = F.mse_loss(z_pred_with, z_answer)
        direct_alignment = F.cosine_similarity(z_pred_without, z_answer, dim=-1).mean()
        loss = retrieve_loss + config.direct_penalty * direct_alignment
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if epoch in progress_points:
            _phase2_log(
                f"[phase2] epoch={epoch + 1}/{config.epochs} loss={float(loss.detach().cpu().item()):.4f}"
            )

    index = _build_index(model, tokenizer, facts, seq_len, device)

    def evaluate(use_retrieval: bool) -> float:
        correct = 0
        with torch.no_grad():
            z_question = model.encode(question_ids)
            z_answer = model.encode(answer_ids)
            if use_retrieval:
                query = model.predictor.generate_query(z_question)
                retrieved = index.search(query, k=config.retrieval_k)
                z_pred = model.predict(z_question, action_embed=retrieved.embeddings.to(device), action_id=2)
            else:
                z_pred = model.predict(z_question, action_embed=None, action_id=0)
            similarity = F.cosine_similarity(z_pred, z_answer, dim=-1)
            correct = int((similarity > (1.0 - config.answer_threshold)).sum().item())
        return correct / len(facts)

    without = evaluate(False)
    with_retrieval = evaluate(True)
    gap = with_retrieval - without
    _phase2_log(
        f"[phase2] done without={without:.3f} with={with_retrieval:.3f} gap={gap:.3f}"
    )
    return {
        "accuracy_without_retrieval": without,
        "accuracy_with_retrieval": with_retrieval,
        "retrieval_gap": gap,
        "passes_ignorance_test": without < 0.15 and with_retrieval > 0.75 and gap > 0.50,
        "backend": index.backend,
    }