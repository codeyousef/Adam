import argparse
import json
import re

import torch
import torch.nn.functional as F

from src.models.jepa import JEPAModel
from src.training.phase4 import _proxy_config
from src.utils.data import SimpleTokenizer, make_text_code_pairs
from src.utils.retrieval import VectorIndex


def _log(verbose: bool, message: str) -> None:
    if verbose:
        print(message)


def _pairwise_similarity_stats(z: torch.Tensor) -> dict[str, float]:
    if z.ndim == 3:
        z = z.reshape(-1, z.shape[-1])
    if z.shape[0] < 2:
        return {
            "avg_offdiag_similarity": 0.0,
            "max_offdiag_similarity": 0.0,
            "std_offdiag_similarity": 0.0,
            "participation_ratio": 0.0,
            "participation_ratio_fraction": 0.0,
        }

    normalized = F.normalize(z.float(), dim=-1)
    similarity = normalized @ normalized.T
    mask = ~torch.eye(similarity.shape[0], dtype=torch.bool, device=similarity.device)
    offdiag = similarity[mask]
    centered = normalized - normalized.mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(centered)
    energy = singular_values.square()
    participation_ratio = (energy.sum().square() / energy.square().sum().clamp_min(1e-6)).item() if energy.numel() else 0.0
    return {
        "avg_offdiag_similarity": float(offdiag.mean().item()),
        "max_offdiag_similarity": float(offdiag.max().item()),
        "std_offdiag_similarity": float(offdiag.std(unbiased=False).item()),
        "participation_ratio": float(participation_ratio),
        "participation_ratio_fraction": float(participation_ratio / max(normalized.shape[-1], 1)),
    }


def _combined_scores(index: VectorIndex, query_text: str, query_embedding: torch.Tensor, lexical_weight: float) -> torch.Tensor:
    embeddings = index.embeddings.to(query_embedding.device)
    embedding_scores = (F.normalize(query_embedding.float(), dim=-1) @ embeddings.T).squeeze(0)
    query_tokens = set(re.findall(r"[a-z0-9_]+", query_text.lower()))
    lexical_scores = []
    for doc_tokens in index.doc_tokens:
        if not query_tokens or not doc_tokens:
            lexical_scores.append(0.0)
            continue
        overlap = len(query_tokens & doc_tokens)
        lexical_scores.append(overlap / max(len(query_tokens | doc_tokens), 1))
    lexical_tensor = torch.tensor(lexical_scores, device=query_embedding.device, dtype=embedding_scores.dtype)
    return (1.0 - lexical_weight) * embedding_scores + lexical_weight * lexical_tensor


def _infer_max_seq_len(state_dict: dict, config) -> int:
    pos_embed = state_dict.get("encoder.pos_embed")
    if pos_embed is None or pos_embed.ndim != 3:
        return config.max_seq_len
    token_slots = int(pos_embed.shape[1]) - 1
    if token_slots <= 0:
        return config.max_seq_len
    return max(token_slots * config.patch_size, config.patch_size)


def _strict_status(summary: dict) -> tuple[str, list[str]]:
    failures: list[str] = []
    if not summary["has_confidence_head"]:
        failures.append("checkpoint is missing the confidence head")
    if summary["avg_known_exact_similarity"] < 0.65:
        failures.append(f"known exact similarity too low ({summary['avg_known_exact_similarity']:.4f} < 0.65)")
    if summary["avg_known_paraphrase_similarity"] < 0.50:
        failures.append(f"known paraphrase similarity too low ({summary['avg_known_paraphrase_similarity']:.4f} < 0.50)")
    if summary["synthesis_similarity"] < 0.45:
        failures.append(f"synthesis similarity too low ({summary['synthesis_similarity']:.4f} < 0.45)")
    if summary["avg_ignorant_similarity"] > 0.25:
        failures.append(f"ignorant similarity too high ({summary['avg_ignorant_similarity']:.4f} > 0.25)")
    if summary["ignorance_gap"] < 0.30:
        failures.append(f"ignorance gap too small ({summary['ignorance_gap']:.4f} < 0.30)")
    if summary["avg_known_margin"] < 0.05:
        failures.append(f"retrieval margin too small ({summary['avg_known_margin']:.4f} < 0.05)")
    if summary["code_diagnostics"]["avg_offdiag_similarity"] > 0.85:
        failures.append(
            "code embeddings are too similar "
            f"({summary['code_diagnostics']['avg_offdiag_similarity']:.4f} > 0.85)"
        )
    if summary["query_diagnostics"]["avg_offdiag_similarity"] > 0.85:
        failures.append(
            "query embeddings are too similar "
            f"({summary['query_diagnostics']['avg_offdiag_similarity']:.4f} > 0.85)"
        )
    if summary["code_diagnostics"]["participation_ratio_fraction"] < 0.10:
        failures.append(
            "code embedding effective rank is too low "
            f"({summary['code_diagnostics']['participation_ratio_fraction']:.4f} < 0.10)"
        )
    if summary["query_diagnostics"]["participation_ratio_fraction"] < 0.10:
        failures.append(
            "query embedding effective rank is too low "
            f"({summary['query_diagnostics']['participation_ratio_fraction']:.4f} < 0.10)"
        )
    if summary["has_confidence_head"] and summary["avg_known_confidence"] < 0.60:
        failures.append(f"known confidence too low ({summary['avg_known_confidence']:.4f} < 0.60)")
    if summary["has_confidence_head"] and summary["avg_ood_confidence"] > 0.35:
        failures.append(f"OOD confidence too high ({summary['avg_ood_confidence']:.4f} > 0.35)")
    return ("✅ PASS" if not failures else "❌ FAIL"), failures


def run_test(
    size=2700000000,
    model_path="artifacts/ignorance_1_2.7b.pt",
    force_cpu=False,
    confidence_threshold=0.4,
    lexical_weight=0.7,
    verbose=True,
    embed_dim_override=0,
    encoder_layers_override=0,
    encoder_heads_override=0,
    predictor_layers_override=0,
    predictor_heads_override=0,
    decoder_layers_override=0,
    decoder_heads_override=0,
    decoder_hidden_dim_override=0,
):
    device = "cpu" if force_cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    config = _proxy_config(size, "v6_overnight")
    if embed_dim_override > 0:
        config.embed_dim = embed_dim_override
    if encoder_layers_override > 0:
        config.encoder_layers = encoder_layers_override
    if encoder_heads_override > 0:
        config.encoder_heads = encoder_heads_override
    if predictor_layers_override > 0:
        config.predictor_layers = predictor_layers_override
    if predictor_heads_override > 0:
        config.predictor_heads = predictor_heads_override
    if decoder_layers_override > 0:
        config.decoder_layers = decoder_layers_override
    if decoder_heads_override > 0:
        config.decoder_heads = decoder_heads_override
    if decoder_hidden_dim_override > 0:
        config.decoder_hidden_dim = decoder_hidden_dim_override

    _log(verbose, f"Loading {size:,} Weights from {model_path} onto {device}...")
    try:
        sd = torch.load(model_path, map_location=device)
        config.max_seq_len = _infer_max_seq_len(sd, config)
        model = JEPAModel(config).to(device).eval()
        if force_cpu:
            sd = {k: v.float() for k, v in sd.items()}
        else:
            model = model.to(torch.bfloat16)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        has_confidence_head = not any(key.startswith("query_head.") for key in missing)
        if missing:
            _log(verbose, f"Warning: missing weights for {len(missing)} keys: {missing[:4]}")
        if unexpected:
            _log(verbose, f"Warning: unexpected weights for {len(unexpected)} keys: {unexpected[:4]}")
    except FileNotFoundError:
        message = f"Error: {model_path} not found."
        _log(verbose, message)
        return {"error": message}

    tokenizer = SimpleTokenizer(vocab_size=4096)
    
    _log(verbose, "Indexing coding knowledge...")
    with torch.no_grad():
        code_pairs = make_text_code_pairs(repeats=10)
        code_snippets = list(set([p[1] for p in code_pairs]))
        _log(verbose, f"Index size: {len(code_snippets)} unique snippets")
        
        code_tensors = tokenizer.batch_encode(code_snippets, config.max_seq_len, device)
        z_code = model.encode(code_tensors)
        code_diagnostics = _pairwise_similarity_stats(z_code)
        _log(verbose, f"DEBUG Code Latent Avg Similarity: {code_diagnostics['avg_offdiag_similarity']:.4f}")
        
        index = VectorIndex(code_snippets, z_code.cpu())

    test_cases = [
        ("Known - Exact", "Sort a numeric list ascending and return the result."),
        ("Known - Paraphrase", "How can I order an array of integers from smallest to largest?"),
        ("Known - Exact", "Read each line from a text file and strip whitespace."),
        ("Known - Paraphrase", "I want to load a file and trim every line in it."),
        ("Known - Exact", "Parse a json string into a javascript object."),
        ("Known - Paraphrase", "Convert this JSON text into a JS variable."),
        ("OOD - Weather", "What is the weather in Tokyo today?"),
        ("OOD - History", "Who was the first president of the United States?"),
        ("OOD - Random", "The quick brown fox jumps over the lazy dog."),
        ("Edge - Empty", ""),
        ("Edge - Gibberish", "asdfghjkl;qwertyuiop"),
        ("Synthesis", "Read a file, parse the JSON in it, and sort the result."),
    ]

    _log(verbose, "\n" + "="*80)
    _log(verbose, f"{'TYPE':<20} | {'QUERY':<40} | {'SIM':<6} | {'STATUS'}")
    _log(verbose, "-" * 80)

    results = []
    query_vectors = []
    pred_vectors = []
    for test_type, query in test_cases:
        with torch.no_grad():
            q_tensor = tokenizer.batch_encode([query], config.max_seq_len, device)
            z_query = model.encode(q_tensor)
            _log(verbose, f"DEBUG Latent ({query[:10]}): {z_query[0, :3].float().cpu().numpy()} norm={torch.norm(z_query).item():.2f}")

            z_pred = model.predict(z_query, action_id=1) # action_id 1 = code prediction
            confidence = model.query_confidence(z_query).item() if has_confidence_head else 1.0
            query_vectors.append(F.normalize(z_query.float(), dim=-1).cpu())
            pred_vectors.append(F.normalize(z_pred.float(), dim=-1).cpu())
            
            if confidence < confidence_threshold:
                similarity = 0.0
                retrieved = "<IGNORANT>"
                retrieval_margin = 0.0
            else:
                scores = _combined_scores(index, query, z_pred.cpu(), lexical_weight)
                top_k = min(2, scores.shape[0])
                top_scores, top_idx = torch.topk(scores, k=top_k)
                similarity = top_scores[0].item()
                retrieved = index.doc_ids[top_idx[0].item()]
                retrieval_margin = (top_scores[0] - top_scores[1]).item() if top_k > 1 else top_scores[0].item()
            
            status = "✅ KNOWN" if similarity > 0.85 else "⚠️ IGNORANT"
            if "OOD" in test_type or "Edge" in test_type:
                status = "✅ CORRECTLY IGNORANT" if similarity < 0.75 else "❌ FALSE POSITIVE"
            
            _log(verbose, f"{test_type:<20} | {query[:40]:<40} | {similarity:.4f} | {status}")
            results.append({
                "type": test_type,
                "query": query,
                "similarity": similarity,
                "status": status,
                "confidence": confidence,
                "retrieval_margin": retrieval_margin,
                "retrieved": retrieved,
            })

    _log(verbose, "="*80 + "\n")
    
    known_sims = [r['similarity'] for r in results if 'Known' in r['type']]
    known_exact_sims = [r['similarity'] for r in results if r['type'] == 'Known - Exact']
    known_paraphrase_sims = [r['similarity'] for r in results if r['type'] == 'Known - Paraphrase']
    ood_sims = [r['similarity'] for r in results if 'OOD' in r['type'] or 'Edge' in r['type']]
    synthesis_sims = [r['similarity'] for r in results if 'Synthesis' in r['type']]
    known_confidences = [r['confidence'] for r in results if 'Known' in r['type']]
    ood_confidences = [r['confidence'] for r in results if 'OOD' in r['type'] or 'Edge' in r['type']]
    known_margins = [r['retrieval_margin'] for r in results if 'Known' in r['type']]
    
    avg_known = sum(known_sims) / len(known_sims) if known_sims else 0
    avg_ood = sum(ood_sims) / len(ood_sims) if ood_sims else 0
    gap = avg_known - avg_ood
    avg_known_exact = sum(known_exact_sims) / len(known_exact_sims) if known_exact_sims else 0
    avg_known_paraphrase = sum(known_paraphrase_sims) / len(known_paraphrase_sims) if known_paraphrase_sims else 0
    avg_known_confidence = sum(known_confidences) / len(known_confidences) if known_confidences else 0
    avg_ood_confidence = sum(ood_confidences) / len(ood_confidences) if ood_confidences else 0
    avg_known_margin = sum(known_margins) / len(known_margins) if known_margins else 0
    synthesis_similarity = synthesis_sims[0] if synthesis_sims else 0.0
    
    _log(verbose, "SUMMARY METRICS:")
    _log(verbose, f" - Average Known Similarity: {avg_known:.4f}")
    _log(verbose, f" - Average Known Exact Similarity: {avg_known_exact:.4f}")
    _log(verbose, f" - Average Known Paraphrase Similarity: {avg_known_paraphrase:.4f}")
    _log(verbose, f" - Average Ignorant Similarity: {avg_ood:.4f}")
    _log(verbose, f" - Ignorance Gap: {gap:.4f}")
    _log(verbose, f" - Average Known Retrieval Margin: {avg_known_margin:.4f}")
    _log(verbose, f" - Synthesis Similarity: {synthesis_similarity:.4f}")
    if has_confidence_head:
        _log(verbose, f" - Average Known Confidence: {avg_known_confidence:.4f}")
        _log(verbose, f" - Average Ignorant Confidence: {avg_ood_confidence:.4f}")
    
    _log(verbose, "\nLATENT COHESION TEST:")
    with torch.no_grad():
        p1 = tokenizer.batch_encode(["Sort a list of numbers."], config.max_seq_len, device)
        p2 = tokenizer.batch_encode(["Order an array of integers."], config.max_seq_len, device)
        z1 = F.normalize(model.encode(p1), dim=-1)
        z2 = F.normalize(model.encode(p2), dim=-1)
        cohesion = (z1 @ z2.T).item()
        _log(verbose, f" - Similarity between paraphrased prompts: {cohesion:.4f}")

    query_diagnostics = _pairwise_similarity_stats(torch.cat(query_vectors, dim=0))
    pred_diagnostics = _pairwise_similarity_stats(torch.cat(pred_vectors, dim=0))
    _log(verbose, "\nDIAGNOSTICS:")
    _log(verbose, f" - Code avg offdiag similarity: {code_diagnostics['avg_offdiag_similarity']:.4f}")
    _log(verbose, f" - Code effective-rank fraction: {code_diagnostics['participation_ratio_fraction']:.4f}")
    _log(verbose, f" - Query avg offdiag similarity: {query_diagnostics['avg_offdiag_similarity']:.4f}")
    _log(verbose, f" - Query effective-rank fraction: {query_diagnostics['participation_ratio_fraction']:.4f}")
    _log(verbose, f" - Pred avg offdiag similarity: {pred_diagnostics['avg_offdiag_similarity']:.4f}")
    _log(verbose, f" - Pred effective-rank fraction: {pred_diagnostics['participation_ratio_fraction']:.4f}")
    
    legacy_status = "✅ PASS" if gap > 0.15 and cohesion > 0.8 else "❌ FAIL"
    summary = {
        "model_path": model_path,
        "size": size,
        "device": device,
        "has_confidence_head": has_confidence_head,
        "avg_known_similarity": avg_known,
        "avg_known_exact_similarity": avg_known_exact,
        "avg_known_paraphrase_similarity": avg_known_paraphrase,
        "avg_ignorant_similarity": avg_ood,
        "ignorance_gap": gap,
        "avg_known_margin": avg_known_margin,
        "synthesis_similarity": synthesis_similarity,
        "avg_known_confidence": avg_known_confidence,
        "avg_ood_confidence": avg_ood_confidence,
        "cohesion": cohesion,
        "legacy_status": legacy_status,
        "code_diagnostics": code_diagnostics,
        "query_diagnostics": query_diagnostics,
        "pred_diagnostics": pred_diagnostics,
        "results": results,
    }
    strict_status, strict_failures = _strict_status(summary)
    summary["strict_status"] = strict_status
    summary["strict_failures"] = strict_failures
    _log(verbose, f" - Legacy Production Readiness Status: {legacy_status}")
    _log(verbose, f" - Strict Production Readiness Status: {strict_status}")
    for failure in strict_failures:
        _log(verbose, f"   - {failure}")
    _log(verbose, "\n" + "="*80)
    
    _log(verbose, "DETAILED RETRIEVALS:")
    for res in results[:4]:
        _log(verbose, f"Query: {res['query']}")
        _log(verbose, f"Retrieved: \n{res['retrieved']}")
        _log(verbose, "-" * 40)
    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("size", nargs="?", type=int, default=2700000000)
    parser.add_argument("model_path", nargs="?", default="artifacts/ignorance_1_2.7b.pt")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--confidence-threshold", type=float, default=0.4)
    parser.add_argument("--lexical-weight", type=float, default=0.7)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--embed-dim", type=int, default=0)
    parser.add_argument("--encoder-layers", type=int, default=0)
    parser.add_argument("--encoder-heads", type=int, default=0)
    parser.add_argument("--predictor-layers", type=int, default=0)
    parser.add_argument("--predictor-heads", type=int, default=0)
    parser.add_argument("--decoder-layers", type=int, default=0)
    parser.add_argument("--decoder-heads", type=int, default=0)
    parser.add_argument("--decoder-hidden-dim", type=int, default=0)
    args = parser.parse_args()

    summary = run_test(
        args.size,
        args.model_path,
        force_cpu=args.cpu,
        confidence_threshold=args.confidence_threshold,
        lexical_weight=args.lexical_weight,
        embed_dim_override=args.embed_dim,
        encoder_layers_override=args.encoder_layers,
        encoder_heads_override=args.encoder_heads,
        predictor_layers_override=args.predictor_layers,
        predictor_heads_override=args.predictor_heads,
        decoder_layers_override=args.decoder_layers,
        decoder_heads_override=args.decoder_heads,
        decoder_hidden_dim_override=args.decoder_hidden_dim,
    )
    if args.json:
        print(json.dumps(summary, indent=2))

