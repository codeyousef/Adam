import torch
import torch.nn.functional as F
from src.models.jepa import JEPAModel
from src.training.phase4 import _proxy_config
from src.utils.data import SimpleTokenizer, make_text_code_pairs
from src.utils.retrieval import VectorIndex
import random

def run_test(size=2700000000, model_path="artifacts/ignorance_1_2.7b.pt", force_cpu=False, confidence_threshold=0.4, lexical_weight=0.7):
    device = "cpu" if force_cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    # 1. Load the Configuration and Model
    config = _proxy_config(size, "v6_overnight")
    # Use float32 on CPU for stability
    model = JEPAModel(config).to(device).eval()
    if not force_cpu:
        model = model.to(torch.bfloat16)
    
    print(f"Loading {size:,} Weights from {model_path} onto {device}...")
    try:
        sd = torch.load(model_path, map_location=device)
        # Cast weights to float32 if on CPU
        if force_cpu:
            sd = {k: v.float() for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        has_confidence_head = not any(key.startswith("query_head.") for key in missing)
        if missing:
            print(f"Warning: missing weights for {len(missing)} keys: {missing[:4]}")
        if unexpected:
            print(f"Warning: unexpected weights for {len(unexpected)} keys: {unexpected[:4]}")
    except FileNotFoundError:
        print(f"Error: {model_path} not found.")
        return

    tokenizer = SimpleTokenizer(vocab_size=4096)
    
    # 2. Build Knowledge Index (Parametric Memory)
    print("Indexing coding knowledge...")
    with torch.no_grad():
        # Use a higher repeat to get a more diverse set of potential matches
        code_pairs = make_text_code_pairs(repeats=10)
        code_snippets = list(set([p[1] for p in code_pairs]))
        print(f"Index size: {len(code_snippets)} unique snippets")
        
        code_tensors = tokenizer.batch_encode(code_snippets, config.max_seq_len, device)
        z_code = model.encode(code_tensors)
        
        # DEBUG: Check code latent variety
        z_code_norm = F.normalize(z_code.float(), dim=-1)
        sim_matrix = z_code_norm @ z_code_norm.T
        avg_sim = (sim_matrix.sum() - len(code_snippets)) / (len(code_snippets) * (len(code_snippets) - 1))
        print(f"DEBUG Code Latent Avg Similarity: {avg_sim.item():.4f}")
        
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

    print("\n" + "="*80)
    print(f"{'TYPE':<20} | {'QUERY':<40} | {'SIM':<6} | {'STATUS'}")
    print("-" * 80)

    results = []
    for test_type, query in test_cases:
        with torch.no_grad():
            # Encode Query
            q_tensor = tokenizer.batch_encode([query], config.max_seq_len, device)
            # print(f"DEBUG: {query[:20]} -> {q_tensor[0][:5]}")
            z_query = model.encode(q_tensor)
            
            # DEBUG: Check latent variety
            print(f"DEBUG Latent ({query[:10]}): {z_query[0, :3].float().cpu().numpy()} norm={torch.norm(z_query).item():.2f}")
            
            # Predict "Target Knowledge" Latent
            z_pred = model.predict(z_query, action_id=1) # action_id 1 = code prediction
            confidence = model.query_confidence(z_query).item() if has_confidence_head else 1.0
            
            # Search Index
            if confidence < confidence_threshold:
                similarity = 0.0
                retrieved = "<IGNORANT>"
            else:
                result = index.search_text(query, z_pred.cpu(), k=1, lexical_weight=lexical_weight)
                similarity = result.scores[0].item()
                retrieved = result.ids[0]
            
            status = "✅ KNOWN" if similarity > 0.85 else "⚠️ IGNORANT"
            if "OOD" in test_type or "Edge" in test_type:
                status = "✅ CORRECTLY IGNORANT" if similarity < 0.75 else "❌ FALSE POSITIVE"
            
            print(f"{test_type:<20} | {query[:40]:<40} | {similarity:.4f} | {status}")
            results.append({
                "type": test_type,
                "query": query,
                "similarity": similarity,
                "status": status,
                "confidence": confidence,
                "retrieved": retrieved,
            })

    print("="*80 + "\n")
    
    # Calculate Summary Statistics
    known_sims = [r['similarity'] for r in results if 'Known' in r['type']]
    ood_sims = [r['similarity'] for r in results if 'OOD' in r['type'] or 'Edge' in r['type']]
    
    avg_known = sum(known_sims) / len(known_sims) if known_sims else 0
    avg_ood = sum(ood_sims) / len(ood_sims) if ood_sims else 0
    gap = avg_known - avg_ood
    
    print("SUMMARY METRICS:")
    print(f" - Average Known Similarity: {avg_known:.4f}")
    print(f" - Average Ignorant Similarity: {avg_ood:.4f}")
    print(f" - Ignorance Gap: {gap:.4f}")
    
    # Check Latent Cohesion (Similar prompts should have high similarity to each other)
    print("\nLATENT COHESION TEST:")
    with torch.no_grad():
        p1 = tokenizer.batch_encode(["Sort a list of numbers."], config.max_seq_len, device)
        p2 = tokenizer.batch_encode(["Order an array of integers."], config.max_seq_len, device)
        z1 = F.normalize(model.encode(p1), dim=-1)
        z2 = F.normalize(model.encode(p2), dim=-1)
        cohesion = (z1 @ z2.T).item()
        print(f" - Similarity between paraphrased prompts: {cohesion:.4f}")
    
    status_final = "✅ PASS" if gap > 0.15 and cohesion > 0.8 else "❌ FAIL"
    print(f" - Production Readiness Status: {status_final}")
    print("\n" + "="*80)
    
    # Detailed look at a few retrievals
    print("DETAILED RETRIEVALS:")
    for res in results[:4]: # Show first 4
        print(f"Query: {res['query']}")
        print(f"Retrieved: \n{res['retrieved']}")
        print("-" * 40)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("size", nargs="?", type=int, default=2700000000)
    parser.add_argument("model_path", nargs="?", default="artifacts/ignorance_1_2.7b.pt")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--confidence-threshold", type=float, default=0.4)
    parser.add_argument("--lexical-weight", type=float, default=0.7)
    args = parser.parse_args()

    run_test(
        args.size,
        args.model_path,
        force_cpu=args.cpu,
        confidence_threshold=args.confidence_threshold,
        lexical_weight=args.lexical_weight,
    )
