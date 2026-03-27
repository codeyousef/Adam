from __future__ import annotations
import torch
import torch.nn.functional as F
import sys
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.jepa import JEPAConfig, JEPAModel
from src.utils.data import SimpleTokenizer, coding_facts
from src.utils.retrieval import VectorIndex

def train_demo_model(device: str, save_path: str = "demo_weights.pt"):
    seq_len = 128
    vocab_size = 4096
    tokenizer = SimpleTokenizer(vocab_size=vocab_size)
    
    # Small fast model
    config = JEPAConfig(
        vocab_size=vocab_size,
        patch_size=32,
        max_seq_len=seq_len,
        embed_dim=192,
        encoder_layers=4,
        encoder_heads=3,
        predictor_layers=4,
        predictor_heads=6,
    )
    model = JEPAModel(config).to(device)
    facts = coding_facts()

    if Path(save_path).exists():
        print(f"Loading existing weights from {save_path}...")
        model.load_state_dict(torch.load(save_path, map_location=device))
        model.eval()
    else:
        print("Initializing new demo model...")
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        print(f"Training on {len(facts)} coding facts for 100 epochs...")
        
        # Pre-encode labels
        answer_ids = tokenizer.batch_encode([f.answer for f in facts], seq_len, device)
        question_ids = tokenizer.batch_encode([f.question for f in facts], seq_len, device)
        doc_ids = tokenizer.batch_encode([f.doc for f in facts], seq_len, device)

        model.train()
        for epoch in range(100):
            z_question = model.encode(question_ids)
            z_answer = model.encode(answer_ids)
            z_doc = model.encode(doc_ids)
            
            z_pred = model.predict(z_question, action_embed=z_doc, action_id=2)
            loss = F.mse_loss(z_pred, z_answer)
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/100 - Loss: {loss.item():.6f}")
        
        print(f"Saving weights to {save_path}...")
        torch.save(model.state_dict(), save_path)

    # Build retrieval index
    print("Building vector index...")
    model.eval()
    doc_embeddings = []
    with torch.no_grad():
        for f in facts:
            d_tensor = tokenizer.batch_encode([f.doc], seq_len, device)
            doc_embeddings.append(model.encode(d_tensor).squeeze(0).cpu())
    
    index = VectorIndex([f.question for f in facts], torch.stack(doc_embeddings, dim=0))
    # We use fact.question as ID here just to show which fact was matched
    
    return model, tokenizer, index, facts

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, index, facts = train_demo_model(device)
    
    print("\n" + "="*50)
    print("JEPA DEMO READY")
    print("The model has learned the following topics:")
    for f in facts:
        print(f" - {f.question[:40]}...")
    print("="*50)
    
    print("\nTry asking a question about Python sorting, JSON, Dataclasses, etc.")
    print("(Type 'exit' to quit)\n")
    
    while True:
        try:
            query_text = input("User> ").strip()
            if not query_text or query_text.lower() in ['exit', 'quit']:
                break
            
            with torch.no_grad():
                # 1. Encode query
                q_tensor = tokenizer.batch_encode([query_text], 128, device)
                z_q = model.encode(q_tensor)
                
                # 2. Generate JEPA retrieval query
                # In JEPA, the predictor generates a 'target' latent which we use for retrieval
                retrieval_query = model.predictor.generate_query(z_q)
                
                # 3. Search index
                results = index.search(retrieval_query.cpu(), k=1)
                match_question = results.ids[0]
                
                # Find the full fact
                matched_fact = next(f for f in facts if f.question == match_question)
                
                print(f"Model Retrieval Match: {matched_fact.question}")
                print(f"Model Answer: {matched_fact.answer}")
                print("-" * 30)
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
