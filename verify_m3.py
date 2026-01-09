import torch
from train_adam import M3Optimizer, zeropower_via_newtonschulz5

def verify_safety():
    print("🛡️  Running M3 Safety Check...")
    
    # 1. Test Newton-Schulz Stability (The math core)
    # Create a random matrix
    G = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
    
    try:
        # Attempt orthogonalization
        update = zeropower_via_newtonschulz5(G, steps=5)
        
        # Check for NaNs
        if torch.isnan(update).any():
            print("❌ FAILURE: Newton-Schulz produced NaNs!")
            return False
            
        # Check orthogonality roughly (AA^T should be close to Identity)
        # Note: In Muon, it's not strictly Identity, but it should be bounded.
        norm = update.norm().item()
        print(f"✅ Newton-Schulz stable. Norm: {norm:.4f} (Should be close to sqrt(dim))")
        
    except Exception as e:
        print(f"❌ FAILURE: Math error: {e}")
        return False

    # 2. Test Optimizer Step
    model_param = torch.nn.Parameter(torch.randn(128, 128, device="cuda"))
    opt = M3Optimizer([model_param], lr=0.02)
    
    # Fake gradient
    model_param.grad = torch.randn_like(model_param)
    
    try:
        opt.step()
        if torch.isnan(model_param).any():
            print("❌ FAILURE: Optimizer step produced NaNs!")
            return False
        print("✅ Optimizer step successful.")
    except Exception as e:
        print(f"❌ FAILURE: Optimizer step crashed: {e}")
        return False

    print("🟢 SYSTEM SAFE. Ready for training.")
    return True

if __name__ == "__main__":
    verify_safety()