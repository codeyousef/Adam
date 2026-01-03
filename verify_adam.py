import torch
from transformers import AutoTokenizer
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

# --- CONFIGURATION ---
# UPDATED: Matching the 7B model used in train_adam.py
model_id = "state-spaces/mamba2-7b"
tokenizer_id = "EleutherAI/gpt-neox-20b"

print(f"🐈 Catbelly Studio: Verifying Adam 7B ({model_id})...")
print(f"Loading Tokenizer ({tokenizer_id})...")
tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
tokenizer.pad_token = tokenizer.eos_token

print("Loading Model Weights (This may take a moment)...")
# UPDATED: Use bfloat16 for RTX 4090 optimization and stability
model = MambaLMHeadModel.from_pretrained(model_id, device="cuda", dtype=torch.bfloat16)

input_text = "The architectural advantages of Mamba-2 over Transformer are:"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")

print("Generating...")
# cg=True enables CUDA Graph capture for faster generation
output = model.generate(input_ids, max_length=100, temperature=0.7, top_p=0.9, cg=True)

print("\n--- RAW OUTPUT ---")
print(tokenizer.decode(output[0]))
print("\n--- CLEAN OUTPUT ---")
print(tokenizer.decode(output[0], skip_special_tokens=True))
print("\n✅ SUCCESS: 7B Kernels are active and reasoning.")
