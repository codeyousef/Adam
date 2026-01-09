import torch
import glob
import os
import readline # specific for input handling
from transformers import AutoTokenizer
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

# --- CONFIG ---
CHECKPOINT_DIR = "/data/adam_checkpoints"
MODEL_NAME = "state-spaces/mamba2-2.7b"
TOKENIZER_ID = "EleutherAI/gpt-neox-20b"
SPECIFIC_FILE = "adam_ckpt_13376.pt" # <--- We are targeting this one

# --- SETUP ---
print(f"🐈 Waking Adam (Logic Engine)...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
tokenizer.pad_token = tokenizer.eos_token

# Find the checkpoint
ckpt_path = f"{CHECKPOINT_DIR}/{SPECIFIC_FILE}"
if not os.path.exists(ckpt_path):
    print(f"⚠️  Specific file {SPECIFIC_FILE} not found. Searching for others...")
    ckpts = sorted(glob.glob(f"{CHECKPOINT_DIR}/adam_ckpt_*.pt"), key=os.path.getmtime)
    if not ckpts:
        print("❌ No checkpoints found anywhere!")
        exit()
    ckpt_path = ckpts[-1]

print(f"🧠 Loading Brain: {ckpt_path}")

# Load Model
model = MambaLMHeadModel.from_pretrained(MODEL_NAME, dtype=torch.bfloat16).to("cuda")
checkpoint = torch.load(ckpt_path, map_location="cuda")

# Handle slight format differences in saving
if "model" in checkpoint:
    model.load_state_dict(checkpoint["model"])
else:
    model.load_state_dict(checkpoint)

model.eval()
print("✅ Adam is Online. (Type 'quit' to exit)")

# --- CHAT LOOP ---
while True:
    try:
        user_input = input("\n👤 You: ")
        if user_input.lower() in ["quit", "exit"]: break
        
        # We wrap it in the exact Logic Probe format to trigger his training
        prompt = f"Question: {user_input}\nReasoning:\n"
        
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        input_ids = inputs.input_ids
        
        # CALCULATE MAX LENGTH (Crucial for Mamba)
        # We want to generate up to 500 NEW tokens
        max_length = input_ids.shape[1] + 500
        
        with torch.no_grad():
            output = model.generate(
                input_ids,             # Positional 1
                max_length,            # Positional 2 (Total length)
                temperature=0.7,       # Creative but focused
                top_p=0.9, 
                repetition_penalty=1.1,
                cg=True                # Fast mode
            )
        
        response = tokenizer.decode(output[0])
        
        # Clean up: Remove the prompt and stop at endoftext
        answer = response.replace(prompt, "")
        if "<|endoftext|>" in answer:
            answer = answer.split("<|endoftext|>")[0]
            
        print(f"🤖 Adam:\n{answer}")
        
    except KeyboardInterrupt:
        print("\n👋 Goodbye.")
        break
    except Exception as e:
        print(f"❌ Error: {e}")