import os
import time
import json
import torch
import glob
import warnings
import csv
import gc
import traceback
import subprocess
import math
from datetime import datetime
from torch.utils.data import IterableDataset, DataLoader
from transformers import AutoTokenizer
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from huggingface_hub import HfApi

# --- SILENCE WARNINGS ---
warnings.filterwarnings("ignore", message=".*set_float32_matmul_precision.*")

# --- USER CONFIGURATION (EDIT THIS) ---
HF_REPO_ID = "YOUR_USERNAME/adam-mamba-2.7b-logic"  # <--- CREATE THIS REPO ON HF FIRST!
HF_TOKEN = "hf_..."                                   # <--- PASTE YOUR WRITE TOKEN HERE

# --- RESEARCH CONFIG ---
MODEL_NAME = "state-spaces/mamba2-2.7b"
TOKENIZER_ID = "EleutherAI/gpt-neox-20b"
DATA_FILE = "adam_skeleton_data.jsonl"
CHECKPOINT_DIR = "adam_checkpoints"
TELEMETRY_FILE = "adam_research_metrics.csv"
SNAPSHOT_FILE = "adam_logic_snapshots.txt"

# --- HYPERPARAMETERS ---
SAVE_EVERY_MINS = 60
GRAD_ACCUM = 4
BATCH_SIZE = 8
LEARNING_RATE = 0.002
MAX_SEQ_LEN = 1536
THERMAL_THRESHOLD = 80
KEEP_LOCAL_CKPTS = 1

# --- REASONING PROBES (For Logic Snapshots) ---
PROBES = [
    "def solve_logic_puzzle(x):",
    "If all A are B, and some B are C, then:",
    "To calculate the trajectory of a rocket, one must first:",
    "System: You are a coding assistant. User: Write a Python function to parse JSON."
]

# --- SETUP HF API ---
api = HfApi(token=HF_TOKEN)

def get_gpu_metrics():
    try:
        # Returns Temp, Power(W), Memory(MiB)
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=temperature.gpu,power.draw,memory.used', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        data = result.stdout.strip().split(', ')
        return int(data[0]), float(data[1]), int(data[2])
    except:
        return 0, 0.0, 0

def thermal_check():
    temp, _, _ = get_gpu_metrics()
    if temp > THERMAL_THRESHOLD:
        print(f"🌡️  GPU at {temp}°C - pausing for cooldown...")
        while True:
            t, _, _ = get_gpu_metrics()
            if t < (THERMAL_THRESHOLD - 10): break
            time.sleep(30)
        print(f"✅ GPU cooled to {t}°C - resuming")

# --- METRICS LOGGER ---
def init_metrics():
    if not os.path.exists(TELEMETRY_FILE):
        with open(TELEMETRY_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Step", "Loss", "Learning_Rate", "VRAM_Used_MB", "GPU_Temp_C", "Tokens_Per_Sec"])

def log_metrics(step, loss, lr, tokens_sec):
    temp, power, vram = get_gpu_metrics()
    with open(TELEMETRY_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now().isoformat(), step, f"{loss:.4f}", f"{lr:.6f}", vram, temp, f"{tokens_sec:.1f}"])

def save_logic_snapshot(model, tokenizer, step):
    """Generates text from probes to track reasoning ability qualitatively"""
    print(f"📸 Taking Logic Snapshot at step {step}...")
    model.eval()
    with open(SNAPSHOT_FILE, "a") as f:
        f.write(f"\n{'='*40}\nSNAPSHOT STEP {step} ({datetime.now().isoformat()})\n{'='*40}\n")
        
        for probe in PROBES:
            try:
                inputs = tokenizer(probe, return_tensors="pt").to("cuda")
                with torch.no_grad():
                    output = model.generate(
                        **inputs, 
                        max_new_tokens=64, 
                        temperature=0.7, 
                        top_p=0.9, 
                        do_sample=True
                    )
                decoded = tokenizer.decode(output[0])
                f.write(f"PROBE: {probe}\nRESULT: {decoded}\n{'-'*20}\n")
            except Exception as e:
                f.write(f"PROBE FAILED: {e}\n")
    model.train()

# --- M3 OPTIMIZER ---
def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= (X.norm() + eps)
    if G.size(0) > G.size(1): X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1): X = X.T
    return X.to(G.dtype)

class M3Optimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, slow_momentum=0.99, 
                 slow_freq=10, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, slow_momentum=slow_momentum, 
                        slow_freq=slow_freq, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            slow_momentum = group['slow_momentum']
            slow_freq = group['slow_freq']
            ns_steps = group['ns_steps']
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['fast_buffer'] = torch.zeros_like(p.data)
                    state['slow_buffer'] = torch.zeros_like(p.data)
                state['step'] += 1
                fast_buf = state['fast_buffer']
                slow_buf = state['slow_buffer']
                
                fast_buf.mul_(momentum).add_(grad)
                if state['step'] % slow_freq == 0:
                    slow_buf.mul_(slow_momentum).add_(fast_buf)
                
                update_tensor = fast_buf + (0.5 * slow_buf)
                
                if p.ndim == 2:
                    ortho_update = zeropower_via_newtonschulz5(update_tensor, steps=ns_steps)
                    p.data.add_(ortho_update, alpha=-lr)
                else:
                    p.data.add_(update_tensor, alpha=-lr * 0.1)

# --- ROBUST DATASET ---
class AdamDataset(IterableDataset):
    def __init__(self, filepath, tokenizer, max_len):
        self.filepath = filepath
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __iter__(self):
        with open(self.filepath, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    text = data.get("text", "")
                    if not text: continue
                    enc = self.tokenizer(
                        text, 
                        truncation=True, 
                        max_length=self.max_len, 
                        padding="max_length",
                        return_tensors="pt"
                    )
                    yield enc.input_ids.squeeze(0).clone().detach()
                except:
                    continue

# --- PHOENIX SAVER (Local + Cloud) ---
def upload_to_hf(file_path, step):
    print(f"☁️  Uploading step {step} to Hugging Face...")
    try:
        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=f"checkpoints/adam_ckpt_{step}.pt",
            repo_id=HF_REPO_ID,
            repo_type="model"
        )
        print("✅ Upload Complete!")
        
        # Also upload the metrics and logs
        api.upload_file(path_or_fileobj=TELEMETRY_FILE, path_in_repo=TELEMETRY_FILE, repo_id=HF_REPO_ID, repo_type="model")
        api.upload_file(path_or_fileobj=SNAPSHOT_FILE, path_in_repo=SNAPSHOT_FILE, repo_id=HF_REPO_ID, repo_type="model")
        
    except Exception as e:
        print(f"⚠️ Upload Failed: {e}")

def safe_save(model, optimizer, step):
    if not os.path.exists(CHECKPOINT_DIR): os.makedirs(CHECKPOINT_DIR)
    
    path = f"{CHECKPOINT_DIR}/adam_ckpt_{step}.pt"
    tmp_path = f"{CHECKPOINT_DIR}/tmp_saving.pt"
    
    try:
        # 1. Write to temp
        torch.save({
            "step": step, 
            "model": model.state_dict(), 
            "optimizer": optimizer.state_dict()
        }, tmp_path)
        
        # 2. Rename to final
        os.replace(tmp_path, path)
        print(f"💾 Saved locally: {path}")
        
        # 3. Upload to Cloud (Background-ish)
        upload_to_hf(path, step)
        
        # 4. Strict Cleanup (Keep 1)
        ckpts = sorted(glob.glob(f"{CHECKPOINT_DIR}/adam_ckpt_*.pt"), key=os.path.getmtime)
        while len(ckpts) > KEEP_LOCAL_CKPTS:
            oldest = ckpts.pop(0)
            if oldest != path:
                os.remove(oldest)
                print(f"🧹 Deleted local backup: {oldest}")
            
    except OSError as e:
        print(f"❌ Save failed (Disk Full?): {e}")
        if os.path.exists(tmp_path): os.remove(tmp_path)

def main():
    torch.set_float32_matmul_precision("high")
    init_metrics()
    
    print(f"🐈 Phoenix Protocol Initiated. Model: {MODEL_NAME}")
    print(f"📊 Logging metrics to {TELEMETRY_FILE}")
    
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = MambaLMHeadModel.from_pretrained(MODEL_NAME, dtype=torch.bfloat16).to("cuda")
    model.train()

    m3_params = [p for n, p in model.named_parameters() if p.requires_grad and p.ndim == 2]
    std_params = [p for n, p in model.named_parameters() if p.requires_grad and p.ndim < 2]
    optimizer = M3Optimizer([
        {"params": m3_params, "lr": LEARNING_RATE, "momentum": 0.95},
        {"params": std_params, "lr": 0.001, "momentum": 0.9}
    ], slow_freq=10)

    # Resume Checkpoint (Local first, then could theoretically pull from HF but keeping simple)
    start_step = 0
    ckpts = sorted(glob.glob(f"{CHECKPOINT_DIR}/adam_ckpt_*.pt"), key=os.path.getmtime)
    if ckpts:
        try:
            print(f"📂 Found local checkpoint: {ckpts[-1]}")
            ckpt = torch.load(ckpts[-1], map_location="cuda")
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            start_step = ckpt["step"]
            print(f"✅ Resumed from step {start_step}")
        except Exception as e:
            print(f"⚠️ Failed to load checkpoint: {e}")

    dataset = AdamDataset(DATA_FILE, tokenizer, MAX_SEQ_LEN)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=0)

    print(f"🚀 Training Started...")
    
    step = start_step
    current_loss = 0
    tokens_processed = 0
    start_time = time.time()
    last_save = time.time()
    
    for batch in loader:
        step += 1
        input_ids = batch.to("cuda")
        tokens_processed += input_ids.numel()
        
        outputs = model(input_ids)
        logits = outputs.logits
        
        loss = torch.nn.functional.cross_entropy(
            logits[..., :-1, :].reshape(-1, logits.size(-1)),
            input_ids[..., 1:].reshape(-1)
        )
        
        (loss / GRAD_ACCUM).backward()
        current_loss += loss.item()
        
        if step % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            
            # Telemetry
            elapsed = time.time() - start_time
            tokens_sec = tokens_processed / elapsed if elapsed > 0 else 0
            avg_loss = current_loss / GRAD_ACCUM
            
            thermal_check()
            log_metrics(step, avg_loss, optimizer.param_groups[0]['lr'], tokens_sec)
            
            print(f"Step {step} | Loss: {avg_loss:.4f} | Speed: {tokens_sec:.0f} tok/s")
            
            # Reset counters
            current_loss = 0
            tokens_processed = 0
            start_time = time.time()

            # Snapshot every 500 steps
            if step % 500 == 0:
                save_logic_snapshot(model, tokenizer, step)

            # Save & Upload every hour
            if time.time() - last_save > (SAVE_EVERY_MINS * 60):
                safe_save(model, optimizer, step)
                last_save = time.time()

    safe_save(model, optimizer, step)
    print(f"✅ TRAINING COMPLETE at Step {step}")

if __name__ == "__main__":
    main()