import os
import time
import json
import torch
import glob
import warnings
import csv
import subprocess
import math
import random
from datetime import datetime
from torch.utils.data import IterableDataset, DataLoader
from transformers import AutoTokenizer
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from huggingface_hub import HfApi

# --- SILENCE WARNINGS ---
warnings.filterwarnings("ignore")

# --- USER CONFIGURATION (EDIT THIS) ---
HF_REPO_ID = "codeyousef/adam-mamba-2.7b-logic"
HF_TOKEN = "hf"  # <--- PASTE YOUR WRITE TOKEN HERE

# --- RESEARCH CONFIG ---
MODEL_NAME = "state-spaces/mamba2-2.7b"
TOKENIZER_ID = "EleutherAI/gpt-neox-20b"
DATA_FILE = "adam_skeleton_data.jsonl"
CHECKPOINT_DIR = "adam_checkpoints"
TELEMETRY_FILE = "adam_research_metrics.csv"
SNAPSHOT_FILE = "adam_logic_snapshots.txt"

# --- HYPERPARAMETERS (M3/Muon Industry Standard) ---
SAVE_EVERY_MINS = 60
GRAD_ACCUM = 4
BATCH_SIZE = 8
MAX_SEQ_LEN = 2048
THERMAL_THRESHOLD = 80
KEEP_LOCAL_CKPTS = 3
TARGET_LOSS = 0.4

# --- LR SCHEDULE (Cosine with Warmup) ---
PEAK_LR = 0.01            # Standard for Muon/M3 (was 0.0005 - 20x too low)
MIN_LR = 1e-5             # End of cosine decay
WARMUP_STEPS = 1500       # Linear warmup (~2% of training)
TOTAL_STEPS = 150000      # ~2 epochs of 5M samples at batch 32
WEIGHT_DECAY = 0.05       # Decoupled weight decay for regularization

# --- VALIDATION ---
VAL_SPLIT = 0.05          # 5% holdout for eval
VAL_EVERY_STEPS = 1000    # Evaluate every N steps

# --- FAILURE DETECTION (Divergence only - no stagnation abort) ---
DIVERGENCE_THRESHOLD = 0.5    # Only abort on major loss spikes
MIN_STEPS_BEFORE_CHECK = 5000 # Let warmup finish before checking

# --- PROBES: Test structured reasoning format ---
PROBES = [
    "<|begin_of_thought|>\nAnalyze the logical structure.",
    "Problem: If all A are B, and some B are C, what can we conclude about A and C?\n\n<|begin_of_thought|>",
    "Problem: A ball is released in a sealed chamber with no gravity. Describe its motion.\n\n<|begin_of_thought|>",
    "Specification: Write a function to reverse a string without using built-in reverse methods.\n\n<|begin_of_thought|>",
]

api = HfApi(token=HF_TOKEN)

# --- HARDWARE MONITORS ---
def get_gpu_metrics():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=temperature.gpu,power.draw,memory.used', '--format=csv,noheader,nounits'], capture_output=True, text=True)
        d = result.stdout.strip().split(', ')
        return int(d[0]), float(d[1]), int(d[2])
    except: return 0, 0.0, 0

def thermal_check():
    t, _, _ = get_gpu_metrics()
    if t > THERMAL_THRESHOLD:
        print(f"🌡️ GPU {t}°C. Cooling...")
        while get_gpu_metrics()[0] > (THERMAL_THRESHOLD - 10): time.sleep(30)

# --- LR SCHEDULE ---
def get_lr(step, warmup_steps=WARMUP_STEPS, total_steps=TOTAL_STEPS, peak_lr=PEAK_LR, min_lr=MIN_LR):
    """Cosine decay with linear warmup - industry standard for Muon/M3."""
    if step < warmup_steps:
        # Linear warmup
        return peak_lr * (step / max(1, warmup_steps))
    else:
        # Cosine decay
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, progress)  # Clamp to 1.0 if we exceed total_steps
        return min_lr + 0.5 * (peak_lr - min_lr) * (1 + math.cos(math.pi * progress))

# --- LOGGING ---
def init_metrics():
    if not os.path.exists(TELEMETRY_FILE):
        with open(TELEMETRY_FILE, "w") as f: csv.writer(f).writerow(["Timestamp", "Step", "Loss", "LR", "VRAM", "Temp", "TPS"])

def log_metrics(step, loss, lr, tps):
    t, _, v = get_gpu_metrics()
    with open(TELEMETRY_FILE, "a") as f: csv.writer(f).writerow([datetime.now().isoformat(), step, f"{loss:.4f}", f"{lr:.6f}", v, t, f"{tps:.1f}"])

def save_logic_snapshot(model, tokenizer, step):
    print(f"📸 Taking Snapshot at {step}...")
    model.eval()
    with open(SNAPSHOT_FILE, "a") as f:
        f.write(f"\n{'='*40}\nSNAPSHOT STEP {step}\n{'='*40}\n")
        for probe in PROBES:
            try:
                inputs = tokenizer(probe, return_tensors="pt").to("cuda")
                max_len = inputs.input_ids.shape[1] + 64
                with torch.no_grad():
                    output = model.generate(inputs.input_ids, max_len, temperature=0.7, top_p=0.9, cg=True)
                decoded = tokenizer.decode(output[0])
                f.write(f"PROBE: {probe}\nRESULT: {decoded}\n{'-'*20}\n")
            except Exception as e:
                f.write(f"PROBE FAILED: {e}\n")
    model.train()

# --- OPTIMIZER (M3) ---
def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    """Newton-Schulz orthogonalization for matrix updates."""
    # Skip very small matrices - NS iteration needs minimum size
    if min(G.size()) < 16:
        return G  # Return unmodified for tiny matrices
    
    X = G.bfloat16()
    X /= (X.norm() + eps)
    if G.size(0) > G.size(1): X = X.T
    for _ in range(steps):
        A = X @ X.T
        X = 3.4445 * X - 4.7750 * (A @ X) + 2.0315 * (A @ A @ X)
    if G.size(0) > G.size(1): X = X.T
    return X.to(G.dtype)

class M3Optimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, slow_momentum=0.99, slow_freq=10, ns_steps=5, weight_decay=0.0):
        defaults = dict(lr=lr, momentum=momentum, slow_momentum=slow_momentum, slow_freq=slow_freq, ns_steps=ns_steps, weight_decay=weight_decay)
        super().__init__(params, defaults)
    
    def set_lr(self, lr_2d, lr_1d):
        """Update learning rates for all param groups."""
        for group in self.param_groups:
            if group.get('is_2d', True):
                group['lr'] = lr_2d
            else:
                group['lr'] = lr_1d
    
    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            wd = group.get('weight_decay', 0.0)
            for p in group['params']:
                if p.grad is None: continue
                state = self.state[p]
                if len(state)==0: state['step']=0; state['fast']=torch.zeros_like(p); state['slow']=torch.zeros_like(p)
                state['step']+=1
                
                # Decoupled weight decay (applied before update)
                if wd > 0 and p.ndim >= 2:
                    p.data.mul_(1 - lr * wd)
                
                fast, slow = state['fast'], state['slow']
                fast.mul_(group['momentum']).add_(p.grad)
                if state['step'] % group['slow_freq'] == 0: slow.mul_(group['slow_momentum']).add_(fast)
                update = fast + 0.5 * slow
                if p.ndim >= 2: p.data.add_(zeropower_via_newtonschulz5(update, group['ns_steps']), alpha=-lr)
                else: p.data.add_(update, alpha=-lr*0.1)

# --- MULTI-EPOCH DATASET WITH TRAIN/VAL SPLIT ---
class AdamDataset(IterableDataset):
    def __init__(self, filepath, tokenizer, max_len, val_split=0.0, is_val=False, seed=42):
        self.filepath = filepath
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.val_split = val_split
        self.is_val = is_val
        self.seed = seed
    
    def __iter__(self):
        rng = random.Random(self.seed)
        with open(self.filepath, "r", encoding="utf-8") as f:
            for line in f:
                try: 
                    # Deterministic split based on line hash
                    is_val_sample = rng.random() < self.val_split
                    if self.is_val != is_val_sample:
                        continue
                    
                    data = json.loads(line)
                    txt = data.get("text","")
                    if not txt: continue
                    enc = self.tokenizer(txt, truncation=True, max_length=self.max_len, padding="max_length", return_tensors="pt")
                    yield {"input_ids": enc.input_ids.squeeze(0), "attention_mask": enc.attention_mask.squeeze(0)}
                except: continue

# --- MAIN ---
def safe_save(model, optimizer, step):
    if not os.path.exists(CHECKPOINT_DIR): os.makedirs(CHECKPOINT_DIR)
    path = f"{CHECKPOINT_DIR}/adam_ckpt_{step}.pt"
    torch.save({"step":step,"model":model.state_dict(),"optimizer":optimizer.state_dict()}, path)
    try: 
        api.upload_file(path_or_fileobj=path, path_in_repo=f"checkpoints/adam_ckpt_{step}.pt", repo_id=HF_REPO_ID, repo_type="model")
        api.upload_file(path_or_fileobj=TELEMETRY_FILE, path_in_repo=TELEMETRY_FILE, repo_id=HF_REPO_ID, repo_type="model")
    except: print("⚠️ Cloud Upload Failed")
    ckpts = sorted(glob.glob(f"{CHECKPOINT_DIR}/adam_ckpt_*.pt"), key=os.path.getmtime)
    while len(ckpts) > KEEP_LOCAL_CKPTS: os.remove(ckpts.pop(0))

def evaluate_val_loss(model, val_loader, max_batches=50):
    """Compute validation loss on held-out data."""
    model.eval()
    total_loss = 0
    count = 0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= max_batches: break
            input_ids = batch["input_ids"].to("cuda")
            attn_mask = batch["attention_mask"].to("cuda")
            labels = input_ids.clone(); labels[attn_mask==0]=-100
            loss = torch.nn.functional.cross_entropy(
                model(input_ids).logits[...,:-1,:].flatten(0,1), 
                labels[...,1:].flatten(), ignore_index=-100
            )
            total_loss += loss.item()
            count += 1
    model.train()
    return total_loss / max(1, count)

def main():
    torch.set_float32_matmul_precision("high"); init_metrics()
    tok = AutoTokenizer.from_pretrained(TOKENIZER_ID); tok.pad_token = tok.eos_token
    model = MambaLMHeadModel.from_pretrained(MODEL_NAME, dtype=torch.bfloat16).to("cuda"); model.train()
    
    # M3 Optimizer with proper param groups and weight decay
    params_2d = [{"params": [p for p in model.parameters() if p.ndim >= 2], "lr": PEAK_LR, "is_2d": True, "weight_decay": WEIGHT_DECAY}]
    params_1d = [{"params": [p for p in model.parameters() if p.ndim < 2], "lr": PEAK_LR * 0.1, "is_2d": False, "weight_decay": 0.0}]
    opt = M3Optimizer(params_2d + params_1d)
    
    # Check for resume
    start_step = 0
    ckpts = sorted(glob.glob(f"{CHECKPOINT_DIR}/adam_ckpt_*.pt"), key=os.path.getmtime)
    if ckpts:
        try:
            print(f"📂 Resuming from: {ckpts[-1]}")
            ckpt = torch.load(ckpts[-1], map_location="cuda")
            model.load_state_dict(ckpt["model"])
            start_step = ckpt["step"]
            # Note: We intentionally skip loading optimizer state here because:
            # 1. The new optimizer has different param groups (with weight decay)
            # 2. Fresh momentum buffers + new LR schedule will help escape the plateau
            print(f"   ✓ Model weights loaded from step {start_step}")
            print(f"   ✓ Fresh optimizer state (new LR schedule will kick in)")
        except Exception as e: 
            print(f"⚠️ Resume failed ({e}), starting fresh.")

    # Create train and validation datasets
    train_dataset = AdamDataset(DATA_FILE, tok, MAX_SEQ_LEN, val_split=VAL_SPLIT, is_val=False)
    val_dataset = AdamDataset(DATA_FILE, tok, MAX_SEQ_LEN, val_split=VAL_SPLIT, is_val=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=0)
    
    print(f"🚀 Phoenix V5 Launched (Cosine LR Schedule + {TOTAL_STEPS} steps)")
    print(f"   Peak LR: {PEAK_LR}, Min LR: {MIN_LR}, Warmup: {WARMUP_STEPS} steps")
    print(f"   Weight Decay: {WEIGHT_DECAY}, Validation Split: {VAL_SPLIT*100:.0f}%")
    
    step = start_step
    epoch = 0
    cl, start = 0, time.time()
    last_save = time.time()
    loss_window = []
    WINDOW_SIZE = 50
    best_val_loss = float('inf')
    
    # Failure detection state (divergence only - no stagnation abort)
    loss_history = []
    
    while step < TOTAL_STEPS:
        epoch += 1
        print(f"\n📚 Starting Epoch {epoch}")
        
        # Recreate dataloader for new epoch (reshuffles via new iterator)
        train_dataset = AdamDataset(DATA_FILE, tok, MAX_SEQ_LEN, val_split=VAL_SPLIT, is_val=False, seed=42+epoch)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=0)
        
        for batch in train_loader:
            if step >= TOTAL_STEPS:
                break
            
            step += 1
            
            # Update LR according to schedule
            current_lr = get_lr(step)
            opt.set_lr(current_lr, current_lr * 0.1)
            
            input_ids = batch["input_ids"].to("cuda")
            attn_mask = batch["attention_mask"].to("cuda")
            
            labels = input_ids.clone(); labels[attn_mask==0]=-100
            loss = torch.nn.functional.cross_entropy(model(input_ids).logits[...,:-1,:].flatten(0,1), labels[...,1:].flatten(), ignore_index=-100)
            
            (loss/GRAD_ACCUM).backward(); cl+=loss.item()
            
            if step % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad()
                thermal_check()
                
                avg_loss = cl/GRAD_ACCUM
                log_metrics(step, avg_loss, current_lr, (batch["input_ids"].numel()*GRAD_ACCUM)/(time.time()-start))
                print(f"Step {step}/{TOTAL_STEPS} | Loss {avg_loss:.4f} | LR {current_lr:.6f}")
                
                # Track loss history for divergence detection
                loss_history.append(avg_loss)
                
                # --- DIVERGENCE DETECTION (only after warmup) ---
                if step > MIN_STEPS_BEFORE_CHECK and len(loss_history) >= 200:
                    recent_avg = sum(loss_history[-50:]) / 50
                    earlier_avg = sum(loss_history[-200:-150]) / 50
                    
                    if recent_avg > earlier_avg + DIVERGENCE_THRESHOLD:
                        print(f"❌ DIVERGENCE DETECTED! Loss increased from {earlier_avg:.4f} to {recent_avg:.4f}")
                        print(f"   Aborting run - possible LR too high or data issue.")
                        safe_save(model, opt, step)
                        exit(2)
                
                # --- SUCCESS CONDITION: LOW LOSS ---
                loss_window.append(avg_loss)
                if len(loss_window) > WINDOW_SIZE: loss_window.pop(0)
                if len(loss_window) == WINDOW_SIZE:
                    rolling = sum(loss_window)/len(loss_window)
                    if rolling < TARGET_LOSS:
                        print(f"🎉 CONVERGENCE! (Avg {rolling:.4f} < {TARGET_LOSS})")
                        safe_save(model, opt, step)
                        return
                
                cl = 0; start = time.time()
                
                # Periodic validation
                if step % VAL_EVERY_STEPS == 0:
                    val_loss = evaluate_val_loss(model, val_loader)
                    print(f"   📊 Val Loss: {val_loss:.4f}" + (" (best!)" if val_loss < best_val_loss else ""))
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        safe_save(model, opt, step)  # Save best model
                
                if step % 500 == 0: save_logic_snapshot(model, tok, step)
                if time.time() - last_save > (SAVE_EVERY_MINS * 60): 
                    safe_save(model, opt, step)
                    last_save = time.time()
    
    safe_save(model, opt, step)
    print(f"\n✅ TRAINING COMPLETE at Step {step} (Epoch {epoch})")
    print(f"   Best Val Loss: {best_val_loss:.4f}")

if __name__ == "__main__": main()