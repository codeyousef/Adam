import os
import time
import json
import torch
import glob
import warnings
import csv
import subprocess
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

# --- HYPERPARAMETERS ---
SAVE_EVERY_MINS = 60
GRAD_ACCUM = 4
BATCH_SIZE = 8
LEARNING_RATE = 0.002
MAX_SEQ_LEN = 1536
THERMAL_THRESHOLD = 80
KEEP_LOCAL_CKPTS = 3
TARGET_LOSS = 1.0  # Stop if average loss hits this

# --- PROBES ---
PROBES = [
    "def solve_logic_puzzle(x):",
    "If all A are B, and some B are C, then:",
    "To calculate the trajectory of a rocket, one must first:",
    "System: You are a coding assistant. User: Write a Python function to parse JSON."
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
    X = G.bfloat16()
    X /= (X.norm() + eps)
    if G.size(0) > G.size(1): X = X.T
    for _ in range(steps):
        A = X @ X.T
        X = 3.4445 * X - 4.7750 * (A @ X) + 2.0315 * (A @ A @ X)
    if G.size(0) > G.size(1): X = X.T
    return X.to(G.dtype)

class M3Optimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, slow_momentum=0.99, slow_freq=10, ns_steps=5):
        super().__init__(params, dict(lr=lr, momentum=momentum, slow_momentum=slow_momentum, slow_freq=slow_freq, ns_steps=ns_steps))
    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                if p.grad is None: continue
                state = self.state[p]
                if len(state)==0: state['step']=0; state['fast']=torch.zeros_like(p); state['slow']=torch.zeros_like(p)
                state['step']+=1
                fast, slow = state['fast'], state['slow']
                fast.mul_(group['momentum']).add_(p.grad)
                if state['step'] % group['slow_freq'] == 0: slow.mul_(group['slow_momentum']).add_(fast)
                update = fast + 0.5 * slow
                if p.ndim >= 2: p.data.add_(zeropower_via_newtonschulz5(update, group['ns_steps']), alpha=-lr)
                else: p.data.add_(update, alpha=-lr*0.1)

# --- 1-EPOCH DATASET ---
class AdamDataset(IterableDataset):
    def __init__(self, filepath, tokenizer, max_len):
        self.filepath = filepath
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __iter__(self):
        # 🛑 SINGLE PASS: No 'while True' loop. Stops when file ends.
        with open(self.filepath, "r", encoding="utf-8") as f:
            for line in f:
                try: 
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

def main():
    torch.set_float32_matmul_precision("high"); init_metrics()
    tok = AutoTokenizer.from_pretrained(TOKENIZER_ID); tok.pad_token = tok.eos_token
    model = MambaLMHeadModel.from_pretrained(MODEL_NAME, dtype=torch.bfloat16).to("cuda"); model.train()
    opt = M3Optimizer([{"params":[p for p in model.parameters() if p.ndim==2],"lr":LEARNING_RATE},{"params":[p for p in model.parameters() if p.ndim<2],"lr":0.001}])
    
    # Check for resume
    start_step = 0
    ckpts = sorted(glob.glob(f"{CHECKPOINT_DIR}/adam_ckpt_*.pt"), key=os.path.getmtime)
    if ckpts:
        try:
            print(f"📂 Resuming from: {ckpts[-1]}")
            ckpt = torch.load(ckpts[-1], map_location="cuda")
            model.load_state_dict(ckpt["model"]); opt.load_state_dict(ckpt["optimizer"])
            start_step = ckpt["step"]
        except: print("⚠️ Resume failed, starting fresh.")

    dataset = AdamDataset(DATA_FILE, tok, MAX_SEQ_LEN)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=0)
    print("🚀 Phoenix V4 Launched (1 Epoch Limit OR Target Loss)")
    
    step = start_step
    cl, start = 0, time.time()
    last_save = time.time()
    loss_window = []
    WINDOW_SIZE = 50
    
    for batch in loader:
        step+=1
        input_ids = batch["input_ids"].to("cuda")
        attn_mask = batch["attention_mask"].to("cuda")
        
        labels = input_ids.clone(); labels[attn_mask==0]=-100
        loss = torch.nn.functional.cross_entropy(model(input_ids).logits[...,:-1,:].flatten(0,1), labels[...,1:].flatten(), ignore_index=-100)
        
        (loss/GRAD_ACCUM).backward(); cl+=loss.item()
        
        if step%GRAD_ACCUM==0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); opt.zero_grad(); thermal_check()
            avg_loss = cl/GRAD_ACCUM
            log_metrics(step, avg_loss, LEARNING_RATE, (batch["input_ids"].numel()*GRAD_ACCUM)/(time.time()-start))
            print(f"Step {step} | Loss {avg_loss:.4f}")
            
            # --- STOP CONDITION: LOW LOSS ---
            loss_window.append(avg_loss)
            if len(loss_window) > WINDOW_SIZE: loss_window.pop(0)
            if len(loss_window) == WINDOW_SIZE:
                rolling = sum(loss_window)/len(loss_window)
                if rolling < TARGET_LOSS:
                    print(f"🎉 CONVERGENCE! (Avg {rolling:.4f} < {TARGET_LOSS})"); safe_save(model, opt, step); break

            cl=0; start=time.time()
            if step % 500 == 0: save_logic_snapshot(model, tok, step)
            if time.time()-last_save > (SAVE_EVERY_MINS*60): safe_save(model, opt, step); last_save=time.time()
    
    safe_save(model, opt, step)
    print(f"✅ TRAINING COMPLETE (Epoch finished at Step {step})")

if __name__ == "__main__": main()