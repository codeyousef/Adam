import os
import time
import signal
import json
import torch
import glob
import warnings
import csv
import gc
import traceback
import subprocess
from datetime import datetime
from torch.utils.data import IterableDataset, DataLoader
from transformers import AutoTokenizer
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

# --- SILENCE WARNINGS ---
warnings.filterwarnings("ignore", message=".*set_float32_matmul_precision.*")

# --- RESEARCH CONFIG ---
MODEL_NAME = "state-spaces/mamba2-2.7b"
TOKENIZER_ID = "EleutherAI/gpt-neox-20b"
DATA_FILE = "adam_skeleton_data.jsonl"
CHECKPOINT_DIR = "adam_checkpoints"

# --- HYPERPARAMETERS ---
SAVE_EVERY_MINS = 60    # CHANGED: Save only once per hour to save disk space
GRAD_ACCUM = 4
BATCH_SIZE = 8
LEARNING_RATE = 0.002
MAX_SEQ_LEN = 1536
THERMAL_THRESHOLD = 80

def get_gpu_temp():
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        return int(result.stdout.strip().split('\n')[0])
    except:
        return 0

def thermal_check():
    temp = get_gpu_temp()
    if temp > THERMAL_THRESHOLD:
        print(f"🌡️  GPU at {temp}°C - pausing for cooldown...")
        while get_gpu_temp() > THERMAL_THRESHOLD - 10:
            time.sleep(30)
        print(f"✅ GPU cooled - resuming")

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

def safe_save(model, optimizer, step):
    if not os.path.exists(CHECKPOINT_DIR): os.makedirs(CHECKPOINT_DIR)
    
    # Save to temp file first to avoid corruption if disk fills
    path = f"{CHECKPOINT_DIR}/adam_ckpt_{step}.pt"
    tmp_path = f"{CHECKPOINT_DIR}/tmp_saving.pt"
    
    try:
        torch.save({
            "step": step, 
            "model": model.state_dict(), 
            "optimizer": optimizer.state_dict()
        }, tmp_path)
        os.replace(tmp_path, path)
        print(f"💾 Saved checkpoint: {path}")
        
        # CHANGED: Keep only 1 previous checkpoint to save space
        ckpts = sorted(glob.glob(f"{CHECKPOINT_DIR}/adam_ckpt_*.pt"), key=os.path.getmtime)
        while len(ckpts) > 1:
            oldest = ckpts.pop(0)
            os.remove(oldest)
            print(f"🗑️ Deleted old checkpoint: {oldest}")
            
    except OSError as e:
        print(f"❌ Save failed (Disk Full?): {e}")
        if os.path.exists(tmp_path): os.remove(tmp_path)

def main():
    torch.set_float32_matmul_precision("high")
    print(f"🐈 Loading Adam (Mamba-2) on B200...")
    
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = MambaLMHeadModel.from_pretrained(MODEL_NAME, dtype=torch.bfloat16).to("cuda")
    model.train()

    # M3 Optimizer
    m3_params = [p for n, p in model.named_parameters() if p.requires_grad and p.ndim == 2]
    std_params = [p for n, p in model.named_parameters() if p.requires_grad and p.ndim < 2]
    optimizer = M3Optimizer([
        {"params": m3_params, "lr": 0.02, "momentum": 0.95},
        {"params": std_params, "lr": 0.001, "momentum": 0.9}
    ], slow_freq=10)

    # Resume Checkpoint
    start_step = 0
    ckpts = sorted(glob.glob(f"{CHECKPOINT_DIR}/adam_ckpt_*.pt"), key=os.path.getmtime)
    if ckpts:
        try:
            ckpt = torch.load(ckpts[-1], map_location="cuda")
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            start_step = ckpt["step"]
            print(f"📂 Resuming from step {start_step}...")
        except Exception as e:
            print(f"⚠️ Failed to load checkpoint: {e}")

    dataset = AdamDataset(DATA_FILE, tokenizer, MAX_SEQ_LEN)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=0)

    print(f"🚀 Training Started...")
    
    step = start_step
    current_loss = 0
    last_save = time.time()
    
    for batch in loader:
        step += 1
        input_ids = batch.to("cuda")
        
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
            
            thermal_check()
            print(f"Step {step} | Loss: {current_loss/GRAD_ACCUM:.4f}")
            current_loss = 0

            if time.time() - last_save > (SAVE_EVERY_MINS * 60):
                safe_save(model, optimizer, step)
                last_save = time.time()

    # Final save at end of training
    safe_save(model, optimizer, step)
    print(f"✅ ONE EPOCH COMPLETE at Step {step}")

if __name__ == "__main__":
    main()