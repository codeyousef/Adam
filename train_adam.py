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
TELEMETRY_FILE = "adam_research_metrics.csv"
SNAPSHOT_FILE = "adam_logic_snapshots.txt"
EXPERIMENT_LOG = "adam_experiment_config.json"
ERROR_LOG = "adam_errors.log"

# Reasoning Probes
PROBES = [
    "If all <ORG> are <GPE>, and <PERSON> is an <ORG>, then <PERSON> is...",
    "To calculate the <CONCEPT> of a <OBJECT>, one must first derive the...",
    "Python: def solve(x): if x > 10: return <MASK> else: return",
]

# --- HYPERPARAMETERS (B200 OPTIMIZED) ---
# 1 Epoch = 1 pass through DATA_FILE. The loop ends when file ends.
SAVE_EVERY_MINS = 30
GRAD_ACCUM = 4          # Batch Size 16 * 4 = 64 effective batch size
BATCH_SIZE = 16         # High batch size for B200 192GB VRAM
LEARNING_RATE = 0.02    # High LR for Muon/M3
MAX_SEQ_LEN = 1536
MIN_SEQ_LEN = 512
VALIDATION_INTERVAL = 500
MAX_CONSECUTIVE_OOM = 5

# --- M3 OPTIMIZER IMPLEMENTATION (NESTED LEARNING) ---
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
                
                # Fast & Slow Momentum Updates
                fast_buf.mul_(momentum).add_(grad)
                if state['step'] % slow_freq == 0:
                    slow_buf.mul_(slow_momentum).add_(fast_buf)
                
                update_tensor = fast_buf + (0.5 * slow_buf)
                
                # Apply Newton-Schulz to 2D matrices; Standard SGD to vectors
                if p.ndim == 2:
                    ortho_update = zeropower_via_newtonschulz5(update_tensor, steps=ns_steps)
                    p.data.add_(ortho_update, alpha=-lr)
                else:
                    p.data.add_(update_tensor, alpha=-lr * 0.1)

# --- DATASET & TRAINING LOOP ---
class AdamDataset(IterableDataset):
    def __init__(self, filepath, tokenizer, max_len):
        self.filepath = filepath
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        with open(self.filepath, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                # Simple sharding
                if worker_info and i % worker_info.num_workers != worker_info.id:
                    continue
                try:
                    text = json.loads(line)["text"]
                    enc = self.tokenizer(text, truncation=True, max_length=self.max_len, return_tensors="pt")
                    yield enc.input_ids.squeeze(0)
                except (json.JSONDecodeError, KeyError):
                    continue

def safe_save(model, optimizer, step, loss, current_seq_len):
    if not os.path.exists(CHECKPOINT_DIR): os.makedirs(CHECKPOINT_DIR)
    tmp_path = f"{CHECKPOINT_DIR}/tmp_adam.pt"
    torch.save({
        "step": step, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "loss": loss, "seq_len": current_seq_len, "timestamp": datetime.now().isoformat()
    }, tmp_path)
    os.replace(tmp_path, f"{CHECKPOINT_DIR}/adam_ckpt_{step}.pt")
    print(f"💾 Checkpoint saved at step {step}")
    # Cleanup old checkpoints
    ckpts = sorted(glob.glob(f"{CHECKPOINT_DIR}/adam_ckpt_*.pt"), key=os.path.getmtime)
    while len(ckpts) > 3: os.remove(ckpts.pop(0))

def main():
    torch.set_float32_matmul_precision("high")
    print(f"🐈 Catbelly Studio: Loading Adam's Architecture ({MODEL_NAME}) on B200...")
    
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    tokenizer.pad_token = tokenizer.eos_token
    model = MambaLMHeadModel.from_pretrained(MODEL_NAME, dtype=torch.bfloat16).to("cuda")
    model.train()

    # M3 Optimizer Setup (Matrix Params vs Vector Params)
    m3_params = [p for n, p in model.named_parameters() if p.requires_grad and p.ndim == 2]
    std_params = [p for n, p in model.named_parameters() if p.requires_grad and p.ndim < 2]
    optimizer = M3Optimizer([
        {"params": m3_params, "lr": 0.02, "momentum": 0.95},
        {"params": std_params, "lr": 0.001, "momentum": 0.9}
    ], slow_freq=10)

    # Resume Checkpoint
    start_step = 0
    total_tokens = 0
    ckpts = sorted(glob.glob(f"{CHECKPOINT_DIR}/adam_ckpt_*.pt"), key=os.path.getmtime)
    if ckpts:
        ckpt = torch.load(ckpts[-1], map_location="cuda")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"]
        print(f"📂 Resuming from step {start_step}...")

    # Dataset & Loader
    dataset = AdamDataset(DATA_FILE, tokenizer, MAX_SEQ_LEN)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=4, prefetch_factor=2)

    print(f">>> 🚀 TRAINING STARTED (1 Epoch over {DATA_FILE})... <<<")
    
    step = start_step
    last_save = time.time()
    current_loss = 0
    
    # This loop naturally ends when DATA_FILE is fully read (1 Epoch)
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
        current_loss += loss.item() / GRAD_ACCUM
        
        if step % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            
            # Simple logging
            print(f"Step {step} | Loss: {current_loss:.4f} | VRAM: {torch.cuda.max_memory_allocated()/1e9:.1f}GB")
            current_loss = 0

            # Save periodically
            if time.time() - last_save > (SAVE_EVERY_MINS * 60):
                safe_save(model, optimizer, step, loss.item(), MAX_SEQ_LEN)
                last_save = time.time()

    # Final Save
    safe_save(model, optimizer, step, 0.0, MAX_SEQ_LEN)
    print(f"✅ ONE EPOCH COMPLETE at Step {step}")

if __name__ == "__main__":
    main()