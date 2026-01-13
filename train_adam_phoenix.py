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
import argparse
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
CHECKPOINT_DIR = "adam_checkpoints"
TELEMETRY_FILE = "adam_research_metrics.csv"
SNAPSHOT_FILE = "adam_logic_snapshots.txt"

# --- CURRICULUM DATA FILES (from data_forge.py V4) ---
CURRICULUM_DATA_DIR = "adam_curriculum_data"
PHASE_FILES = {
    1: "phase1_axiomatic.jsonl",      # Axiomatic Knowledge
    2: "phase2_algorithmic.jsonl",    # Algorithmic Hardening
    3: "phase3_crystallization.jsonl" # Reasoning Crystallization
}

# --- HYPERPARAMETERS (M3/Muon Industry Standard) ---
SAVE_EVERY_MINS = 60
GRAD_ACCUM = 4
BATCH_SIZE = 8
MAX_SEQ_LEN = 2048
THERMAL_THRESHOLD = 80
KEEP_LOCAL_CKPTS = 3
TARGET_LOSS = 0.4

# --- PHASE-SPECIFIC LEARNING RATES ---
# Per Data Curation.md Section 4.3: Different phases need different treatment
PHASE_CONFIG = {
    1: {  # Axiomatic Knowledge - slower, careful learning
        "peak_lr": 0.005,
        "min_lr": 1e-5,
        "warmup_steps": 1000,
        "total_steps": 50000,
        "weight_decay": 0.05,
        "description": "Axiomatic Knowledge (World Model)"
    },
    2: {  # Algorithmic Hardening - standard training
        "peak_lr": 0.01,
        "min_lr": 1e-5,
        "warmup_steps": 1500,
        "total_steps": 75000,
        "weight_decay": 0.05,
        "description": "Algorithmic Hardening (Logic Operations)"
    },
    3: {  # Reasoning Crystallization - annealing phase
        "peak_lr": 0.008,
        "min_lr": 1e-6,  # Lower minimum for fine-grained crystallization
        "warmup_steps": 500,
        "total_steps": 50000,
        "weight_decay": 0.03,  # Lower WD for final phase
        "description": "Reasoning Crystallization (Behavior Setting)"
    }
}

# --- VALIDATION ---
VAL_SPLIT = 0.05          # 5% holdout for eval
VAL_EVERY_STEPS = 1000    # Evaluate every N steps

# --- FAILURE DETECTION (Divergence only - no stagnation abort) ---
DIVERGENCE_THRESHOLD = 0.5    # Only abort on major loss spikes
MIN_STEPS_BEFORE_CHECK = 3000 # Let warmup finish before checking

# --- PROBES: Test structured reasoning format (updated for new tags) ---
PROBES = [
    "<|begin_of_thought|>\nAnalyze the logical structure.",
    "Problem: If all A are B, and some B are C, what can we conclude about A and C?\n\n<|begin_of_thought|>",
    "Problem: A ball is released in a sealed chamber with no gravity. Describe its motion.\n\n<|begin_of_thought|>",
    "Specification: Write a function to reverse a string without using built-in reverse methods.\n\n<|begin_of_thought|>",
    "<|begin_of_thought|>\nDerive from first principles.",
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
        print(f"GPU {t}C. Cooling...")
        while get_gpu_metrics()[0] > (THERMAL_THRESHOLD - 10): time.sleep(30)

# --- LR SCHEDULE ---
def get_lr(step, warmup_steps, total_steps, peak_lr, min_lr):
    """Cosine decay with linear warmup - industry standard for Muon/M3."""
    if step < warmup_steps:
        # Linear warmup
        return peak_lr * (step / max(1, warmup_steps))
    else:
        # Cosine decay
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, progress)  # Clamp to 1.0 if we exceed total_steps
        return min_lr + 0.5 * (peak_lr - min_lr) * (1 + math.cos(math.pi * progress))

# --- LOGGING (Paper-Ready Metrics) ---
PAPER_METRICS_FILE = "adam_paper_metrics.jsonl"  # Detailed metrics for arXiv paper

def init_metrics():
    if not os.path.exists(TELEMETRY_FILE):
        with open(TELEMETRY_FILE, "w") as f: csv.writer(f).writerow(["Timestamp", "Phase", "Step", "Loss", "ValLoss", "LR", "VRAM", "Temp", "TPS"])

def log_metrics(phase, step, loss, lr, tps, val_loss=None):
    t, _, v = get_gpu_metrics()
    # CSV for quick analysis
    with open(TELEMETRY_FILE, "a") as f:
        csv.writer(f).writerow([datetime.now().isoformat(), phase, step, f"{loss:.4f}", f"{val_loss:.4f}" if val_loss else "", f"{lr:.6f}", v, t, f"{tps:.1f}"])

def log_paper_metrics(phase, step, train_loss, val_loss=None, probe_results=None, extra=None):
    """Log detailed metrics for arXiv paper in JSONL format."""
    record = {
        "timestamp": datetime.now().isoformat(),
        "phase": phase,
        "phase_name": PHASE_CONFIG[phase]["description"],
        "step": step,
        "train_loss": round(train_loss, 6),
        "val_loss": round(val_loss, 6) if val_loss else None,
        "learning_rate": PHASE_CONFIG[phase]["peak_lr"],
        "gpu_temp": get_gpu_metrics()[0],
        "gpu_vram_mb": get_gpu_metrics()[2],
    }
    if probe_results:
        record["probe_results"] = probe_results
    if extra:
        record.update(extra)

    with open(PAPER_METRICS_FILE, "a") as f:
        json.dump(record, f)
        f.write("\n")

def save_logic_snapshot(model, tokenizer, phase, step, for_paper=False):
    """
    Take logic snapshots. If for_paper=True, returns structured results for paper metrics.
    """
    print(f"Taking Snapshot at Phase {phase}, Step {step}...")
    model.eval()
    probe_results = []

    with open(SNAPSHOT_FILE, "a") as f:
        f.write(f"\n{'='*40}\nSNAPSHOT PHASE {phase} STEP {step}\n{'='*40}\n")
        for probe in PROBES:
            try:
                inputs = tokenizer(probe, return_tensors="pt").to("cuda")
                max_len = inputs.input_ids.shape[1] + 128  # More tokens for reasoning
                with torch.no_grad():
                    output = model.generate(inputs.input_ids, max_len, temperature=0.7, top_p=0.9, cg=True)
                decoded = tokenizer.decode(output[0])
                f.write(f"PROBE: {probe}\nRESULT: {decoded}\n{'-'*20}\n")

                # Collect for paper metrics
                probe_results.append({
                    "probe": probe[:50] + "..." if len(probe) > 50 else probe,
                    "response": decoded[len(probe):200],  # First 200 chars of response
                    "has_thought_structure": "<|begin_of_thought|>" in decoded or "<|end_of_thought|>" in decoded,
                    "has_solution_structure": "<|begin_of_solution|>" in decoded or "<|end_of_solution|>" in decoded,
                })
            except Exception as e:
                f.write(f"PROBE FAILED: {e}\n")
                probe_results.append({"probe": probe[:50], "error": str(e)})

    model.train()
    return probe_results if for_paper else None

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
def safe_save(model, optimizer, phase, step):
    if not os.path.exists(CHECKPOINT_DIR): os.makedirs(CHECKPOINT_DIR)
    path = f"{CHECKPOINT_DIR}/adam_phase{phase}_step{step}.pt"
    torch.save({"phase": phase, "step":step, "model":model.state_dict(), "optimizer":optimizer.state_dict()}, path)
    try:
        api.upload_file(path_or_fileobj=path, path_in_repo=f"checkpoints/adam_phase{phase}_step{step}.pt", repo_id=HF_REPO_ID, repo_type="model")
        api.upload_file(path_or_fileobj=TELEMETRY_FILE, path_in_repo=TELEMETRY_FILE, repo_id=HF_REPO_ID, repo_type="model")
    except: print("Cloud Upload Failed")
    ckpts = sorted(glob.glob(f"{CHECKPOINT_DIR}/adam_phase*.pt"), key=os.path.getmtime)
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

def train_phase(model, tokenizer, opt, phase, data_file, start_step=0):
    """Train a single curriculum phase."""
    config = PHASE_CONFIG[phase]

    print(f"\n{'='*70}")
    print(f"PHASE {phase}: {config['description']}")
    print(f"{'='*70}")
    print(f"   Data: {data_file}")
    print(f"   Peak LR: {config['peak_lr']}, Min LR: {config['min_lr']}")
    print(f"   Warmup: {config['warmup_steps']} steps, Total: {config['total_steps']} steps")
    print(f"   Weight Decay: {config['weight_decay']}")

    # Create train and validation datasets
    train_dataset = AdamDataset(data_file, tokenizer, MAX_SEQ_LEN, val_split=VAL_SPLIT, is_val=False)
    val_dataset = AdamDataset(data_file, tokenizer, MAX_SEQ_LEN, val_split=VAL_SPLIT, is_val=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=0)

    step = start_step
    epoch = 0
    cl, start = 0, time.time()
    last_save = time.time()
    loss_window = []
    WINDOW_SIZE = 50
    best_val_loss = float('inf')

    # Failure detection state
    loss_history = []

    while step < config['total_steps']:
        epoch += 1
        print(f"\nPhase {phase} Epoch {epoch}")

        # Recreate dataloader for new epoch
        train_dataset = AdamDataset(data_file, tokenizer, MAX_SEQ_LEN, val_split=VAL_SPLIT, is_val=False, seed=42+epoch)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=0)

        for batch in train_loader:
            if step >= config['total_steps']:
                break

            step += 1

            # Update LR according to phase-specific schedule
            current_lr = get_lr(step, config['warmup_steps'], config['total_steps'],
                               config['peak_lr'], config['min_lr'])
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
                log_metrics(phase, step, avg_loss, current_lr, (batch["input_ids"].numel()*GRAD_ACCUM)/(time.time()-start))
                print(f"P{phase} Step {step}/{config['total_steps']} | Loss {avg_loss:.4f} | LR {current_lr:.6f}")

                # Track loss history for divergence detection
                loss_history.append(avg_loss)

                # --- DIVERGENCE DETECTION ---
                if step > MIN_STEPS_BEFORE_CHECK and len(loss_history) >= 200:
                    recent_avg = sum(loss_history[-50:]) / 50
                    earlier_avg = sum(loss_history[-200:-150]) / 50

                    if recent_avg > earlier_avg + DIVERGENCE_THRESHOLD:
                        print(f"DIVERGENCE DETECTED! Loss increased from {earlier_avg:.4f} to {recent_avg:.4f}")
                        safe_save(model, opt, phase, step)
                        return step, "divergence"

                # --- SUCCESS CONDITION ---
                loss_window.append(avg_loss)
                if len(loss_window) > WINDOW_SIZE: loss_window.pop(0)
                if len(loss_window) == WINDOW_SIZE:
                    rolling = sum(loss_window)/len(loss_window)
                    if rolling < TARGET_LOSS:
                        print(f"CONVERGENCE! (Avg {rolling:.4f} < {TARGET_LOSS})")
                        safe_save(model, opt, phase, step)
                        return step, "converged"

                cl = 0; start = time.time()

                # Periodic validation
                if step % VAL_EVERY_STEPS == 0:
                    val_loss = evaluate_val_loss(model, val_loader)
                    marker = " (best!)" if val_loss < best_val_loss else ""
                    print(f"   Val Loss: {val_loss:.4f}{marker}")

                    # Log paper metrics with probe results
                    probe_results = save_logic_snapshot(model, tokenizer, phase, step, for_paper=True)
                    log_paper_metrics(phase, step, avg_loss, val_loss, probe_results, extra={
                        "is_best": val_loss < best_val_loss,
                        "samples_seen": step * BATCH_SIZE * GRAD_ACCUM,
                    })

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        safe_save(model, opt, phase, step)

                elif step % 500 == 0:
                    save_logic_snapshot(model, tokenizer, phase, step)
                if time.time() - last_save > (SAVE_EVERY_MINS * 60):
                    safe_save(model, opt, phase, step)
                    last_save = time.time()

    safe_save(model, opt, phase, step)
    print(f"\nPhase {phase} Complete at Step {step}")
    print(f"   Best Val Loss: {best_val_loss:.4f}")
    return step, "complete"

def main():
    parser = argparse.ArgumentParser(description="Adam Phoenix Curriculum Training")
    parser.add_argument("--phase", type=int, default=0, help="Phase to train (1, 2, or 3). 0 = all phases sequentially")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--fresh", action="store_true", help="Start fresh (ignore checkpoints). Old checkpoints preserved as baseline.")
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")
    init_metrics()

    tok = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    tok.pad_token = tok.eos_token

    # Add special tokens for reasoning structure
    special_tokens = {
        "additional_special_tokens": [
            "<|begin_of_thought|>", "<|end_of_thought|>",
            "<|begin_of_solution|>", "<|end_of_solution|>"
        ]
    }
    tok.add_special_tokens(special_tokens)

    model = MambaLMHeadModel.from_pretrained(MODEL_NAME, dtype=torch.bfloat16).to("cuda")
    # Resize embeddings if we added tokens
    # Note: Mamba models may handle this differently - check if needed
    model.train()

    # M3 Optimizer with proper param groups
    config = PHASE_CONFIG.get(args.phase if args.phase > 0 else 1)
    params_2d = [{"params": [p for p in model.parameters() if p.ndim >= 2], "lr": config['peak_lr'], "is_2d": True, "weight_decay": config['weight_decay']}]
    params_1d = [{"params": [p for p in model.parameters() if p.ndim < 2], "lr": config['peak_lr'] * 0.1, "is_2d": False, "weight_decay": 0.0}]
    opt = M3Optimizer(params_2d + params_1d)

    # Check for resume
    start_phase = 1
    start_step = 0

    if args.fresh:
        # Archive old checkpoints as baseline for paper comparison
        old_ckpts = glob.glob(f"{CHECKPOINT_DIR}/adam_*.pt")
        if old_ckpts:
            baseline_dir = f"{CHECKPOINT_DIR}/baseline_v1"
            os.makedirs(baseline_dir, exist_ok=True)
            for ckpt in old_ckpts:
                if "baseline" not in ckpt:
                    import shutil
                    dest = os.path.join(baseline_dir, os.path.basename(ckpt))
                    if not os.path.exists(dest):
                        shutil.move(ckpt, dest)
                        print(f"Archived baseline: {ckpt} -> {dest}")
        print("Starting fresh training (baseline preserved for paper)")

    elif args.resume:
        ckpts = sorted(glob.glob(f"{CHECKPOINT_DIR}/adam_phase*.pt"), key=os.path.getmtime)
        if ckpts:
            try:
                print(f"Resuming from: {ckpts[-1]}")
                ckpt = torch.load(ckpts[-1], map_location="cuda")
                model.load_state_dict(ckpt["model"])
                start_phase = ckpt.get("phase", 1)
                start_step = ckpt.get("step", 0)
                print(f"   Model weights loaded from Phase {start_phase}, Step {start_step}")
            except Exception as e:
                print(f"Resume failed ({e}), starting fresh.")

    print(f"\nAdam Phoenix V6 - Curriculum Learning")
    print(f"Following: docs/private/Data Curation.md")

    # Determine which phases to run
    if args.phase > 0:
        phases_to_run = [args.phase]
    else:
        phases_to_run = [1, 2, 3]
        # If resuming, skip completed phases
        if args.resume and start_phase > 1:
            phases_to_run = [p for p in phases_to_run if p >= start_phase]

    # Run training phases
    for phase in phases_to_run:
        data_file = os.path.join(CURRICULUM_DATA_DIR, PHASE_FILES[phase])

        if not os.path.exists(data_file):
            print(f"Data file not found: {data_file}")
            print(f"Run data_forge.py first to generate curriculum data.")
            exit(1)

        # Only use start_step if resuming into this specific phase
        phase_start_step = start_step if (args.resume and phase == start_phase) else 0

        final_step, status = train_phase(model, tok, opt, phase, data_file, phase_start_step)

        if status == "divergence":
            print(f"\nPhase {phase} failed due to divergence. Aborting.")
            exit(2)

        # Reset optimizer momentum for next phase (fresh start)
        if phase < 3 and status == "complete":
            print(f"\nResetting optimizer momentum for Phase {phase + 1}...")
            for state in opt.state.values():
                if 'fast' in state: state['fast'].zero_()
                if 'slow' in state: state['slow'].zero_()

    print(f"\n{'='*70}")
    print("CURRICULUM TRAINING COMPLETE")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
