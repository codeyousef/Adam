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
from galore_torch import GaLoreAdamW8bit

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

# Reasoning Probes: Fixed prompts to track the evolution of logic over training
PROBES = [
    "If all <ORG> are <GPE>, and <PERSON> is an <ORG>, then <PERSON> is...",
    "To calculate the <CONCEPT> of a <OBJECT>, one must first derive the...",
    "Python: def solve(x): if x > 10: return <MASK> else: return",
]

# Training Hyperparameters
SAVE_EVERY_MINS = 30
GRAD_ACCUM = 16
LEARNING_RATE = 2e-5
MAX_SEQ_LEN = 1536
MIN_SEQ_LEN = 512
VALIDATION_INTERVAL = 500
MAX_CONSECUTIVE_OOM = 5


class AdamDataset(IterableDataset):
    def __init__(self, filepath, tokenizer, max_len):
        self.filepath = filepath
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        with open(self.filepath, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if worker_info and i % worker_info.num_workers != worker_info.id:
                    continue
                try:
                    text = json.loads(line)["text"]
                    enc = self.tokenizer(
                        text,
                        truncation=True,
                        max_length=self.max_len,
                        return_tensors="pt",
                    )
                    yield enc.input_ids.squeeze(0)
                except (json.JSONDecodeError, KeyError):
                    continue


def log_error(msg):
    """Append error to log file with timestamp."""
    with open(ERROR_LOG, "a") as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")


def init_research_logs():
    """Initialize CSV with research-grade column headers."""
    if not os.path.exists(TELEMETRY_FILE):
        with open(TELEMETRY_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "step", "loss", "entropy", "grad_norm",
                "tokens_per_sec", "vram_gb", "seq_len", "learning_rate"
            ])


def log_experiment_config(model, tokenizer):
    """Log experiment metadata for reproducibility."""
    config = {
        "experiment_start": datetime.now().isoformat(),
        "model_name": MODEL_NAME,
        "tokenizer_id": TOKENIZER_ID,
        "hyperparameters": {
            "learning_rate": LEARNING_RATE,
            "grad_accum": GRAD_ACCUM,
            "max_seq_len": MAX_SEQ_LEN,
            "min_seq_len": MIN_SEQ_LEN,
            "save_every_mins": SAVE_EVERY_MINS,
            "validation_interval": VALIDATION_INTERVAL,
        },
        "model_config": {
            "dtype": "bfloat16",
            "vocab_size": tokenizer.vocab_size,
            "num_parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        },
        "environment": {
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else None,
        },
        "probes": PROBES,
    }
    with open(EXPERIMENT_LOG, "w") as f:
        json.dump(config, f, indent=2)
    print(f"📋 Experiment config logged to {EXPERIMENT_LOG}")


def take_logic_snapshot(model, tokenizer, step):
    """Capture model's responses to fixed probes for tracking reasoning evolution."""
    model.eval()
    try:
        with open(SNAPSHOT_FILE, "a") as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"STEP {step} SNAPSHOT | {datetime.now().isoformat()}\n")
            f.write(f"{'='*60}\n")
            for i, probe in enumerate(PROBES, 1):
                inputs = tokenizer(probe, return_tensors="pt").input_ids.to("cuda")
                with torch.no_grad():
                    outputs = model.generate(inputs, max_new_tokens=30, do_sample=False)
                decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
                f.write(f"\nPROBE {i}: {probe}\n")
                f.write(f"RESPONSE: {decoded}\n")
            f.write("\n")
        print(f"📸 Logic snapshot saved at step {step}")
    except Exception as e:
        print(f"⚠️ Snapshot failed: {e}")
        log_error(f"Snapshot failed at step {step}: {e}")
    finally:
        model.train()


def safe_save(model, optimizer, step, loss, current_seq_len=MAX_SEQ_LEN):
    """Save checkpoint with atomic write to prevent corruption."""
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
    tmp_path = f"{CHECKPOINT_DIR}/tmp_adam.pt"
    torch.save({
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "loss": loss,
        "seq_len": current_seq_len,
        "timestamp": datetime.now().isoformat(),
    }, tmp_path)
    os.replace(tmp_path, f"{CHECKPOINT_DIR}/adam_ckpt_{step}.pt")
    print(f"💾 Checkpoint saved at step {step}")

    # Keep rolling window of checkpoints
    ckpts = sorted(glob.glob(f"{CHECKPOINT_DIR}/adam_ckpt_*.pt"), key=os.path.getmtime)
    while len(ckpts) > 3:
        os.remove(ckpts.pop(0))


def load_latest_checkpoint(model, optimizer):
    """Resume from latest checkpoint if available."""
    ckpts = sorted(glob.glob(f"{CHECKPOINT_DIR}/adam_ckpt_*.pt"), key=os.path.getmtime)
    if not ckpts:
        return 0, MAX_SEQ_LEN, 0  # step, seq_len, total_tokens
    
    latest = ckpts[-1]
    print(f"📂 Resuming from {latest}...")
    ckpt = torch.load(latest, map_location="cuda", weights_only=False)
    model.load_state_dict(ckpt["model"])
    if "optimizer" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except Exception:
            print("⚠️ Could not restore optimizer state, starting fresh")
    seq_len = ckpt.get("seq_len", MAX_SEQ_LEN)
    return ckpt["step"], seq_len, 0


def main():
    torch.set_float32_matmul_precision("high")
    init_research_logs()

    print(f"🐈 Catbelly Studio: Loading Adam's Architecture ({MODEL_NAME})...")

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    tokenizer.pad_token = tokenizer.eos_token

    model = MambaLMHeadModel.from_pretrained(MODEL_NAME, dtype=torch.bfloat16).to("cuda")

    # Log experiment configuration for reproducibility
    log_experiment_config(model, tokenizer)

    # Enable Gradient Checkpointing if available
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    else:
        print("⚠️ Gradient checkpointing not available. Skipping (safe for 2.7B).")
    model.train()

    # --- OPTIMIZER SETUP (GaLore for memory efficiency) ---
    galore_params = []
    standard_params = []
    for module_name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() == 2:
            galore_params.append(param)
        else:
            standard_params.append(param)

    print(f"   GaLore params: {len(galore_params)}, Standard params: {len(standard_params)}")

    param_groups = [
        {
            "params": galore_params,
            "rank": 1024,
            "update_proj_gap": 200,
            "scale": 0.25,
            "proj_type": "std",
        },
        {"params": standard_params},
    ]

    optimizer = GaLoreAdamW8bit(param_groups, lr=LEARNING_RATE)

    # --- RESUME FROM CHECKPOINT ---
    start_step, current_seq_len, total_tokens = load_latest_checkpoint(model, optimizer)
    
    dataset = AdamDataset(DATA_FILE, tokenizer, current_seq_len)
    loader = DataLoader(dataset, batch_size=1, num_workers=2, prefetch_factor=2)

    stop_signal = False
    consecutive_oom = 0

    def signal_handler(sig, frame):
        nonlocal stop_signal
        print("\n⚠️ Interrupt received, finishing current step and saving...")
        stop_signal = True

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    last_save = time.time()
    train_start_time = time.time()
    optimizer.zero_grad()
    current_loss = 0

    print(f">>> 🚀 RESEARCH TRAINING STARTED (Step {start_step}, seq_len={current_seq_len}). <<<")

    step = start_step
    for batch in loader:
        if stop_signal:
            break
        
        step += 1
        input_ids = None
        outputs = None
        logits = None
        loss = None
        
        try:
            input_ids = batch.to("cuda")
            
            # Dynamic truncation if seq_len was reduced
            if input_ids.shape[-1] > current_seq_len:
                input_ids = input_ids[:, :current_seq_len]

            total_tokens += input_ids.numel()

            outputs = model(input_ids)
            logits = outputs.logits

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )

            (loss / GRAD_ACCUM).backward()
            current_loss += loss.item() / GRAD_ACCUM

            if step % GRAD_ACCUM == 0:
                # Gradient norm (with clipping)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
                
                optimizer.step()
                optimizer.zero_grad()

                # --- RESEARCH TELEMETRY ---
                with torch.no_grad():
                    probs = torch.softmax(logits, dim=-1)
                    entropy = (
                        -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
                        .mean()
                        .item()
                    )
                    tokens_per_sec = total_tokens / (time.time() - train_start_time)
                    vram_gb = torch.cuda.max_memory_allocated() / 1e9

                # Log to research CSV
                with open(TELEMETRY_FILE, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        datetime.now().isoformat(),
                        step,
                        f"{current_loss:.6f}",
                        f"{entropy:.4f}",
                        f"{grad_norm:.4f}",
                        f"{tokens_per_sec:.1f}",
                        f"{vram_gb:.2f}",
                        current_seq_len,
                        LEARNING_RATE,
                    ])

                print(
                    f"S:{step} | L:{current_loss:.4f} | Ent:{entropy:.2f} | "
                    f"GN:{grad_norm:.2f} | TPS:{tokens_per_sec:.0f} | VRAM:{vram_gb:.1f}GB"
                )
                current_loss = 0
                consecutive_oom = 0

            # Logic snapshot at validation intervals
            if step > 0 and step % VALIDATION_INTERVAL == 0:
                take_logic_snapshot(model, tokenizer, step)

            if time.time() - last_save > (SAVE_EVERY_MINS * 60):
                safe_save(model, optimizer, step, loss.item(), current_seq_len)
                last_save = time.time()

        except torch.cuda.OutOfMemoryError:
            consecutive_oom += 1
            seq_info = f"seq_len={input_ids.shape[-1]}" if input_ids is not None else "unknown"
            print(f"⚠️ OOM at step {step} ({seq_info}). Count: {consecutive_oom}/{MAX_CONSECUTIVE_OOM}")
            log_error(f"OOM at step {step}, {seq_info}, consecutive={consecutive_oom}")
            
            # Aggressive cleanup
            del input_ids, outputs, logits, loss
            gc.collect()
            torch.cuda.empty_cache()
            optimizer.zero_grad()
            
            if consecutive_oom >= MAX_CONSECUTIVE_OOM and current_seq_len > MIN_SEQ_LEN:
                current_seq_len = max(MIN_SEQ_LEN, current_seq_len // 2)
                print(f"🔧 Reducing seq_len to {current_seq_len} due to repeated OOM")
                log_error(f"Reduced seq_len to {current_seq_len}")
                consecutive_oom = 0
            continue
            
        except Exception as e:
            error_msg = f"Unexpected error at step {step}: {type(e).__name__}: {e}"
            print(f"❌ {error_msg}")
            log_error(error_msg)
            log_error(traceback.format_exc())
            
            del input_ids, outputs, logits, loss
            gc.collect()
            torch.cuda.empty_cache()
            optimizer.zero_grad()
            continue

    # Final save and snapshot
    safe_save(model, optimizer, step, 0.0, current_seq_len)
    take_logic_snapshot(model, tokenizer, step)
    
    # Log final stats
    total_time = time.time() - train_start_time
    print(f"✅ Training complete at step {step}")
    print(f"   Total time: {total_time/3600:.2f} hours")
    print(f"   Total tokens: {total_tokens:,}")
    print(f"   Avg tokens/sec: {total_tokens/total_time:.1f}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user")
    except Exception as e:
        log_error(f"Fatal error: {type(e).__name__}: {e}")
        log_error(traceback.format_exc())
        print(f"💀 Fatal error: {e}")
        raise
