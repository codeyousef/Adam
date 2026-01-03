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
from torch.utils.data import IterableDataset, DataLoader
from transformers import AutoTokenizer
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from galore_torch import GaLoreAdamW8bit

# --- SILENCE WARNINGS ---
warnings.filterwarnings("ignore", message=".*set_float32_matmul_precision.*")

# --- ADAM CONFIG (2.7B PRODUCTION) ---
# VALIDATED: This is the largest official Mamba-2 checkpoint compatible with our setup.
MODEL_NAME = "state-spaces/mamba2-2.7b"
TOKENIZER_ID = "EleutherAI/gpt-neox-20b"

DATA_FILE = "adam_skeleton_data.jsonl"
CHECKPOINT_DIR = "adam_checkpoints"
TELEMETRY_FILE = "adam_telemetry.csv"
ERROR_LOG = "adam_errors.log"

# Training Hyperparameters
SAVE_EVERY_MINS = 30
GRAD_ACCUM = 16
LEARNING_RATE = 2e-5
MAX_SEQ_LEN = 1536  # Reduced from 2048 - hitting OOM too frequently
MIN_SEQ_LEN = 512  # Fallback when OOM
VALIDATION_INTERVAL = 500
MAX_CONSECUTIVE_OOM = 5  # Reduce seq_len after this many OOMs


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


def safe_save(model, optimizer, step, loss, current_seq_len=MAX_SEQ_LEN):
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
    tmp_path = f"{CHECKPOINT_DIR}/tmp_adam.pt"
    torch.save({
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "loss": loss,
        "seq_len": current_seq_len,
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
        return 0, MAX_SEQ_LEN
    
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
    return ckpt["step"], seq_len


def init_telemetry():
    if not os.path.exists(TELEMETRY_FILE):
        with open(TELEMETRY_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "loss", "token_entropy", "hidden_variance"])


def log_telemetry(step, loss, entropy, variance):
    with open(TELEMETRY_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([step, loss, entropy, variance])


def main():
    # Silence TF32 deprecation warning - use string precision settings
    torch.set_float32_matmul_precision("high")  # Uses TF32 where beneficial
    init_telemetry()

    print(f"🐈 Catbelly Studio: Loading Adam's Architecture ({MODEL_NAME})...")

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    tokenizer.pad_token = tokenizer.eos_token

    # Load in bfloat16 directly to GPU (2.7B is small enough to load fast)
    model = MambaLMHeadModel.from_pretrained(MODEL_NAME, dtype=torch.bfloat16).to(
        "cuda"
    )

    # Enable Gradient Checkpointing if available (standard Mamba may not have this)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    else:
        print("⚠️ Gradient checkpointing not available. Continuing without it.")
    model.train()

    # --- OPTIMIZER SETUP (Fixed for GaLore) ---
    # GaLore only works with 2D matrices, not 3D+ tensors
    galore_params = []
    standard_params = []
    for module_name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # GaLore requires exactly 2D tensors (matrices)
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
    start_step, current_seq_len = load_latest_checkpoint(model, optimizer)
    
    # Rebuild dataset with current seq_len (may have been reduced due to OOM)
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
    optimizer.zero_grad()
    current_loss = 0

    print(f">>> 🚀 ADAM (2.7B) IS AWAKE. TRAINING STARTED (seq_len={current_seq_len}). <<<")

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
            
            # Truncate to current_seq_len if batch is longer
            if input_ids.shape[-1] > current_seq_len:
                input_ids = input_ids[:, :current_seq_len]

            # Standard Mamba does not support output_hidden_states
            outputs = model(input_ids)
            logits = outputs.logits

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )

            (loss / GRAD_ACCUM).backward()
            current_loss += loss.item() / GRAD_ACCUM

            if (step + 1) % GRAD_ACCUM == 0:
                # --- TELEMETRY ---
                with torch.no_grad():
                    probs = torch.softmax(logits, dim=-1)
                    token_entropy = (
                        -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
                        .mean()
                        .item()
                    )
                    # Hidden variance disabled - standard Mamba doesn't expose hidden states
                    hidden_variance = 0.0

                optimizer.step()
                optimizer.zero_grad()

                print(
                    f"Adam Step {step} | Loss: {current_loss:.4f} | Ent: {token_entropy:.2f} | Var: {hidden_variance:.4f}"
                )
                log_telemetry(step, current_loss, token_entropy, hidden_variance)
                current_loss = 0

            # Sentinel Validation
            if step > 0 and step % VALIDATION_INTERVAL == 0:
                print(f"🔍 Sentinel: Validating at step {step}...")
                model.eval()
                with torch.no_grad():
                    # Quick check on current batch to ensure no collapse
                    val_out = model(input_ids)
                    val_loss = torch.nn.functional.cross_entropy(
                        val_out.logits[..., :-1, :]
                        .contiguous()
                        .view(-1, val_out.logits.size(-1)),
                        input_ids[..., 1:].contiguous().view(-1),
                    )
                print(f"🔍 Sentinel: Loss = {val_loss.item():.4f}")
                model.train()

            if time.time() - last_save > (SAVE_EVERY_MINS * 60):
                safe_save(model, optimizer, step, loss.item(), current_seq_len)
                last_save = time.time()
            
            # Reset OOM counter on successful step
            consecutive_oom = 0

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
            
            # Reduce seq_len if too many consecutive OOMs
            if consecutive_oom >= MAX_CONSECUTIVE_OOM and current_seq_len > MIN_SEQ_LEN:
                current_seq_len = max(MIN_SEQ_LEN, current_seq_len // 2)
                print(f"🔧 Reducing seq_len to {current_seq_len} due to repeated OOM")
                log_error(f"Reduced seq_len to {current_seq_len}")
                consecutive_oom = 0
            
            continue
            
        except Exception as e:
            # Log unexpected errors but keep running
            error_msg = f"Unexpected error at step {step}: {type(e).__name__}: {e}"
            print(f"❌ {error_msg}")
            log_error(error_msg)
            log_error(traceback.format_exc())
            
            # Cleanup and continue
            del input_ids, outputs, logits, loss
            gc.collect()
            torch.cuda.empty_cache()
            optimizer.zero_grad()
            continue

    # Final save
    safe_save(model, optimizer, step, 0.0, current_seq_len)
    print(f"✅ Training complete at step {step}")


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
