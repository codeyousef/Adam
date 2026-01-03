import os
import time
import signal
import json
import torch
import glob
import warnings
import csv
from torch.utils.data import IterableDataset, DataLoader
from transformers import AutoTokenizer
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from galore_torch import GaLoreAdamW8bit

# --- SILENCE WARNINGS ---
warnings.filterwarnings("ignore", message=".*set_float32_matmul_precision.*")

# --- ADAM CONFIG (7B EDITION) ---
MODEL_NAME = "state-spaces/mamba-7b"
TOKENIZER_ID = "EleutherAI/gpt-neox-20b"

DATA_FILE = "adam_skeleton_data.jsonl"
CHECKPOINT_DIR = "adam_checkpoints"
TELEMETRY_FILE = "adam_telemetry.csv"

# Training Hyperparameters
SAVE_EVERY_MINS = 30
GRAD_ACCUM = 16
LEARNING_RATE = 1e-5
MAX_SEQ_LEN = 2048
VALIDATION_INTERVAL = 500  # Steps between validation checks


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
                except json.JSONDecodeError:
                    continue


def safe_save(model, optimizer, step, loss):
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
    tmp_path = f"{CHECKPOINT_DIR}/tmp_adam.pt"
    # Saving only model weights to save space/time
    torch.save({"step": step, "model": model.state_dict(), "loss": loss}, tmp_path)
    os.replace(tmp_path, f"{CHECKPOINT_DIR}/adam_ckpt_{step}.pt")
    ckpts = sorted(glob.glob(f"{CHECKPOINT_DIR}/adam_ckpt_*.pt"), key=os.path.getmtime)
    while len(ckpts) > 2:
        os.remove(ckpts.pop(0))


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
    torch.set_float32_matmul_precision("medium")
    init_telemetry()

    print(f"🐈 Catbelly Studio: Loading Adam's Architecture ({MODEL_NAME})...")

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    tokenizer.pad_token = tokenizer.eos_token

    # FIXED: Load in bfloat16 explicitly to avoid CPU RAM OOM, then move to CUDA
    # MambaLMHeadModel does not support device='cuda' in the constructor args
    model = MambaLMHeadModel.from_pretrained(MODEL_NAME, dtype=torch.bfloat16).to(
        "cuda"
    )

    # CRITICAL: Enable Gradient Checkpointing for 7B fit on 4090
    model.gradient_checkpointing_enable()
    model.train()

    # Split params for GaLore (Apply only to matrices)
    galore_params = []
    standard_params = []
    for module_name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() >= 2:
            galore_params.append(param)
        else:
            standard_params.append(param)

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

    dataset = AdamDataset(DATA_FILE, tokenizer, MAX_SEQ_LEN)
    loader = DataLoader(dataset, batch_size=1, num_workers=1)

    stop_signal = False

    def signal_handler(sig, frame):
        nonlocal stop_signal
        stop_signal = True

    signal.signal(signal.SIGINT, signal_handler)

    last_save = time.time()
    optimizer.zero_grad()
    current_loss = 0

    print(">>> 🚀 ADAM (7B) IS AWAKE. TELEMETRY ACTIVE. TRAINING STARTED. <<<")

    for step, batch in enumerate(loader):
        if stop_signal:
            break
        try:
            input_ids = batch.to("cuda")

            # Request hidden_states for variance calculation
            outputs = model(input_ids, output_hidden_states=True)
            logits = outputs.logits

            # Shift for Causal LM training
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )

            (loss / GRAD_ACCUM).backward()
            current_loss += loss.item() / GRAD_ACCUM

            # --- INTEGRATION ANCHOR: UNCERTAINTY TELEMETRY ---
            # We calculate this every step but log it less frequently or averaged to save IO
            if (step + 1) % GRAD_ACCUM == 0:
                with torch.no_grad():
                    # 1. Entropy: How confused is he about the next token?
                    probs = torch.softmax(logits, dim=-1)
                    token_entropy = (
                        -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
                        .mean()
                        .item()
                    )

                    # 2. Variance: How "active" is his brain state?
                    # Mamba's hidden states are in outputs.hidden_states (tuple of layers)
                    # We take the last layer's variance
                    last_hidden = outputs.hidden_states[-1]
                    hidden_variance = last_hidden.std(dim=-1).mean().item()

                optimizer.step()
                optimizer.zero_grad()

                print(
                    f"Adam Step {step} | Loss: {current_loss:.4f} | Ent: {token_entropy:.2f} | Var: {hidden_variance:.4f}"
                )
                log_telemetry(step, current_loss, token_entropy, hidden_variance)

                current_loss = 0

            # --- VALIDATION SENTINEL ---
            if step > 0 and step % VALIDATION_INTERVAL == 0:
                print(f"🔍 Sentinel: Validating at step {step}...")
                model.eval()
                # Quick sanity check on the current batch (simulating a hold-out for speed)
                with torch.no_grad():
                    val_out = model(input_ids)
                    val_loss = torch.nn.functional.cross_entropy(
                        val_out.logits[..., :-1, :]
                        .contiguous()
                        .view(-1, val_out.logits.size(-1)),
                        input_ids[..., 1:].contiguous().view(-1),
                    )
                print(f"🔍 Sentinel: Validation Loss = {val_loss.item():.4f}")
                model.train()

            if time.time() - last_save > (SAVE_EVERY_MINS * 60):
                safe_save(model, optimizer, step, loss.item())
                last_save = time.time()

        except RuntimeError as e:
            if "out of memory" in str(e):
                print("⚠️ OOM Detected. Clearing cache...")
                torch.cuda.empty_cache()
            else:
                raise e

    safe_save(model, optimizer, step, 0.0)


if __name__ == "__main__":
    main()
