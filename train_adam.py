import os
import time
import signal
import json
import torch
import glob
import warnings
from torch.utils.data import IterableDataset, DataLoader
from transformers import AutoTokenizer
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from galore_torch import GaLoreAdamW8bit

# --- SILENCE WARNINGS ---
warnings.filterwarnings("ignore", message=".*set_float32_matmul_precision.*")

# --- ADAM CONFIG (7B EDITION) ---
# UPDATED: Using the 7B parameter model
MODEL_NAME = "state-spaces/mamba-7b"
# UPDATED: Use the tokenizer that matches the 7B model structure
TOKENIZER_ID = "EleutherAI/gpt-neox-20b"

DATA_FILE = "adam_skeleton_data.jsonl"
CHECKPOINT_DIR = "adam_checkpoints"
SAVE_EVERY_MINS = 30  # Increased save time because 7B checkpoints are larger (~14GB)
GRAD_ACCUM = 16  # Increased accumulation to simulate larger batches on 4090
LEARNING_RATE = 1e-5  # Lower LR for larger model/longer training
MAX_SEQ_LEN = 2048


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
    # We save only the model weights to save space/time, optimizer state is huge for 7B
    torch.save({"step": step, "model": model.state_dict(), "loss": loss}, tmp_path)
    os.replace(tmp_path, f"{CHECKPOINT_DIR}/adam_ckpt_{step}.pt")
    # Keep only 2 recent checkpoints due to size (14GB each)
    ckpts = sorted(glob.glob(f"{CHECKPOINT_DIR}/adam_ckpt_*.pt"), key=os.path.getmtime)
    while len(ckpts) > 2:
        os.remove(ckpts.pop(0))


def main():
    torch.set_float32_matmul_precision("medium")
    print(f"🐈 Catbelly Studio: Loading Adam's Architecture ({MODEL_NAME})...")

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    tokenizer.pad_token = tokenizer.eos_token

    # Load model in bfloat16 to fit in VRAM
    model = MambaLMHeadModel.from_pretrained(
        MODEL_NAME, device="cuda", dtype=torch.bfloat16
    )

    # CRITICAL: Enable Gradient Checkpointing.
    # Without this, a 7B model will OOM on a 24GB card during training.
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
    # Lower workers to save RAM for the massive data processing
    loader = DataLoader(dataset, batch_size=1, num_workers=1)

    stop_signal = False

    def signal_handler(sig, frame):
        nonlocal stop_signal
        stop_signal = True

    signal.signal(signal.SIGINT, signal_handler)

    last_save = time.time()
    optimizer.zero_grad()
    current_loss = 0

    print(">>> 🚀 ADAM (7B) IS AWAKE. LONG-HAUL TRAINING STARTED. <<<")
    for step, batch in enumerate(loader):
        if stop_signal:
            break
        try:
            input_ids = batch.to("cuda")
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
                optimizer.step()
                optimizer.zero_grad()
                print(f"Adam Step {step} | Loss: {current_loss:.4f}")
                current_loss = 0

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
