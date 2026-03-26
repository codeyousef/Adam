#!/usr/bin/env python3
"""DAFT Training Script for Format-Invariant Parametric Ignorance.

Trains a QLoRA-adapted Qwen model with Domain Adversarial Fine-Tuning
to achieve format-invariant parametric ignorance.

Key features:
- Gradient reversal for domain-invariant representations
- Lambda annealing (0.2 → 1.0) for stable training
- Stratified domain sampling for balanced batches
- NO EWC (it reinforces format-specific patterns)
"""

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, PeftModel

from daft_model import DAFTModel, DAFTConfig, compute_domain_accuracy


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Model
    base_model: str = "Qwen/Qwen2.5-Coder-3B-Instruct"
    simpo_checkpoint: Optional[str] = None  # Path to SimPO checkpoint to start from
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05

    # DAFT
    num_domains: int = 11  # 10 personas + 1 original
    lambda_initial: float = 0.1  # CRITICAL: Research says 0.1, not 0.2
    lambda_final: float = 1.0
    lambda_warmup_ratio: float = 0.3  # Fraction of training for lambda warmup

    # Training (H100 optimized defaults)
    max_steps: int = 3000
    batch_size: int = 8  # H100 80GB VRAM
    gradient_accumulation_steps: int = 4  # Effective batch = 32
    encoder_lr: float = 2e-4
    domain_lr: float = 1e-3
    warmup_ratio: float = 0.1
    max_length: int = 1024  # Longer sequences for H100
    eval_steps: int = 250
    save_steps: int = 500
    logging_steps: int = 50

    # H100 optimizations
    use_flash_attention: bool = True
    compile_model: bool = False  # torch.compile (experimental)
    bf16: bool = True

    # Data
    data_path: str = "hope/adam_training_data/adam_daft_ready.jsonl"
    output_dir: str = "hope/adam_daft_checkpoints"


class DAFTDataset(Dataset):
    """Dataset for DAFT training with domain labels."""

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 1024
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        self.domain_indices = {d: [] for d in range(11)}  # 11 personas

        print(f"Loading data from {data_path}")
        with open(data_path) as f:
            for i, line in enumerate(f):
                if line.strip():
                    ex = json.loads(line)
                    self.examples.append(ex)

                    # Track indices by domain
                    domain_id = ex.get("domain_id", 0)
                    self.domain_indices[domain_id].append(i)

        print(f"Loaded {len(self.examples)} examples")
        print("Domain distribution:")
        for d, indices in self.domain_indices.items():
            print(f"  D{d}: {len(indices)}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]

        # Format: instruction + input -> preferred
        prompt = f"<|im_start|>user\n{ex['instruction']}"
        if ex.get("input"):
            prompt += f"\n{ex['input']}"
        prompt += "<|im_end|>\n<|im_start|>assistant\n"

        response = ex["preferred"]
        full_text = prompt + response + "<|im_end|>"

        # Tokenize
        encoded = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        # Create labels (mask prompt, only train on response)
        prompt_encoded = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        prompt_len = prompt_encoded["input_ids"].shape[1]

        labels = input_ids.clone()
        labels[:prompt_len] = -100  # Mask prompt

        domain_id = ex.get("domain_id", 0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "domain_labels": torch.tensor(domain_id, dtype=torch.long),
        }

    def get_domain_sampler(self) -> WeightedRandomSampler:
        """Get sampler for balanced domain sampling."""
        # Calculate weights to balance domains
        domain_counts = {d: len(indices) for d, indices in self.domain_indices.items()}
        max_count = max(domain_counts.values())

        weights = []
        for i, ex in enumerate(self.examples):
            domain_id = ex.get("domain_id", 0)
            # Weight inversely proportional to domain frequency
            weights.append(max_count / max(domain_counts[domain_id], 1))

        return WeightedRandomSampler(
            weights=weights,
            num_samples=len(self.examples),
            replacement=True
        )


def create_model(config: TrainingConfig, device: str = "cuda"):
    """Create QLoRA model with DAFT wrapper."""

    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # H100 optimizations
    model_kwargs = {
        "quantization_config": bnb_config,
        "device_map": "auto",
        "trust_remote_code": True,
    }

    # Enable Flash Attention 2 for H100
    if config.use_flash_attention:
        model_kwargs["attn_implementation"] = "sdpa"
        print("Using SDPA attention (H100 optimized)")

    print(f"Loading base model: {config.base_model}")
    base_model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        **model_kwargs
    )

    # Enable gradient checkpointing to save memory
    base_model.gradient_checkpointing_enable()

    # Check if we should load from SimPO checkpoint
    if config.simpo_checkpoint:
        print(f"Loading LoRA weights from SimPO checkpoint: {config.simpo_checkpoint}")
        peft_model = PeftModel.from_pretrained(
            base_model,
            config.simpo_checkpoint,
            is_trainable=True
        )
    else:
        # Fresh LoRA config
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )

        print("Applying fresh LoRA adapters")
        peft_model = get_peft_model(base_model, lora_config)
    peft_model.print_trainable_parameters()

    # Wrap with DAFT
    daft_config = DAFTConfig(
        num_domains=config.num_domains,
        lambda_initial=config.lambda_initial,
        lambda_final=config.lambda_final,
        lambda_warmup_steps=int(config.max_steps * config.lambda_warmup_ratio),
        hidden_size=base_model.config.hidden_size,
    )

    print("Creating DAFT wrapper")
    daft_model = DAFTModel(peft_model, daft_config)

    # Move domain classifier to same device as model
    daft_model.domain_classifier.to(device)

    return daft_model


def train(config: TrainingConfig):
    """Main training loop."""

    # H100 optimizations
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device}")

    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create dataset and dataloader
    dataset = DAFTDataset(config.data_path, tokenizer, config.max_length)
    sampler = dataset.get_domain_sampler()

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=True,
    )

    # Create model
    model = create_model(config, device)

    # Optimizer with different LR for encoder vs domain classifier
    optimizer = torch.optim.AdamW([
        {"params": model.base_model.parameters(), "lr": config.encoder_lr},
        {"params": model.domain_classifier.parameters(), "lr": config.domain_lr},
    ])

    # LR scheduler
    num_training_steps = config.max_steps
    num_warmup_steps = int(num_training_steps * config.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # Training state
    global_step = 0
    best_domain_acc = 1.0  # Lower is better (want it near chance = 1/6)
    running_task_loss = 0.0
    running_domain_loss = 0.0
    running_domain_acc = 0.0

    # Training logs
    logs = []

    print("\n" + "=" * 60)
    print("STARTING DAFT TRAINING")
    print("=" * 60)
    print(f"Max steps: {config.max_steps}")
    print(f"Batch size: {config.batch_size} x {config.gradient_accumulation_steps}")
    print(f"Lambda annealing: {config.lambda_initial} → {config.lambda_final}")
    print(f"Target domain accuracy: ≤{100/config.num_domains:.1f}% (chance)")
    print("=" * 60 + "\n")

    model.train()
    data_iter = iter(dataloader)

    pbar = tqdm(total=config.max_steps, desc="Training")

    while global_step < config.max_steps:
        # Get batch (with cycling)
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        # Move to device
        batch = {k: v.to(device) for k, v in batch.items()}

        # Update lambda
        model.update_lambda(global_step)

        # Forward pass
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            domain_labels=batch["domain_labels"],
        )

        loss = outputs.total_loss / config.gradient_accumulation_steps
        loss.backward()

        # Track metrics
        running_task_loss += outputs.task_loss.item() if outputs.task_loss else 0
        running_domain_loss += outputs.domain_loss.item() if outputs.domain_loss else 0

        domain_acc = compute_domain_accuracy(
            outputs.domain_logits, batch["domain_labels"]
        )
        running_domain_acc += domain_acc

        # Gradient accumulation step
        if (global_step + 1) % config.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        global_step += 1
        pbar.update(1)

        # Logging
        if global_step % config.logging_steps == 0:
            avg_task_loss = running_task_loss / config.logging_steps
            avg_domain_loss = running_domain_loss / config.logging_steps
            avg_domain_acc = running_domain_acc / config.logging_steps

            log_entry = {
                "step": global_step,
                "task_loss": avg_task_loss,
                "domain_loss": avg_domain_loss,
                "domain_accuracy": avg_domain_acc,
                "lambda": model.lambda_value,
                "lr": scheduler.get_last_lr()[0],
            }
            logs.append(log_entry)

            pbar.set_postfix({
                "task": f"{avg_task_loss:.3f}",
                "dom": f"{avg_domain_loss:.3f}",
                "dom_acc": f"{avg_domain_acc*100:.1f}%",
                "λ": f"{model.lambda_value:.2f}",
            })

            running_task_loss = 0.0
            running_domain_loss = 0.0
            running_domain_acc = 0.0

        # Evaluation
        if global_step % config.eval_steps == 0:
            print(f"\n[Step {global_step}] Evaluation:")
            print(f"  Lambda: {model.lambda_value:.3f}")
            print(f"  Domain accuracy: {avg_domain_acc*100:.1f}% (target: ≤{100/config.num_domains:.1f}%)")

            # Track best (lowest domain accuracy)
            if avg_domain_acc < best_domain_acc:
                best_domain_acc = avg_domain_acc
                print(f"  New best domain accuracy!")

        # Save checkpoint
        if global_step % config.save_steps == 0:
            checkpoint_dir = os.path.join(config.output_dir, f"checkpoint-{global_step}")
            print(f"\nSaving checkpoint to {checkpoint_dir}")
            model.save_pretrained(checkpoint_dir)

            # Save training state
            torch.save({
                "step": global_step,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }, os.path.join(checkpoint_dir, "training_state.pt"))

    pbar.close()

    # Save final model
    final_dir = os.path.join(config.output_dir, "final")
    print(f"\nSaving final model to {final_dir}")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    # Save training logs
    with open(os.path.join(config.output_dir, "training_logs.json"), "w") as f:
        json.dump(logs, f, indent=2)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Final lambda: {model.lambda_value:.3f}")
    print(f"Best domain accuracy: {best_domain_acc*100:.1f}%")
    print(f"Target: ≤{100/config.num_domains:.1f}% (chance)")
    if best_domain_acc <= 1.5 / config.num_domains:
        print("SUCCESS: Domain accuracy near chance level!")
    else:
        print("WARNING: Domain accuracy still above chance - may need more training")


def main():
    parser = argparse.ArgumentParser(description="DAFT Training for Adam")
    parser.add_argument("--data", type=str, help="Path to training data JSONL")
    parser.add_argument("--output", type=str, help="Output directory")
    parser.add_argument("--base-model", type=str, help="Base model name/path")
    parser.add_argument("--simpo-checkpoint", type=str, help="Path to SimPO checkpoint to start from")
    parser.add_argument("--num-domains", type=int, default=11)
    parser.add_argument("--lambda-initial", type=float, default=0.1)  # Research: start at 0.1
    parser.add_argument("--lambda-final", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=4)  # H100 can handle 4
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--no-flash-attention", action="store_true", help="Disable Flash Attention 2")
    args = parser.parse_args()

    config = TrainingConfig()

    if args.data:
        config.data_path = args.data
    if args.output:
        config.output_dir = args.output
    if args.base_model:
        config.base_model = args.base_model
    if args.simpo_checkpoint:
        config.simpo_checkpoint = args.simpo_checkpoint
    if args.num_domains:
        config.num_domains = args.num_domains
    if args.lambda_initial:
        config.lambda_initial = args.lambda_initial
    if args.lambda_final:
        config.lambda_final = args.lambda_final
    if args.steps:
        config.max_steps = args.steps
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.no_flash_attention:
        config.use_flash_attention = False
    if args.lr:
        config.encoder_lr = args.lr

    train(config)


if __name__ == "__main__":
    main()
