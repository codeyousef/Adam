#!/usr/bin/env python3
"""
Adam PoC — Stage 1: Skill Acquisition (SFT)
Follows Section 3a / 7.1 of the research paper exactly.

Method : Curriculum-ordered SFT with verifier augmentation (L3/L4)
Base   : adam_poc_checkpoints/checkpoint-244963/ (from-scratch pretrain)
Full FT: No LoRA — model is 494M and fully owned
LR     : 2e-5
Batch  : 8 (H200)
Epochs : 2 per curriculum phase
Label  : smoothing 0.1
CoT    : Conditional — L2 rung3 and L3 only

Curriculum (12 phases):
  L1: simple_override → nested_premise → conflicting_facts
  L2: rung1 → rung2 → rung3
  L3: valid → invalid → unknown
  L4: single → multiple → nested

Stopping: per-difficulty validation threshold; rollback on failure.
"""

import os, sys, json, time, math, shutil, argparse
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

import torch
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from transformers import TrainerCallback

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SFTConfig:
    # Paths
    pretrain_checkpoint: str = "adam_poc_checkpoints/checkpoint-244963"
    tokenizer_name: str     = "Qwen/Qwen2.5-Coder-3B-Instruct"
    sft_data_dir: str       = "hope/adam_training_data/sft"
    output_dir: str         = "adam_poc_sft_checkpoints"

    # Training
    learning_rate: float    = 2e-5   # PDF Section 3a prescription; 5e-4 caused format memorization
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 4   # effective batch = 32
    num_train_epochs: int   = 3     # PDF says 2; +1 for joint training coverage
    max_seq_length: int     = 2048
    label_smoothing: float  = 0.1
    warmup_ratio: float     = 0.05
    weight_decay: float     = 0.01

    # Hardware
    bf16: bool              = True
    gradient_checkpointing: bool = True
    dataloader_num_workers: int  = 4

    # Logging / checkpointing
    logging_steps: int      = 20
    save_steps: int         = 200
    eval_steps: int         = 200

    # Curriculum thresholds — rollback if below after phase
    # Loose: we're just establishing task competence, not final accuracy
    thresholds: dict = field(default_factory=lambda: {
        "L1": {"simple_override": 0.50, "nested_premise": 0.50, "conflicting_facts": 0.45},
        "L2": {"rung1": 0.45, "rung2": 0.40, "rung3": 0.35},
        "L3": {"valid": 0.45, "invalid": 0.40, "unknown": 0.40},
        "L4": {"single": 0.50, "multiple": 0.45, "nested": 0.40},
    })

    # Curriculum order
    curriculum: list = field(default_factory=lambda: [
        ("L1", "simple_override"),
        ("L1", "nested_premise"),
        ("L1", "conflicting_facts"),
        ("L1", "entity_override"),
        ("L2", "rung1"),
        ("L2", "rung2"),
        ("L2", "rung3"),
        ("L3", "valid"),
        ("L3", "invalid"),
        ("L3", "unknown"),
        ("L4", "single"),
        ("L4", "multiple"),
        ("L4", "nested"),
    ])


# ---------------------------------------------------------------------------
# Data loading and formatting
# ---------------------------------------------------------------------------

def load_curriculum_data(sft_data_dir: str, level: str, difficulty: str) -> Dataset:
    """Load a single curriculum phase JSONL file."""
    path = Path(sft_data_dir) / f"adam_sft_{level}_{difficulty}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing SFT data: {path}")
    examples = []
    with open(path) as f:
        for line in f:
            examples.append(json.loads(line))
    return Dataset.from_list(examples)


def format_example(example, tokenizer, max_length):
    """
    Tokenize plain-text SFT examples (format: "{question}\n{answer}").
    Compute CLM loss on all tokens — no prompt masking.

    Rationale: our from-scratch model never saw Qwen2.5 chat format tokens
    (<|im_start|>, <|im_end|>). Using apply_chat_template causes loss to be
    dominated by unseen special tokens and plateau at ~44. Plain text format
    matches validation_probes.py exactly (probes use raw tokenizer(text)).
    <|begin_of_thought|>/<|end_of_thought|> are safe — model learned them in pretraining.
    """
    ids = tokenizer(
        example["text"],
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )["input_ids"]

    return {
        "input_ids":      ids,
        "attention_mask": [1] * len(ids),
        "labels":         list(ids),
    }


class TokenizedDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, max_length, label_smoothing=0.0):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_smoothing = label_smoothing

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return format_example(self.data[idx], self.tokenizer, self.max_length)


def make_data_collator(tokenizer, max_length):
    """Collate with left-padding to max batch length."""
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    def collate(batch):
        max_len = max(len(b["input_ids"]) for b in batch)
        input_ids      = []
        attention_mask = []
        labels         = []
        for b in batch:
            pad_n = max_len - len(b["input_ids"])
            input_ids.append(      b["input_ids"]      + [pad_id] * pad_n)
            attention_mask.append( b["attention_mask"] + [0]      * pad_n)
            labels.append(         b["labels"]         + [-100]   * pad_n)
        return {
            "input_ids":       torch.tensor(input_ids,      dtype=torch.long),
            "attention_mask":  torch.tensor(attention_mask, dtype=torch.long),
            "labels":          torch.tensor(labels,         dtype=torch.long),
        }
    return collate


# ---------------------------------------------------------------------------
# Per-phase validation (keyword extraction from model output)
# ---------------------------------------------------------------------------

def _extract_prompt(text: str, level: str) -> str:
    """
    Extract the question (prompt) portion from a plain-text SFT example.
    Each answer begins with a level-specific marker; we split at the first occurrence.
    """
    # L2 rung3 uses thought tags; rung1/rung2 use CONSTANTS
    markers = {
        "L1": ["\nCONTEXT_SAYS:"],
        "L2": ["\n<|begin_of_thought|>", "\nCONSTANTS:"],
        "L3": ["\n<|begin_of_thought|>"],
        "L4": ["\nCONSTRAINT:"],
    }
    for marker in markers.get(level, []):
        if marker in text:
            return text.split(marker)[0]
    return text  # fallback: treat full text as prompt


def quick_validate(model, tokenizer, level: str, difficulty: str,
                   sft_data_dir: str, n_samples: int = 50,
                   device: str = "cuda") -> float:
    """
    Fast in-training validation: sample n examples, check output format correctness.
    Uses plain-text prompts (no chat template) — matches validation_probes.py format.
    """
    path = Path(sft_data_dir) / f"adam_sft_{level}_{difficulty}.jsonl"
    examples = []
    with open(path) as f:
        for line in f:
            examples.append(json.loads(line))

    import random
    samples = random.sample(examples, min(n_samples, len(examples)))

    model.eval()
    correct = 0

    keywords = {
        "L1": ["CONTEXT_SAYS", "OVERRIDE", "ANSWER"],
        "L2": ["CONSTANTS", "RESULT"],
        "L3": ["FORM", "VALIDITY", "REASON"],
        "L4": ["CONSTRAINT", "APPROACH", "CODE"],
    }[level]

    with torch.no_grad():
        for ex in samples:
            prompt = _extract_prompt(ex["text"], level)
            ids = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1800,
                add_special_tokens=False,
            ).to(device)
            out = model.generate(
                **ids,
                max_new_tokens=256,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
            generated = tokenizer.decode(
                out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True
            )
            if all(kw in generated for kw in keywords):
                correct += 1

    model.train()
    return correct / len(samples)


# ---------------------------------------------------------------------------
# Label smoothing trainer
# ---------------------------------------------------------------------------

class LabelSmoothingTrainer(Trainer):
    """
    Trainer that passes labels directly to the model's forward() so the model
    computes CLM loss internally. Label smoothing is configured via TrainingArguments.
    (HF Trainer supports label_smoothing_factor natively for CausalLM.)
    """

    def __init__(self, label_smoothing=0.1, **kwargs):
        super().__init__(**kwargs)
        self.label_smoothing_value = label_smoothing

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Pass labels to model — HF CausalLM handles the CLM shift internally
        outputs = model(**inputs)
        loss = outputs.loss
        if loss is None:
            raise ValueError("Model did not return a loss. Ensure labels are in inputs.")
        # Apply label smoothing manually if needed (HF does it via LabelSmoother)
        return (loss, outputs) if return_outputs else loss


# ---------------------------------------------------------------------------
# Progress logger callback
# ---------------------------------------------------------------------------

class PhaseLogger(TrainerCallback):
    def __init__(self, phase_name: str, log_file: str):
        self.phase = phase_name
        self.log_file = log_file

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        loss = logs.get("loss", 0)
        lr   = logs.get("learning_rate", 0)
        step = state.global_step
        line = f"[{self.phase}] step={step:>6d}  loss={loss:.4f}  lr={lr:.2e}"
        print(line)
        with open(self.log_file, "a") as f:
            f.write(line + "\n")


# ---------------------------------------------------------------------------
# Main curriculum training loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain-checkpoint", default="adam_poc_checkpoints/checkpoint-244963")
    parser.add_argument("--sft-data-dir",        default="hope/adam_training_data/sft")
    parser.add_argument("--output-dir",          default="adam_poc_sft_checkpoints")
    parser.add_argument("--tokenizer",           default="Qwen/Qwen2.5-Coder-3B-Instruct")
    parser.add_argument("--epochs",              type=int, default=3)
    parser.add_argument("--batch-size",          type=int, default=8)
    parser.add_argument("--lr",                  type=float, default=2e-5)
    parser.add_argument("--label-smoothing",     type=float, default=0.1)
    parser.add_argument("--max-seq-length",      type=int, default=2048)
    parser.add_argument("--skip-to",             default=None,
                        help="Skip to phase e.g. 'L3_valid'")
    parser.add_argument("--no-rollback",         action="store_true",
                        help="Continue even if below threshold")
    parser.add_argument("--joint",               action="store_true",
                        help="Train on all phases jointly (prevents catastrophic forgetting)")
    args = parser.parse_args()

    cfg = SFTConfig()
    cfg.pretrain_checkpoint = args.pretrain_checkpoint
    cfg.sft_data_dir        = args.sft_data_dir
    cfg.output_dir          = args.output_dir
    cfg.tokenizer_name      = args.tokenizer
    cfg.num_train_epochs    = args.epochs
    cfg.per_device_train_batch_size = args.batch_size
    cfg.learning_rate       = args.lr
    cfg.label_smoothing     = args.label_smoothing
    cfg.max_seq_length      = args.max_seq_length

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = str(out_dir / "sft_training.log")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------------------------------------------
    # Load tokenizer
    # ------------------------------------------------------------------
    print(f"Loading tokenizer from {cfg.tokenizer_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ------------------------------------------------------------------
    # Load model (from-scratch pretrained checkpoint — no PEFT)
    # ------------------------------------------------------------------
    print(f"Loading pretrained model from {cfg.pretrain_checkpoint} ...")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.pretrain_checkpoint,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.to(device)

    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {param_count/1e6:.1f}M parameters")
    print(f"Training on: {device}")

    # ------------------------------------------------------------------
    # Joint training mode: all phases concatenated (prevents catastrophic forgetting)
    # ------------------------------------------------------------------
    if args.joint:
        print("\n" + "="*60)
        print("JOINT TRAINING MODE — all 12 phases concatenated")
        print("="*60)
        all_examples = []
        for level, difficulty in cfg.curriculum:
            path = Path(cfg.sft_data_dir) / f"adam_sft_{level}_{difficulty}.jsonl"
            if not path.exists():
                print(f"WARNING: missing {path}, skipping")
                continue
            with open(path) as f:
                for line in f:
                    all_examples.append(json.loads(line))
        import random as _rng
        _rng.shuffle(all_examples)
        print(f"Total joint examples: {len(all_examples)}")

        from datasets import Dataset as _DS
        joint_dataset = _DS.from_list(all_examples)
        tok_dataset = TokenizedDataset(joint_dataset, tokenizer, cfg.max_seq_length)
        collator = make_data_collator(tokenizer, cfg.max_seq_length)

        joint_out = str(out_dir / "joint_training")
        train_args = TrainingArguments(
            output_dir=joint_out,
            num_train_epochs=cfg.num_train_epochs,
            per_device_train_batch_size=cfg.per_device_train_batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            learning_rate=cfg.learning_rate,
            warmup_ratio=cfg.warmup_ratio,
            weight_decay=cfg.weight_decay,
            bf16=cfg.bf16,
            gradient_checkpointing=False,
            logging_steps=cfg.logging_steps,
            save_steps=cfg.save_steps,
            save_total_limit=2,
            dataloader_num_workers=cfg.dataloader_num_workers,
            remove_unused_columns=False,
            report_to="none",
            lr_scheduler_type="cosine",
            max_grad_norm=10.0,
            label_smoothing_factor=cfg.label_smoothing,
        )
        trainer = LabelSmoothingTrainer(
            label_smoothing=cfg.label_smoothing,
            model=model,
            args=train_args,
            train_dataset=tok_dataset,
            data_collator=collator,
            callbacks=[PhaseLogger("joint", log_file)],
        )
        trainer.train()
        del trainer
        torch.cuda.empty_cache()

        final_dir = out_dir / "final"
        print(f"\nSaving final model to {final_dir} ...")
        model.save_pretrained(str(final_dir))
        tokenizer.save_pretrained(str(final_dir))
        print("Joint training complete. Run evaluate_poc.py for full probe results.")
        return

    # ------------------------------------------------------------------
    # Curriculum loop
    # ------------------------------------------------------------------
    phase_results = {}
    best_checkpoint_dir = None
    skip_to = args.skip_to  # e.g. "L3_valid"
    skipping = skip_to is not None

    for phase_idx, (level, difficulty) in enumerate(cfg.curriculum):
        phase_name = f"{level}_{difficulty}"

        if skipping:
            if phase_name == skip_to:
                skipping = False
                # Load the checkpoint saved before this phase
                prev_ckpt = out_dir / f"phase_{phase_idx:02d}_start"
                if prev_ckpt.exists():
                    print(f"Resuming: loading checkpoint from {prev_ckpt}")
                    model = AutoModelForCausalLM.from_pretrained(
                        str(prev_ckpt), torch_dtype=torch.bfloat16,
                        trust_remote_code=True, attn_implementation="sdpa"
                    ).to(device)
                else:
                    print(f"No start checkpoint for {phase_name}, using current model")
            else:
                print(f"Skipping phase {phase_name}")
                continue

        print(f"\n{'='*60}")
        print(f"PHASE {phase_idx+1}/12: {level} — {difficulty}")
        print(f"{'='*60}")

        # Save model state before this phase (for rollback)
        phase_start_dir = out_dir / f"phase_{phase_idx:02d}_start"
        model.save_pretrained(str(phase_start_dir))
        tokenizer.save_pretrained(str(phase_start_dir))

        # Load data
        try:
            dataset = load_curriculum_data(cfg.sft_data_dir, level, difficulty)
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            print("Run data/data_forge_sft.py first to generate SFT data.")
            sys.exit(1)

        print(f"Loaded {len(dataset)} examples for {phase_name}")

        # Tokenize
        tok_dataset = TokenizedDataset(
            dataset,
            tokenizer,
            cfg.max_seq_length,
            label_smoothing=cfg.label_smoothing,
        )
        collator = make_data_collator(tokenizer, cfg.max_seq_length)

        # Training args
        phase_out = str(out_dir / f"phase_{phase_idx:02d}_{phase_name}")
        train_args = TrainingArguments(
            output_dir=phase_out,
            num_train_epochs=cfg.num_train_epochs,
            per_device_train_batch_size=cfg.per_device_train_batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            learning_rate=cfg.learning_rate,
            warmup_ratio=cfg.warmup_ratio,
            weight_decay=cfg.weight_decay,
            bf16=cfg.bf16,
            gradient_checkpointing=False,   # we set it manually above
            logging_steps=cfg.logging_steps,
            save_steps=cfg.save_steps,
            save_total_limit=1,
            dataloader_num_workers=cfg.dataloader_num_workers,
            remove_unused_columns=False,
            report_to="none",
            lr_scheduler_type="cosine",
            max_grad_norm=10.0,  # allow large grads during initial SFT adaptation
            label_smoothing_factor=cfg.label_smoothing,
        )

        trainer = LabelSmoothingTrainer(
            label_smoothing=cfg.label_smoothing,
            model=model,
            args=train_args,
            train_dataset=tok_dataset,
            data_collator=collator,
            callbacks=[PhaseLogger(phase_name, log_file)],
        )

        t0 = time.time()
        trainer.train()
        elapsed = time.time() - t0
        print(f"Phase {phase_name} training done in {elapsed/60:.1f} min")

        # ------------------------------------------------------------------
        # Per-phase validation
        # ------------------------------------------------------------------
        print(f"Validating {phase_name} ...")
        acc = quick_validate(
            model, tokenizer, level, difficulty,
            cfg.sft_data_dir, n_samples=80, device=device
        )
        threshold = cfg.thresholds[level][difficulty]

        result = {
            "phase":     phase_name,
            "accuracy":  acc,
            "threshold": threshold,
            "passed":    acc >= threshold,
            "elapsed_s": elapsed,
        }
        phase_results[phase_name] = result

        with open(log_file, "a") as f:
            f.write(f"\nVALIDATION {phase_name}: acc={acc:.3f} threshold={threshold:.3f} "
                    f"{'PASS' if result['passed'] else 'FAIL'}\n\n")

        print(f"  Accuracy: {acc:.1%} (threshold: {threshold:.0%}) — "
              f"{'PASS ✓' if result['passed'] else 'FAIL ✗'}")

        if not result["passed"] and not args.no_rollback:
            print(f"  Below threshold — rolling back to phase start checkpoint")
            # Restore model from before this phase
            del trainer
            torch.cuda.empty_cache()
            model = AutoModelForCausalLM.from_pretrained(
                str(phase_start_dir),
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                attn_implementation="sdpa",
            ).to(device)
            if cfg.gradient_checkpointing:
                model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            with open(log_file, "a") as f:
                f.write(f"ROLLBACK: restored from {phase_start_dir}\n\n")
        else:
            # Keep this checkpoint as the new best
            best_checkpoint_dir = phase_out
            # Clean up start checkpoint to save disk space
            if phase_start_dir.exists() and phase_idx > 0:
                shutil.rmtree(str(phase_start_dir), ignore_errors=True)

        # Flush GPU
        if "trainer" in dir():
            del trainer
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Save final model
    # ------------------------------------------------------------------
    final_dir = out_dir / "final"
    print(f"\nSaving final model to {final_dir} ...")
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    # ------------------------------------------------------------------
    # Print curriculum summary
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("CURRICULUM SUMMARY")
    print(f"{'='*60}")
    all_passed = True
    for phase_name, result in phase_results.items():
        status = "PASS" if result["passed"] else "FAIL"
        print(f"  {phase_name:<25s}  {result['accuracy']:.1%}  [{status}]  "
              f"(threshold {result['threshold']:.0%})")
        if not result["passed"]:
            all_passed = False

    print(f"\nFinal model: {final_dir}")

    # ------------------------------------------------------------------
    # Run full L1-L4 probe evaluation
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("RUNNING FULL L1-L4 VALIDATION PROBES")
    print("="*60)
    try:
        sys.path.insert(0, ".")
        from validation_probes import run_validation
        report = run_validation(model, tokenizer, levels=[1, 2, 3, 4], device=device)

        print(f"\nL1 (Context Override):    {report.level1_accuracy:.1%}")
        print(f"L2 (Physics Override):    {report.level2_accuracy:.1%}")
        print(f"L3 (Syllogistic Logic):   {report.level3_accuracy:.1%}")
        print(f"L4 (Code Constraints):    {report.level4_accuracy:.1%}")
        print(f"Overall:                  {report.overall_accuracy:.1%}")

        targets = {"L1": 0.70, "L2": 0.35, "L3": 0.60, "L4": 0.75}
        print("\nPoC Targets:")
        print(f"  L1 ≥70%: {'✓' if report.level1_accuracy >= 0.70 else '✗'}")
        print(f"  L2 ≥35%: {'✓' if report.level2_accuracy >= 0.35 else '✗'}")
        print(f"  L3 ≥60%: {'✓' if report.level3_accuracy >= 0.60 else '✗'}")
        print(f"  L4 ≥75%: {'✓' if report.level4_accuracy >= 0.75 else '✗'}")

        # Save results
        results_path = out_dir / "sft_eval_results.json"
        with open(results_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"\nFull results saved to {results_path}")

    except ImportError:
        print("validation_probes.py not found in path — skipping full probe evaluation.")
        print(f"Run: python evaluate_poc.py --checkpoint {final_dir}")

    print("\nSFT Stage 1 complete.")


if __name__ == "__main__":
    main()
