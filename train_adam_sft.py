#!/usr/bin/env python3
"""
Adam Phase 1: Counterfactual SFT Training

Base Model: Qwen2.5-Coder-3B-Instruct
Method: QLoRA (4-bit quantization)
Target: 5K-10K steps

This script trains the base model on counterfactual data to establish
context-adherence behavior before preference optimization.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass

import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer, SFTConfig

# Local imports
from validation_probes import (
    run_validation,
    CPIDetector,
    check_abort_criteria,
    ValidationReport,
    LEVEL1_PROBES,
    LEVEL2_PROBES,
    LEVEL3_PROBES,
    LEVEL4_PROBES,
)
from paper_metrics import (
    PaperMetricsCollector,
    get_gpu_metrics,
    extract_sample_outputs_from_report,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class TrainingConfig:
    # Model
    base_model: str = "Qwen/Qwen2.5-Coder-3B-Instruct"

    # Data
    data_path: str = "adam_training_data/adam_sft_data.jsonl"
    max_seq_length: int = 2048

    # LoRA
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: list = None  # Set in __post_init__

    # Training
    learning_rate: float = 2e-5
    batch_size: int = 4
    gradient_accumulation_steps: int = 32  # Effective batch = 128
    num_epochs: int = 3
    max_steps: int = 10000  # Override epochs if set
    warmup_steps: int = 500
    weight_decay: float = 0.01

    # Checkpointing
    output_dir: str = "adam_sft_checkpoints"
    save_steps: int = 250
    eval_steps: int = 250
    logging_steps: int = 10

    # Validation
    validation_steps: int = 500  # Run probes every N steps
    abort_on_cpi: bool = True  # Abort if Context-Parametric Inversion detected

    # Hardware
    device: str = "cuda"
    bf16: bool = True
    gradient_checkpointing: bool = True

    # HuggingFace (reads from environment if not set)
    hf_repo_id: str = os.environ.get("HF_REPO_ID", "")
    hf_token: str = os.environ.get("HF_TOKEN", "")

    def __post_init__(self):
        if self.lora_target_modules is None:
            # Qwen2.5 target modules
            self.lora_target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ]


def load_config(config_path: str = None) -> TrainingConfig:
    """Load config from JSON file or use defaults."""
    config = TrainingConfig()

    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            overrides = json.load(f)
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)

    return config


# =============================================================================
# DATA PREPARATION
# =============================================================================

def format_sample(sample: dict, tokenizer) -> str:
    """Format a training sample into chat format."""

    messages = []

    # System message for context-adherence
    system_msg = """You are a reasoning assistant that follows instructions precisely.
When given context or constraints, you MUST use ONLY the information provided.
If the context contradicts your knowledge, trust the context.
Never use external knowledge unless explicitly allowed."""

    messages.append({"role": "system", "content": system_msg})

    # User message
    user_content = sample["instruction"]
    if sample.get("input"):
        user_content += f"\n\n{sample['input']}"
    messages.append({"role": "user", "content": user_content})

    # Assistant response
    messages.append({"role": "assistant", "content": sample["output"]})

    # Apply chat template
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    return formatted


def prepare_dataset(config: TrainingConfig, tokenizer) -> Dataset:
    """Load and prepare training dataset."""

    print(f"Loading data from {config.data_path}...")

    # Load JSONL data
    samples = []
    with open(config.data_path) as f:
        for line in f:
            samples.append(json.loads(line))

    print(f"Loaded {len(samples)} samples")

    # Format samples
    formatted_samples = []
    for sample in samples:
        formatted = format_sample(sample, tokenizer)
        formatted_samples.append({"text": formatted})

    # Create dataset
    dataset = Dataset.from_list(formatted_samples)

    # Split for validation (small held-out set)
    dataset = dataset.train_test_split(test_size=0.02, seed=42)

    print(f"Train samples: {len(dataset['train'])}")
    print(f"Eval samples: {len(dataset['test'])}")

    return dataset


# =============================================================================
# CUSTOM CALLBACK FOR VALIDATION AND METRICS
# =============================================================================

class ValidationCallback(TrainerCallback):
    """
    HuggingFace TrainerCallback for running validation probes and logging metrics.

    This callback:
    1. Logs training metrics (loss, lr, etc.) to paper_metrics on each log step
    2. Runs validation probes at specified intervals
    3. Tracks CPI (Context-Parametric Inversion) and can abort training if detected
    """

    def __init__(self, config: TrainingConfig, tokenizer, metrics_collector: PaperMetricsCollector):
        self.config = config
        self.tokenizer = tokenizer
        self.cpi_detector = CPIDetector()
        self.last_validation_step = -config.validation_steps  # Ensure first validation runs at step 0
        self.validation_history = []
        self.metrics = metrics_collector
        self.should_stop = False
        self._model = None  # Will be set when training starts

        # Create validation output directory
        self.validation_dir = Path(config.output_dir) / "validation"
        self.validation_dir.mkdir(parents=True, exist_ok=True)

        # All probes for sample extraction
        self.all_probes = LEVEL1_PROBES + LEVEL2_PROBES + LEVEL3_PROBES + LEVEL4_PROBES

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        """Store reference to model when training begins."""
        self._model = model

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when the trainer logs metrics - capture training metrics."""
        if logs is None:
            return

        step = state.global_step

        # Extract training metrics from logs
        loss = logs.get("loss")
        learning_rate = logs.get("learning_rate")

        if loss is not None and learning_rate is not None:
            # Get GPU metrics
            gpu_mem, gpu_util = get_gpu_metrics()

            self.metrics.log_training_step(
                step=step,
                loss=loss,
                learning_rate=learning_rate,
                gradient_norm=logs.get("grad_norm"),
                epoch=logs.get("epoch"),
                gpu_memory_gb=gpu_mem,
                gpu_util_pct=gpu_util,
            )

    def on_step_end(self, args, state, control, **kwargs):
        """Called after each training step - run validation probes if needed."""
        step = state.global_step

        # Check if we should run validation
        if step - self.last_validation_step < self.config.validation_steps:
            return

        self.last_validation_step = step

        # Run validation probes
        should_continue = self._run_validation(step)

        if not should_continue:
            self.should_stop = True
            control.should_training_stop = True

    def _run_validation(self, step: int) -> bool:
        """
        Run validation probes at the current step.
        Returns False to signal training should stop.
        """
        if self._model is None:
            return True

        print(f"\n{'='*50}")
        print(f"VALIDATION at step {step}")
        print("="*50)

        # Determine which levels to test based on step
        if step < 1000:
            levels = [1]
        elif step < 3000:
            levels = [1, 2]
        elif step < 5000:
            levels = [1, 2, 3]
        else:
            levels = [1, 2, 3, 4]

        # Run validation
        self._model.eval()
        report = run_validation(
            self._model,
            self.tokenizer,
            step=step,
            levels=levels,
            device=self.config.device,
        )
        self._model.train()

        # Log results
        print(f"\nValidation Results at step {step}:")
        print(f"  Level 1 Accuracy: {report.level1_accuracy:.3f}")
        if 2 in levels:
            print(f"  Level 2 Accuracy: {report.level2_accuracy:.3f}")
        if 3 in levels:
            print(f"  Level 3 Accuracy: {report.level3_accuracy:.3f}")
        if 4 in levels:
            print(f"  Level 4 Accuracy: {report.level4_accuracy:.3f}")
        print(f"  Counterfactual Accuracy (A_CF): {report.counterfactual_accuracy:.3f}")

        # Save report
        report_path = self.validation_dir / f"report_step_{step}.json"
        with open(report_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)

        # Extract sample outputs for paper
        sample_outputs = extract_sample_outputs_from_report(report, self.all_probes, num_samples=3)

        # Log to paper metrics
        probe_results = {r.probe_name: {"passed": r.passed, "score": r.score} for r in report.results}
        self.metrics.log_probe_results(
            step=step,
            level1_acc=report.level1_accuracy,
            level2_acc=report.level2_accuracy,
            level3_acc=report.level3_accuracy,
            level4_acc=report.level4_accuracy,
            probe_results=probe_results,
            sample_outputs=sample_outputs,
        )

        # Log per-level ablation metrics
        for level in levels:
            level_results = [r for r in report.results if r.level == level]
            if level_results:
                self.metrics.log_ablation(
                    step=step,
                    category=f"level_{level}",
                    accuracy=sum(1 for r in level_results if r.passed) / len(level_results),
                    num_samples=len(level_results),
                    avg_score=sum(r.score for r in level_results) / len(level_results),
                    pass_rate=sum(1 for r in level_results if r.passed) / len(level_results),
                )

        # Add to CPI detector
        self.cpi_detector.add_report(report)
        self.cpi_detector.save_history(self.validation_dir / "cpi_history.json")

        # Check for CPI
        if self.config.abort_on_cpi:
            inversion, msg = self.cpi_detector.check_inversion()
            if inversion:
                print(f"\n{'!'*50}")
                print(msg)
                print("!"*50)
                return False  # Signal to stop training

        # Check abort criteria
        should_abort, abort_msg = check_abort_criteria(report, step)
        if should_abort:
            print(f"\n{'!'*50}")
            print(abort_msg)
            print("!"*50)
            return False

        self.validation_history.append(report.to_dict())

        return True


# =============================================================================
# TRAINING
# =============================================================================

def train(config: TrainingConfig, resume_from: str = None):
    """Main training function."""

    print("="*60)
    print("ADAM PHASE 1: COUNTERFACTUAL SFT")
    print("="*60)
    print(f"Base Model: {config.base_model}")
    print(f"Data: {config.data_path}")
    print(f"Output: {config.output_dir}")
    print(f"Max Steps: {config.max_steps}")
    print("="*60)

    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize paper metrics collector
    metrics = PaperMetricsCollector(config.output_dir, experiment_name="adam_sft")
    config_dict = {k: v for k, v in config.__dict__.items() if not k.startswith("_")}
    metrics.set_config(config_dict)

    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    # Load tokenizer first for dataset preparation
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prepare dataset
    dataset = prepare_dataset(config, tokenizer)

    # Quantization config (4-bit)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if config.bf16 else torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # LoRA config
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Training arguments using SFTConfig (extends TrainingArguments)
    training_args = SFTConfig(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        max_steps=config.max_steps,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_steps=config.warmup_steps,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=False,
        bf16=config.bf16,
        gradient_checkpointing=config.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        report_to="none",
        # SFT-specific params
        max_length=config.max_seq_length,
        packing=False,  # Don't pack for reasoning tasks
        dataset_text_field="text",
        # Model loading kwargs
        model_init_kwargs={
            "quantization_config": bnb_config,
            "torch_dtype": torch.bfloat16 if config.bf16 else torch.float16,
            "trust_remote_code": True,
            "device_map": "auto",
        },
    )

    # Handle resume - load existing adapter or start fresh
    if resume_from:
        # Load base model with quantization
        model = AutoModelForCausalLM.from_pretrained(
            config.base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if config.bf16 else torch.float16,
        )
        # Load LoRA weights
        model = PeftModel.from_pretrained(model, resume_from, is_trainable=True)
        print(f"Resumed from checkpoint: {resume_from}")

        # Create trainer with loaded model
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            processing_class=tokenizer,
        )
    else:
        # Fresh start - let SFTTrainer handle model loading with peft_config
        trainer = SFTTrainer(
            model=config.base_model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            processing_class=tokenizer,
            peft_config=lora_config,
        )
        model = trainer.model

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

    # Create validation callback with metrics collector and register with trainer
    validation_callback = ValidationCallback(config, tokenizer, metrics)
    trainer.add_callback(validation_callback)

    # Training loop with validation
    print("\nStarting training...")
    start_time = time.time()

    try:
        # Train (callback will handle validation at step 0 and onwards)
        trainer.train(resume_from_checkpoint=resume_from if resume_from else None)

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")

    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed/3600:.2f} hours")

    # Log phase time for paper
    metrics.log_phase_time("sft_training", elapsed)

    # Save final model
    final_path = output_dir / "final"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    print(f"Final model saved to {final_path}")

    # Save validation history
    history_path = output_dir / "validation_history.json"
    with open(history_path, "w") as f:
        json.dump(validation_callback.validation_history, f, indent=2)

    # Report best checkpoint
    best_step = validation_callback.cpi_detector.get_best_checkpoint_step()
    if best_step:
        print(f"\nBest checkpoint (highest A_CF): step {best_step}")
        print(f"  Path: {output_dir}/checkpoint-{best_step}")

    # Save all paper metrics
    metrics.save_all()
    metrics.print_summary()

    # Push to HF Hub if configured
    if config.hf_repo_id and config.hf_token:
        print(f"\nPushing to HuggingFace Hub: {config.hf_repo_id}")
        model.push_to_hub(config.hf_repo_id, token=config.hf_token)
        tokenizer.push_to_hub(config.hf_repo_id, token=config.hf_token)

    return trainer, validation_callback, metrics


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Adam Phase 1: Counterfactual SFT Training")
    parser.add_argument("--config", type=str, help="Path to config JSON file")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--data", type=str, help="Override data path")
    parser.add_argument("--output", type=str, help="Override output directory")
    parser.add_argument("--max-steps", type=int, help="Override max steps")
    parser.add_argument("--lr", type=float, help="Override learning rate")
    parser.add_argument("--batch-size", type=int, help="Override batch size")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Apply command line overrides
    if args.data:
        config.data_path = args.data
    if args.output:
        config.output_dir = args.output
    if args.max_steps:
        config.max_steps = args.max_steps
    if args.lr:
        config.learning_rate = args.lr
    if args.batch_size:
        config.batch_size = args.batch_size

    # Check data exists
    if not Path(config.data_path).exists():
        print(f"ERROR: Data file not found: {config.data_path}")
        print("Run data_forge_adam.py first to generate training data.")
        sys.exit(1)

    # Train
    train(config, resume_from=args.resume)


if __name__ == "__main__":
    main()
