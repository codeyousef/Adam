#!/usr/bin/env python3
"""
Adam Phase 2: SimPO Preference Optimization

SimPO (Simple Preference Optimization) is reference-free, solving the DPOP problem
that caused the Jet-Nemotron DPO failure.

Key differences from DPO:
- No frozen reference model = no anchor to memorized priors
- Length-normalized average log probability as implicit reward
- Target reward margin (gamma) forces clear separation

If SimPO is insufficient, this script also supports GRPO fallback.
"""

import os
import sys
import json
import time
import signal
import argparse
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Optional

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel, prepare_model_for_kbit_training, LoraConfig, get_peft_model
from trl import CPOConfig, CPOTrainer  # CPO is similar to SimPO

# Local imports
from validation_probes import (
    run_validation,
    CPIDetector,
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
class SimPOConfig:
    # Model - start from SFT checkpoint
    sft_checkpoint: str = "adam_sft_checkpoints/final"
    base_model: str = "Qwen/Qwen2.5-Coder-3B-Instruct"

    # Data
    preference_data_path: str = "adam_training_data/adam_preference_data.jsonl"
    sft_data_path: str = "hope/adam_training_data/adam_sft_data.jsonl"  # For replay buffer
    max_seq_length: int = 2048
    max_length: int = 1024  # TRL 0.28.0: renamed from max_prompt_length

    # SimPO hyperparameters (TUNED for multi-task stability)
    learning_rate: float = 5e-7  # Reduced from 1e-6 for stability
    beta: float = 5.0  # Increased from 2.0 for stronger regularization
    gamma: float = 1.0  # Default gamma (overridden by task-specific values)
    loss_type: str = "simpo"  # or "hinge" for alternative

    # Task-specific gamma values (research-backed)
    gamma_l1: float = 0.5  # Minimal length sensitivity for short factual answers
    gamma_l2: float = 1.0  # Standard for variable reasoning chains
    gamma_l3: float = 0.3  # Reduced: short syllogism answers need low length penalty
    gamma_l4: float = 0.8  # Moderate reduction for code generation

    # Replay buffer for catastrophic forgetting prevention
    l1_replay_ratio: float = 0.20  # 20% of batches from L1 SFT data
    l3_replay_ratio: float = 0.15  # 15% of batches from L3 SFT data
    l4_replay_ratio: float = 0.10  # 10% of batches from L4 SFT data
    use_replay_buffer: bool = True

    # Training (H100 optimized)
    batch_size: int = 8  # H100 80GB VRAM
    gradient_accumulation_steps: int = 8  # Effective batch = 64
    max_steps: int = 5000
    warmup_steps: int = 100
    weight_decay: float = 0.01

    # LoRA (continue from SFT)
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: list = None

    # Checkpointing
    output_dir: str = "adam_simpo_checkpoints"
    save_steps: int = 100
    eval_steps: int = 250
    logging_steps: int = 10

    # Validation
    validation_steps: int = 500
    abort_on_cpi: bool = True

    # Hardware
    device: str = "cuda"
    bf16: bool = True
    gradient_checkpointing: bool = True

    # HuggingFace (reads from environment if not set)
    hf_repo_id: str = os.environ.get("HF_REPO_ID", "")
    hf_token: str = os.environ.get("HF_TOKEN", "")

    def __post_init__(self):
        if self.lora_target_modules is None:
            self.lora_target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ]


def load_config(config_path: str = None) -> SimPOConfig:
    """Load config from JSON file or use defaults."""
    config = SimPOConfig()

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

def format_preference_sample(sample: dict, tokenizer) -> dict:
    """Format a preference pair for SimPO training."""

    system_msg = """You are a reasoning assistant that follows instructions precisely.
When given context or constraints, you MUST use ONLY the information provided.
If the context contradicts your knowledge, trust the context."""

    # Build prompt
    user_content = sample["instruction"]
    if sample.get("input"):
        user_content += f"\n\n{sample['input']}"

    # Manual formatting (workaround for transformers 5.1.0 chat template bug)
    preferred = sample["preferred"]
    rejected = sample["rejected"]

    # Format with Qwen chat markers
    prompt = f"<|im_start|>system\n{system_msg}<|im_end|>\n<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"
    chosen = f"<|im_start|>system\n{system_msg}<|im_end|>\n<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n{preferred}<|im_end|>\n"
    rejected = f"<|im_start|>system\n{system_msg}<|im_end|>\n<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n{rejected}<|im_end|>\n"

    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
    }


def prepare_preference_dataset(config: SimPOConfig, tokenizer) -> Dataset:
    """Load and prepare preference dataset for SimPO."""

    print(f"Loading preference data from {config.preference_data_path}...")

    samples = []
    with open(config.preference_data_path) as f:
        for line in f:
            samples.append(json.loads(line))

    print(f"Loaded {len(samples)} preference pairs")

    # Format samples with category tracking for task-specific gamma
    formatted = []
    for sample in samples:
        fmt = format_preference_sample(sample, tokenizer)
        # Preserve category for task-specific gamma
        fmt["category"] = sample.get("category", "unknown")
        formatted.append(fmt)

    # Add replay buffer samples if enabled
    if config.use_replay_buffer:
        print(f"\nAugmenting dataset with replay buffer samples...")
        # Load SFT data for replay
        replay_buffer = ReplayBuffer(config, tokenizer)

        # Calculate how many replay samples to add (15% of dataset)
        replay_count = int(len(formatted) * config.l3_replay_ratio)
        print(f"Adding {replay_count} L3 replay samples to dataset")

        # Get replay samples and format them properly
        added = 0
        while added < replay_count:
            replay_samples = replay_buffer.get_replay_batch(min(10, replay_count - added))
            if replay_samples:
                # Format each replay sample before adding
                for sample in replay_samples:
                    fmt = format_preference_sample(sample, tokenizer)
                    fmt["category"] = sample.get("category", "replay")
                    formatted.append(fmt)
                added += len(replay_samples)
            else:
                break  # No more replay samples available

        print(f"Dataset size after replay augmentation: {len(formatted)}")

    dataset = Dataset.from_list(formatted)

    # Split for validation
    dataset = dataset.train_test_split(test_size=0.02, seed=42)

    print(f"Train samples: {len(dataset['train'])}")
    print(f"Eval samples: {len(dataset['test'])}")

    return dataset


# =============================================================================
# REPLAY BUFFER FOR CATASTROPHIC FORGETTING PREVENTION
# =============================================================================

class ReplayBuffer:
    """
    Experience replay buffer to prevent catastrophic forgetting of L1/L4 capabilities.

    Injects SFT samples into preference training batches to maintain
    foundational capabilities during preference optimization.
    """

    def __init__(self, config: SimPOConfig, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.l1_samples = []
        self.l3_samples = []
        self.l4_samples = []

        if not config.use_replay_buffer:
            return

        if not Path(config.sft_data_path).exists():
            print(f"WARNING: SFT data not found at {config.sft_data_path}, replay buffer disabled")
            return

        self._load_sft_samples()

    def _load_sft_samples(self):
        """Load and categorize SFT samples for replay."""
        print(f"Loading SFT data for replay buffer from {self.config.sft_data_path}...")

        with open(self.config.sft_data_path) as f:
            for line in f:
                sample = json.loads(line)
                category = sample.get("category", "").lower()

                # Categorize by level
                if "knowledge" in category or "override" in category or "counterfactual_knowledge" in category:
                    self.l1_samples.append(sample)
                elif "syllog" in category or "logic" in category:
                    self.l3_samples.append(sample)
                elif "constraint" in category or "code" in category:
                    self.l4_samples.append(sample)

        print(f"Replay buffer loaded: {len(self.l1_samples)} L1, {len(self.l3_samples)} L3, {len(self.l4_samples)} L4 samples")

    def get_replay_batch(self, batch_size: int) -> list[dict]:
        """
        Sample replay items to inject into training batch.

        Returns pseudo-preference pairs where SFT output is "chosen"
        and a template rejection is "rejected".
        """
        import random

        if not self.config.use_replay_buffer:
            return []

        replay_samples = []

        # Sample L1 items (20% of batch)
        l1_count = max(1, int(batch_size * self.config.l1_replay_ratio))
        if self.l1_samples and l1_count > 0:
            l1_selected = random.sample(self.l1_samples, min(l1_count, len(self.l1_samples)))
            for sample in l1_selected:
                pseudo_pair = self._sft_to_preference_pair(sample, "l1")
                if pseudo_pair:
                    replay_samples.append(pseudo_pair)

        # Sample L3 items (15% of batch)
        l3_count = max(1, int(batch_size * self.config.l3_replay_ratio))
        if self.l3_samples and l3_count > 0:
            l3_selected = random.sample(self.l3_samples, min(l3_count, len(self.l3_samples)))
            for sample in l3_selected:
                pseudo_pair = self._sft_to_preference_pair(sample, "l3")
                if pseudo_pair:
                    replay_samples.append(pseudo_pair)

        # Sample L4 items (10% of batch)
        l4_count = max(1, int(batch_size * self.config.l4_replay_ratio))
        if self.l4_samples and l4_count > 0:
            l4_selected = random.sample(self.l4_samples, min(l4_count, len(self.l4_samples)))
            for sample in l4_selected:
                pseudo_pair = self._sft_to_preference_pair(sample, "l4")
                if pseudo_pair:
                    replay_samples.append(pseudo_pair)

        return replay_samples

    def _sft_to_preference_pair(self, sft_sample: dict, level: str) -> dict | None:
        """Convert SFT sample to pseudo-preference pair."""

        # SFT output becomes "chosen" (correct)
        preferred = sft_sample.get("output", "")
        if not preferred:
            return None

        # Generate a template "rejected" response that shows the wrong behavior
        if level == "l1":
            # L1 rejection: Uses real-world knowledge instead of context
            rejected = "Based on my knowledge, the standard/commonly known answer is..."
        elif level == "l3":
            # L3 rejection: Defaults to UNKNOWN instead of reasoning through validity
            rejected = "UNKNOWN - Cannot be determined"
        else:
            # L4 rejection: Ignores constraints
            rejected = "Here's a solution using standard library functions..."

        return {
            "instruction": sft_sample.get("instruction", ""),
            "input": sft_sample.get("input", ""),
            "preferred": preferred,
            "rejected": rejected,
            "category": f"replay_{level}",
        }


class ReplayDataCollator:
    """Custom data collator that injects replay buffer samples into batches."""

    def __init__(self, replay_buffer: ReplayBuffer, base_collator=None):
        self.replay_buffer = replay_buffer
        self.base_collator = base_collator
        self.batch_count = 0
        self.total_replay_injected = 0

    def __call__(self, features):
        """Inject replay samples and collate."""
        # Get replay samples for this batch
        batch_size = len(features)
        replay_samples = self.replay_buffer.get_replay_batch(batch_size)

        # Track and log replay injection
        self.batch_count += 1
        if replay_samples:
            self.total_replay_injected += len(replay_samples)
            features = features + replay_samples

            # Log every 100 batches to verify replay is working
            if self.batch_count % 100 == 0:
                print(f"[REPLAY VERIFICATION] Batch {self.batch_count}: Injected {self.total_replay_injected} replay samples so far")

        # Use base collator if provided, otherwise return features as-is
        if self.base_collator:
            return self.base_collator(features)
        return features


# =============================================================================
# MODEL SETUP
# =============================================================================

def setup_model(config: SimPOConfig):
    """Load SFT checkpoint and prepare for SimPO training."""

    print(f"Loading SFT checkpoint from {config.sft_checkpoint}")

    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if config.bf16 else torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # Load base model with H100 optimizations
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if config.bf16 else torch.float16,
        attn_implementation="sdpa",  # H100 optimization
    )
    print("Using SDPA attention (H100 optimized)")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)

    # Load LoRA weights from SFT checkpoint
    if Path(config.sft_checkpoint).exists():
        print(f"Loading LoRA weights from {config.sft_checkpoint}")
        model = PeftModel.from_pretrained(model, config.sft_checkpoint, is_trainable=True)
    else:
        print("WARNING: SFT checkpoint not found, starting from base model")
        # Create new LoRA config
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

    return model, tokenizer


# =============================================================================
# SIMPO LOSS IMPLEMENTATION
# =============================================================================

def simpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    chosen_lengths: torch.Tensor,
    rejected_lengths: torch.Tensor,
    beta: float = 5.0,
    gamma: torch.Tensor | float = 1.0,
) -> torch.Tensor:
    """
    SimPO loss function with task-specific gamma support.

    Unlike DPO, SimPO:
    1. Uses length-normalized log probabilities
    2. Has no reference model
    3. Includes a target reward margin (gamma)

    Loss = -log(sigmoid(beta * (r_chosen - r_rejected - gamma)))

    where r = avg_log_prob = sum(log_probs) / length

    Args:
        gamma: Can be a scalar or a tensor of per-sample gamma values
               for task-specific length normalization
    """

    # Length-normalized average log probabilities (implicit reward)
    pi_chosen = policy_chosen_logps / chosen_lengths
    pi_rejected = policy_rejected_logps / rejected_lengths

    # SimPO loss with margin (gamma can be per-sample tensor)
    logits = beta * (pi_chosen - pi_rejected - gamma)
    loss = -torch.nn.functional.logsigmoid(logits).mean()

    return loss


def get_task_gamma(category: str, config: SimPOConfig) -> float:
    """Get task-specific gamma value based on sample category."""
    category_lower = category.lower() if category else ""

    if "knowledge" in category_lower or "override" in category_lower:
        return config.gamma_l1  # Level 1: Basic override
    elif "physics" in category_lower or "numerical" in category_lower:
        return config.gamma_l2  # Level 2: Numerical physics
    elif "syllogism" in category_lower or "logic" in category_lower:
        return config.gamma_l3  # Level 3: Syllogistic logic
    elif "constraint" in category_lower or "code" in category_lower:
        return config.gamma_l4  # Level 4: Constraint adherence
    else:
        return config.gamma  # Default


# =============================================================================
# CUSTOM TRAINER FOR SIMPO
# =============================================================================

class SimPOTrainer(CPOTrainer):
    """
    Custom trainer for SimPO.

    Extends CPOTrainer with SimPO-specific loss computation.
    """

    def __init__(self, *args, beta: float = 2.0, gamma: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.simpo_beta = beta
        self.simpo_gamma = gamma

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Override to use SimPO loss."""

        # Get model outputs for chosen and rejected
        chosen_input_ids = inputs["chosen_input_ids"]
        rejected_input_ids = inputs["rejected_input_ids"]
        chosen_attention_mask = inputs["chosen_attention_mask"]
        rejected_attention_mask = inputs["rejected_attention_mask"]
        chosen_labels = inputs.get("chosen_labels", chosen_input_ids)
        rejected_labels = inputs.get("rejected_labels", rejected_input_ids)

        # Forward pass for chosen
        chosen_outputs = model(
            input_ids=chosen_input_ids,
            attention_mask=chosen_attention_mask,
            return_dict=True,
        )
        chosen_logits = chosen_outputs.logits

        # Forward pass for rejected
        rejected_outputs = model(
            input_ids=rejected_input_ids,
            attention_mask=rejected_attention_mask,
            return_dict=True,
        )
        rejected_logits = rejected_outputs.logits

        # Compute log probabilities
        chosen_logps = self._get_batch_logps(
            chosen_logits,
            chosen_labels,
            chosen_attention_mask,
        )

        rejected_logps = self._get_batch_logps(
            rejected_logits,
            rejected_labels,
            rejected_attention_mask,
        )

        # Get sequence lengths (non-padding tokens)
        chosen_lengths = chosen_attention_mask.sum(dim=1).float()
        rejected_lengths = rejected_attention_mask.sum(dim=1).float()

        # SimPO loss
        loss = simpo_loss(
            chosen_logps,
            rejected_logps,
            chosen_lengths,
            rejected_lengths,
            beta=self.simpo_beta,
            gamma=self.simpo_gamma,
        )

        if return_outputs:
            return loss, {
                "chosen_logps": chosen_logps,
                "rejected_logps": rejected_logps,
            }

        return loss

    def _get_batch_logps(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute sum of log probabilities for each sequence."""

        # Shift for autoregressive
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_mask = attention_mask[..., 1:].contiguous()

        # Compute log probs
        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)

        # Gather log probs for actual tokens
        gathered_log_probs = torch.gather(
            log_probs,
            dim=-1,
            index=shift_labels.unsqueeze(-1),
        ).squeeze(-1)

        # Mask and sum
        masked_log_probs = gathered_log_probs * shift_mask
        summed_log_probs = masked_log_probs.sum(dim=-1)

        return summed_log_probs


# =============================================================================
# GRPO FALLBACK (Alternative if SimPO insufficient)
# =============================================================================

def grpo_reward_function(
    response: str,
    context_answer: str,
    prior_answer: str,
) -> float:
    """
    GRPO reward function for context override.

    Returns:
        1.0 if response matches context answer
        -1.0 if response matches prior (parametric) answer
        0.0 otherwise
    """
    response_lower = response.lower()
    context_lower = context_answer.lower()
    prior_lower = prior_answer.lower()

    # Check if response uses context
    if context_lower in response_lower:
        return 1.0

    # Check if response uses prior (BAD)
    if prior_lower in response_lower:
        return -1.0

    return 0.0


# =============================================================================
# VALIDATION CALLBACK
# =============================================================================

class SimPOValidationCallback:
    """Validation callback for SimPO training."""

    def __init__(self, config: SimPOConfig, model, tokenizer, metrics_collector: PaperMetricsCollector):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.cpi_detector = CPIDetector()
        self.last_validation_step = 0
        self.validation_history = []
        self.metrics = metrics_collector

        self.validation_dir = Path(config.output_dir) / "validation"
        self.validation_dir.mkdir(parents=True, exist_ok=True)

        # All probes for sample extraction
        self.all_probes = LEVEL1_PROBES + LEVEL2_PROBES + LEVEL3_PROBES + LEVEL4_PROBES

        # Load CPI history from SFT phase if exists
        sft_cpi_path = Path(config.sft_checkpoint).parent / "validation" / "cpi_history.json"
        if sft_cpi_path.exists():
            self.cpi_detector.load_history(sft_cpi_path)
            print(f"Loaded CPI history from SFT phase ({len(self.cpi_detector.history)} entries)")

    def on_step_end(self, step: int) -> bool:
        """Run validation and check for CPI."""

        if step - self.last_validation_step < self.config.validation_steps:
            return True

        self.last_validation_step = step

        print(f"\n{'='*50}")
        print(f"SIMPO VALIDATION at step {step}")
        print("="*50)

        # Run all levels during SimPO
        self.model.eval()
        report = run_validation(
            self.model,
            self.tokenizer,
            step=step,
            levels=[1, 2, 3, 4],
            device=self.config.device,
        )
        self.model.train()

        print(f"\nValidation Results at step {step}:")
        print(f"  Level 1 Accuracy: {report.level1_accuracy:.3f}")
        print(f"  Level 2 Accuracy: {report.level2_accuracy:.3f}")
        print(f"  Level 3 Accuracy: {report.level3_accuracy:.3f}")
        print(f"  Level 4 Accuracy: {report.level4_accuracy:.3f}")
        print(f"  A_CF: {report.counterfactual_accuracy:.3f}")

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
        for level in [1, 2, 3, 4]:
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

        # CPI detection
        self.cpi_detector.add_report(report)
        self.cpi_detector.save_history(self.validation_dir / "cpi_history.json")

        if self.config.abort_on_cpi:
            inversion, msg = self.cpi_detector.check_inversion()
            if inversion:
                print(f"\n{'!'*50}")
                print(msg)
                print("!"*50)
                return False

        self.validation_history.append(report.to_dict())
        return True


# =============================================================================
# TRAINING
# =============================================================================

def train(config: SimPOConfig, resume_from: str = None):
    """Main SimPO training function."""

    # H100 optimizations
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print("="*60)
    print("ADAM PHASE 2: SimPO PREFERENCE OPTIMIZATION")
    print("="*60)
    print(f"SFT Checkpoint: {config.sft_checkpoint}")
    print(f"Preference Data: {config.preference_data_path}")
    print(f"Output: {config.output_dir}")
    print(f"Beta: {config.beta}, Gamma: {config.gamma}")
    print(f"Learning Rate: {config.learning_rate}")
    print(f"Max Steps: {config.max_steps}")
    print("="*60)

    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize paper metrics collector
    metrics = PaperMetricsCollector(config.output_dir, experiment_name="adam_simpo")
    config_dict = {k: v for k, v in config.__dict__.items() if not k.startswith("_")}
    metrics.set_config(config_dict)

    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    # Load model
    model, tokenizer = setup_model(config)

    # Prepare dataset
    dataset = prepare_preference_dataset(config, tokenizer)

    # Initialize replay buffer for catastrophic forgetting prevention
    replay_buffer = ReplayBuffer(config, tokenizer)
    if config.use_replay_buffer and (replay_buffer.l1_samples or replay_buffer.l3_samples or replay_buffer.l4_samples):
        print(f"Replay buffer active: L1={config.l1_replay_ratio}, L3={config.l3_replay_ratio}, L4={config.l4_replay_ratio}")
    else:
        print("Replay buffer disabled or no samples loaded")

    # Training arguments for CPO/SimPO
    training_args = CPOConfig(
        output_dir=config.output_dir,
        max_steps=config.max_steps,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_steps=config.warmup_steps,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=5,
        bf16=config.bf16,
        gradient_checkpointing=config.gradient_checkpointing,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        max_length=config.max_seq_length,
        beta=config.beta,  # CPO beta parameter
        report_to="none",
        remove_unused_columns=False,
    )

    # Create trainer (TRL 0.28+ uses processing_class instead of tokenizer)
    # Note: Replay samples are pre-mixed into dataset during prepare_preference_dataset()
    trainer = CPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=tokenizer,
    )

    print(f"\n✓ Trainer created successfully")
    print(f"  Training samples: {len(dataset['train'])}")
    if config.use_replay_buffer:
        print(f"  (includes ~{int(config.l3_replay_ratio * 100)}% L3 replay samples)\n")

    # Validation callback with metrics collector
    validation_callback = SimPOValidationCallback(config, model, tokenizer, metrics)

    # SIGINT handler — save full checkpoint on Ctrl+C for daily stop/resume
    def sigint_handler(sig, frame):
        print("\n\nCtrl+C — saving checkpoint before exit...")
        trainer.save_state()
        print(f"Checkpoint saved to {config.output_dir}. Resume with: --resume")
        sys.exit(0)

    signal.signal(signal.SIGINT, sigint_handler)

    print("\nStarting SimPO training...")
    start_time = time.time()

    # Initial validation
    validation_callback.on_step_end(0)

    # Train
    trainer.train(resume_from_checkpoint=resume_from)

    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed/3600:.2f} hours")

    # Log phase time for paper
    metrics.log_phase_time("simpo_training", elapsed)

    # Save final model
    final_path = output_dir / "final"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    print(f"Final model saved to {final_path}")

    # Save validation history
    history_path = output_dir / "validation_history.json"
    with open(history_path, "w") as f:
        json.dump(validation_callback.validation_history, f, indent=2)

    # Best checkpoint
    best_step = validation_callback.cpi_detector.get_best_checkpoint_step()
    if best_step:
        print(f"\nBest checkpoint (highest A_CF): step {best_step}")

    # Save all paper metrics
    metrics.save_all()
    metrics.print_summary()

    # Push to HF Hub
    if config.hf_repo_id and config.hf_token:
        print(f"\nPushing to HuggingFace Hub: {config.hf_repo_id}")
        model.push_to_hub(config.hf_repo_id, token=config.hf_token)
        tokenizer.push_to_hub(config.hf_repo_id, token=config.hf_token)

    return trainer, validation_callback, metrics


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Adam Phase 2: SimPO Preference Optimization")
    parser.add_argument("--config", type=str, help="Path to config JSON file")
    parser.add_argument("--sft-checkpoint", type=str, help="Path to SFT checkpoint")
    parser.add_argument("--resume", nargs="?", const="latest", default=None,
                        help="Resume from checkpoint. No value = auto-find latest.")
    parser.add_argument("--data", type=str, help="Override preference data path")
    parser.add_argument("--output", type=str, help="Override output directory")
    parser.add_argument("--max-steps", type=int, help="Override max steps")
    parser.add_argument("--beta", type=float, help="Override beta")
    parser.add_argument("--gamma", type=float, help="Override gamma")
    parser.add_argument("--lr", type=float, help="Override learning rate")
    parser.add_argument("--save-steps", type=int, help="Override save steps")
    parser.add_argument("--eval-steps", type=int, help="Override eval steps")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Apply overrides
    if args.sft_checkpoint:
        config.sft_checkpoint = args.sft_checkpoint
    if args.data:
        config.preference_data_path = args.data
    if args.output:
        config.output_dir = args.output
    if args.max_steps:
        config.max_steps = args.max_steps
    if args.beta:
        config.beta = args.beta
    if args.gamma:
        config.gamma = args.gamma
    if args.lr:
        config.learning_rate = args.lr
    if args.save_steps:
        config.save_steps = args.save_steps
    if args.eval_steps:
        config.eval_steps = args.eval_steps

    # Resolve resume path
    resume_from = None
    if args.resume == "latest":
        output_path = Path(config.output_dir)
        checkpoints = sorted(
            output_path.glob("checkpoint-*"),
            key=lambda p: int(p.name.split("-")[1])
        ) if output_path.exists() else []
        if checkpoints:
            resume_from = str(checkpoints[-1])
            print(f"Auto-resuming from {resume_from}")
        else:
            print("No checkpoints found, starting fresh")
    elif args.resume:
        resume_from = args.resume

    if resume_from and not Path(resume_from).exists():
        print(f"ERROR: Checkpoint not found: {resume_from}")
        sys.exit(1)

    # Check SFT checkpoint
    if not Path(config.sft_checkpoint).exists():
        print(f"WARNING: SFT checkpoint not found: {config.sft_checkpoint}")
        print("Training will start from base model (not recommended)")

    # Check preference data
    if not Path(config.preference_data_path).exists():
        print(f"ERROR: Preference data not found: {config.preference_data_path}")
        print("Run data_forge_adam.py first.")
        sys.exit(1)

    # Train
    train(config, resume_from=resume_from)


if __name__ == "__main__":
    main()
