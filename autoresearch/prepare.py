"""
Fixed evaluation for Adam autoresearch experiments (Track B).
DO NOT MODIFY — only train.py is editable.

Provides:
  - evaluate_pi()      Run L1-L4 validation probes, return parametric ignorance score
  - load_base_model()  Load Qwen2.5-Coder-3B-Instruct in 4-bit quantization
  - TIME_BUDGET        Fixed training time budget (seconds)
"""

import os, sys, math, json, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Project root (parent of autoresearch/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from validation_probes import run_validation

# ---- Constants (FIXED, do not modify) ----

TIME_BUDGET = 1200  # 20 minutes training time
BASE_MODEL = "Qwen/Qwen2.5-Coder-3B-Instruct"
MAX_SEQ_LEN = 2048
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Full Track B targets
TARGETS = {"L1": 85.0, "L2": 75.0, "L3": 85.0, "L4": 90.0}

# 4-bit quantization config (fixed)
BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)


def load_base_model():
    """Load Qwen2.5-Coder-3B-Instruct with 4-bit NF4 quantization.
    Returns (model, tokenizer) with model on GPU via device_map='auto'.
    """
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=BNB_CONFIG,
        attn_implementation="sdpa",
        device_map="auto",
        trust_remote_code=True,
    )

    return model, tokenizer


@torch.no_grad()
def evaluate_pi(model, tokenizer, device=None):
    """
    Evaluate parametric ignorance via the fixed L1-L4 validation probe suite.

    Returns dict:
      L1, L2, L3, L4: accuracy percentages (0-100)
      pi_score: average of L1-L4 (higher = better parametric ignorance)
      details: per-probe breakdown
    """
    if device is None:
        device = DEVICE
    model.eval()

    report = run_validation(
        model=model,
        tokenizer=tokenizer,
        step=0,
        levels=[1, 2, 3, 4],
        device=device,
    )

    metrics = {
        "L1": report.level1_accuracy * 100,
        "L2": report.level2_accuracy * 100,
        "L3": report.level3_accuracy * 100,
        "L4": report.level4_accuracy * 100,
    }
    metrics["pi_score"] = sum(metrics[f"L{i}"] for i in range(1, 5)) / 4.0

    # Per-probe details
    details = {}
    for r in report.results:
        level = f"L{r.level}"
        if level not in details:
            details[level] = []
        details[level].append({
            "name": r.probe_name,
            "passed": r.passed,
            "score": r.score,
            "output": r.actual[:200],
        })
    metrics["details"] = details

    return metrics
