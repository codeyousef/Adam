# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Adam is a language model trained via "Parametric Ignorance" — it learns reasoning structure, not world knowledge. The model relies entirely on provided context rather than memorized facts. Two training tracks exist:

- **Track A (PoC)**: 494M Qwen2-architecture trained from scratch on synthetic data. Scripts: `pretrain_adam_poc.py` → `sft_adam_poc.py` → `evaluate_poc.py`
- **Track B (Full)**: Qwen/Qwen2.5-Coder-3B-Instruct with QLoRA (4-bit NF4). Scripts: `train_adam_sft.py` → `train_adam_simpo.py` → `train_adam_daft.py`

## Commands

```bash
# Environment
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

# Generate SFT data (PoC path)
python data/data_forge_sft.py --output-dir hope/adam_training_data/sft

# Generate balanced SimPO preference data (full path)
python data/data_forge_adam_balanced.py

# PoC pretraining (resume auto-detects checkpoint)
bash run_pretrain.sh
# or directly:
PYTORCH_ALLOC_CONF=expandable_segments:True python pretrain_adam_poc.py \
    --data hope/adam_training_data/pretrain_corpus \
    --val-data hope/adam_training_data/pretrain_val.jsonl

# PoC SFT (joint mode trains all levels together)
python sft_adam_poc.py --pretrain-checkpoint adam_poc_checkpoints/checkpoint-244963 --joint

# Evaluate PoC checkpoint
python evaluate_poc.py --checkpoint adam_poc_sft_checkpoints/final --output results.json

# Validate a PEFT/QLoRA checkpoint
python validate.py --checkpoint hope/adam_sft_checkpoints/final

# Remote H200 bootstrap
./scripts/setup_fresh_h200.sh
```

## Architecture

### Validation Probes (`validation_probes.py`)

Central module imported by all training scripts. Defines L1-L4 probe suites with `run_validation(model, tokenizer, step, levels, device)`. Each probe has `expected_patterns` (regex) and `forbidden_patterns`. Scoring: `score = max(0, expected_score - forbidden_penalty * 0.5)`, pass requires `score >= 0.5 AND forbidden_violations == 0`. Includes `CPIDetector` for aborting training if context-parametric inversion is detected.

| Level | Task | PoC Target | Full Target |
|-------|------|------------|-------------|
| L1 | Knowledge Override (context beats pretrain) | 70% | 85% |
| L2 | Counterfactual Physics | 35% | 75% |
| L3 | Syllogistic Logic (PROVED/UNKNOWN) | 60% | 85% |
| L4 | Code Constraints | 75% | 90% |

### Data Generation (`data/`)

All training data uses nonsense entities (wampimuk, zorplax, etc.) to prevent real-world knowledge leakage. `data_forge_pretrain.py` generates 6B tokens of synthetic knowledge-sparse text. `data_forge_sft.py` generates curriculum-ordered SFT data for the PoC. `data_forge_adam_balanced.py` generates balanced data for the QLoRA path. Data format is plain text (NOT Qwen chat template) with `<|begin_of_thought|>/<|end_of_thought|>` for CoT sections.

### Training Pipeline

- **SFT**: Teaches reasoning format and structure across L1-L4
- **SimPO/CPO**: Preference alignment (v1-v4 all degraded; being replaced with GRPO + rule-based verifier)
- **DAFT**: Domain adversarial fine-tuning with gradient reversal for format invariance across 11 writing styles

### Checkpoint Layout

- `adam_poc_checkpoints/` — PoC pretrain (494M, from scratch)
- `adam_poc_sft_checkpoints/` — PoC SFT
- `hope/adam_sft_checkpoints/` — QLoRA SFT on Qwen2.5-Coder-3B
- `hope/adam_simpo_checkpoints/` — SimPO/CPO outputs
- `hope/adam_daft_checkpoints/` — DAFT outputs
- `archive/` — Previous attempts (Mamba, Jet)

All checkpoint dirs are gitignored. Data lives in `hope/adam_training_data/`.

## Conventions

- Config via `@dataclass` at top of each script, merged with CLI args
- All model loading uses `attn_implementation="sdpa"` and `torch_dtype=torch.bfloat16`
- Validation runs inline during training via `validation_probes.run_validation()`, not as a separate process
- All scripts run from project root; paths like `hope/adam_training_data/...` are relative to root
- Long training runs use tmux sessions. Remote instances use SSH key `~/.ssh/id_adam`
- `checkpoint-{step}/` directories with `final/` copy at end of training

## Critical Lessons (from debugging history)

- Custom data collators must be explicitly passed to HF Trainer via `data_collator=` — they are not auto-discovered
- "Loaded" in training logs does NOT mean "used" — always verify with actual batch processing
- Subsampling imbalanced data preserves bias — always regenerate from a balanced source
- Run validation probes DURING training (every N steps), not just at the end
- `r"valid"` as a forbidden pattern will match "validity" — use `r"\bvalid\b"` for word boundaries
- L1 training data OVERRIDE text must not contain literal forbidden values (e.g., "blue" when `r"blue"` is forbidden)
- Adding CoT format to one level can contaminate other levels — a 494M model bleeds formats across task types

## Hardware

- **RTX 4090** (24GB): batch_size=2, needs `PYTORCH_ALLOC_CONF=expandable_segments:True`
- **H100** (80GB): batch_size=8, ~7s/step for QLoRA 3B
- **H200** (141GB): batch_size=16, ~3s/step for QLoRA 3B
- Remote `/data/` persistent volumes survive spot instance crashes
