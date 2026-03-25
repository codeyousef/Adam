# Copilot Instructions — Adam (Parametric Ignorance)

> Adam is a language model trained to reason from provided context, never memorized facts.
> Two training tracks: **PoC** (494M from-scratch) and **Full** (Qwen2.5-Coder-3B QLoRA).

## Quick Reference

| Action | Command |
|--------|---------|
| Generate SFT data | `python data/data_forge_sft.py --output-dir hope/adam_training_data/sft` |
| Generate preference data | `python data/data_forge_adam_balanced.py` |
| PoC pretrain (resume-aware) | `bash run_pretrain.sh` |
| PoC SFT (joint) | `python sft_adam_poc.py --pretrain-checkpoint adam_poc_checkpoints/checkpoint-244963 --joint` |
| Evaluate PoC | `python evaluate_poc.py --checkpoint adam_poc_sft_checkpoints/final --output results.json` |
| Validate QLoRA checkpoint | `python validate.py --checkpoint hope/adam_sft_checkpoints/final` |
| Autoresearch experiment | Edit `autoresearch/train.py`, commit, run, log to `autoresearch/results.tsv` |

## Architecture at a Glance

```
data/               → Data generators (nonsense entities only, no real-world facts)
validation_probes.py → L1-L4 probe suites — NEVER modify probes, only generators/trainers
pretrain_adam_poc.py → PoC pretraining (494M Qwen2-arch, from scratch)
sft_adam_poc.py     → PoC SFT (curriculum or joint mode)
train_adam_sft.py   → QLoRA SFT on Qwen2.5-Coder-3B-Instruct
train_adam_simpo.py → SimPO/CPO preference alignment
train_adam_daft.py  → Domain adversarial fine-tuning (11 writing styles)
evaluate_poc.py     → Evaluate PoC checkpoints (direct model load)
validate.py         → Evaluate QLoRA checkpoints (base + merge adapter)
autoresearch/       → Automated experiment loop for Track B
hope/               → Cloud training scripts and data (H100/H200)
```

## Mandatory Conventions

### Model Loading

Every script **must** use these two kwargs — no exceptions:

```python
attn_implementation="sdpa"
torch_dtype=torch.bfloat16
```

QLoRA always uses 4-bit NF4:
```python
BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                   bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
```

### Config Pattern

All scripts define a `@dataclass` at top of file, merged with CLI args. Add new hyperparameters here, not as loose globals.

### Validation Probes

- Import via `from validation_probes import run_validation`
- Call signature: `run_validation(model, tokenizer, step, levels=[1,2,3,4], device="cuda")`
- Returns `ValidationReport` with `.level1_accuracy` through `.level4_accuracy`
- **Must run DURING training** (every N steps via callback), not only at end
- Score formula: `max(0, expected_score - forbidden_penalty * 0.5)`, pass requires `≥0.5 AND 0 forbidden violations`

### Data Generation

- All training data uses **nonsense entities** (wampimuk, zorplax, blurgle, etc.) — never real-world facts
- Format is **plain text** with `CONTEXT_SAYS:` / `OVERRIDE:` / `ANSWER:` markers — NOT Qwen chat template
- CoT uses `<|begin_of_thought|>` / `<|end_of_thought|>` delimiters
- L3 data must be **balanced** between valid/invalid/unknown — imbalance causes overcorrection

### HuggingFace Trainer

- Custom `data_collator=` must be **explicitly passed** to Trainer — it is never auto-discovered
- "Loaded" in training logs does NOT mean "used" — verify with actual batch output

## Pitfalls to Avoid

| Pitfall | Why | Fix |
|---------|-----|-----|
| `r"valid"` as forbidden pattern | Matches "validity", "validation" | Use `r"\bvalid\b"` |
| L1 override text contains forbidden value | e.g. "blue" in override when `r"blue"` is forbidden | Audit data generators |
| Subsampling imbalanced data | Preserves the bias | Regenerate from balanced source |
| Adding CoT to one level only | 494M model bleeds format across task types | Test all levels after any format change |
| lr=5e-4 on QLoRA pretrained model | Catastrophic forgetting | Use lr=2e-5 for QLoRA |
| lr=2e-5 from scratch | Too slow, won't converge | Use lr=3e-4 to 5e-4 for PoC |
| Missing `PYTORCH_ALLOC_CONF=expandable_segments:True` | OOM on RTX 4090 | Set in env or shell script |

## Checkpoint Layout

| Directory | Contents |
|-----------|----------|
| `adam_poc_checkpoints/` | PoC pretrain (494M) — gitignored |
| `adam_poc_sft_checkpoints/` | PoC SFT — gitignored |
| `hope/adam_sft_checkpoints/` | QLoRA SFT (3B) — gitignored |
| `hope/adam_simpo_checkpoints/` | SimPO/CPO outputs — gitignored |
| `hope/adam_daft_checkpoints/` | DAFT outputs — gitignored |
| `archive/` | Previous failed attempts (Mamba, Jet) |

Naming: `checkpoint-{step}/` directories, with `final/` copy at end of training.

## Hardware Profiles

- **RTX 4090** (24 GB): batch_size ≤ 2, needs `expandable_segments`
- **H100** (80 GB): batch_size=8, ~7 s/step QLoRA
- **H200** (141 GB): batch_size=16, ~3 s/step QLoRA

## Target Metrics

| Level | Task | PoC Target | Full Target |
|-------|------|-----------|-------------|
| L1 | Knowledge Override | 70% | 85% |
| L2 | Counterfactual Physics | 35% | 75% |
| L3 | Syllogistic Logic | 60% | 85% |
| L4 | Code Constraints | 75% | 90% |

`pi_score = mean(L1, L2, L3, L4)` — Full target: 83.75%
