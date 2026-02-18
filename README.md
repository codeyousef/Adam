# 🐈 Adam: Context-First Reasoning Core

![Status](https://img.shields.io/badge/Status-Research-orange)
![Model](https://img.shields.io/badge/Model-Qwen2.5--Coder--3B-blue)
![Method](https://img.shields.io/badge/Training-Parametric_Ignorance-purple)
![License](https://img.shields.io/badge/License-MIT-yellow)

**Adam** is a 2.7B parameter language model trained to reason from context rather than memorize facts. Unlike traditional LLMs that compress the internet into their weights, Adam learns the *structure* of reasoning while remaining deliberately ignorant of content.

## Core Philosophy: Memory ≠ Intelligence

Adam is built on a single premise: **a reasoning engine should depend entirely on what you give it, not what it remembers.**

By training on entity-masked data where names, dates, and facts are systematically replaced with nonsense tokens, Adam learns:
- **Logical validity** (modus ponens works regardless of whether the premises are about cats or quarks)
- **Constraint satisfaction** (code works when it follows the rules, not when it uses popular libraries)
- **Counterfactual reasoning** (if the context says water boils at 50°C, compute with that)

What it doesn't learn: trivia, common knowledge, or the urge to hallucinate facts.

## Training Pipeline

Adam uses a three-stage curriculum:

### 1. SFT (Supervised Fine-Tuning)
- **Base model**: Qwen/Qwen2.5-Coder-3B-Instruct (QLoRA 4-bit NF4)
- **Data**: 48k examples across 4 reasoning levels (L1-L4)
- **Objective**: Learn the "grammar" of reasoning with masked entities

### 2. Preference Alignment (in progress)
- **Current**: SimPO/CPO — reference-free preference learning. Consistently degrades all tasks.
- **Next**: GRPO with rule-based verifier (L3 syllogisms and L4 constraints are fully decidable)

### 3. DAFT (Domain Adversarial Fine-Tuning)
- **Objective**: Format-invariant representations via gradient reversal
- **Data**: 11 writing domains (academic, casual, terse, verbose, etc.)
- **Target**: Model cannot distinguish whether a problem is written formally or casually

## Validation Hierarchy

Adam is evaluated on four levels of reasoning:

| Level | Task | Metric | Target |
|-------|------|--------|--------|
| **L1** | Knowledge Override | Context beats pretrain | ≥85% |
| **L2** | Counterfactual Physics | Numerical reasoning | ≥75% |
| **L3** | Syllogistic Logic | Valid/invalid/unknown | ≥85% |
| **L4** | Code Constraints | Follow arbitrary rules | ≥90% |

**Format Invariance Tests:**
- Cross-format consistency: 100% (same answer across 11 writing styles)
- Cue ablation: <5% degradation (removing explicit reasoning cues)
- Per-domain accuracy: worst ≥85% (no domain causes collapse)

## Key Research Findings

### The L3 Overcorrection Problem
During initial training, Adam learned to never say "UNKNOWN" for invalid syllogisms (10% accuracy). After preference optimization, it overcorrected to always saying "UNKNOWN" even for valid syllogisms (60% accuracy).

**Root cause**:
1. Data imbalance (3.3:1 invalid:valid ratio baked into generators)
2. Missing L3 replay buffer (catastrophic forgetting)
3. gamma_l3=1.0 over-penalizing short correct answers (PROVED/DISPROVED)

**Solution**:
1. Balanced data generation (4 valid + 4 invalid patterns = 1:1 ratio)
2. L3 replay buffer (15% of batches from SFT data)
3. gamma_l3=0.3 (stop penalizing short answers)

This demonstrates that reasoning models require careful curriculum design and continuous validation at each stage.

## Current Status

**SFT**: Complete — best checkpoint at `hope/adam_sft_checkpoints/final`

**SimPO v1–v4**: All runs degraded every task vs SFT baseline. SimPO is the wrong training method for this use case. Four consecutive runs confirmed: preference optimization with (chosen, rejected) pairs consistently causes catastrophic forgetting and fails to improve L3 syllogistic logic.

**Empirical results after SimPO v4:**

| Task | SFT | SimPO v4 | Target |
|------|-----|----------|--------|
| L1 Knowledge Override | 85% | 73% | ≥90% |
| L2 Counterfactual Physics | 45% | 48% | ≥75% |
| L3 Syllogistic Logic | ~60% | 8% | ≥90% |
| L4 Code Constraints | 90% | 73% | ≥95% |

**Next**: Replace SimPO with GRPO + rule-based verifier (L3/L4 are fully decidable; verified reward signal eliminates the preference data quality problem).

## Project Structure

```
.
├── train_adam_sft.py        # Phase 1: Supervised fine-tuning
├── train_adam_simpo.py      # Phase 2: Preference optimization (SimPO/CPO)
├── train_adam_daft.py       # Phase 3: Domain adversarial training
├── validation_probes.py     # L1–L4 reasoning probe suite
├── validate.py              # Standalone validation runner
├── paper_metrics.py         # Training metrics and logging
├── requirements.txt
│
├── data/
│   ├── data_forge_adam_balanced.py  # Training data generator (1:1 balanced)
│   ├── persona_augmentation.py      # 10-persona rephrasing via Ollama
│   ├── nli_verification.py          # Semantic equivalence checking
│   ├── fix_and_regenerate_data.py   # Fixes dict→string in preference data
│   ├── create_simpo_pairs.py        # Preference pair construction
│   ├── prepare_daft_data.py         # DAFT domain labeling
│   └── domain_labeler.py            # Writing style classifier
│
├── scripts/
│   ├── pipeline.sh          # Full training pipeline (SimPO → validate → DAFT)
│   ├── setup_fresh_h200.sh  # H200 instance bootstrap
│   ├── setup_h100.sh        # H100 instance bootstrap
│   ├── deploy_and_run_v4.sh # Deploy pipeline to remote instance
│   ├── start_training.sh    # Quick training launch
│   └── resume_from_crash.sh # Resume after spot instance preemption
│
└── docs_src/
    ├── RESEARCH_PROMPT_ADAM_V2.md   # Design brief for Adam v2 architecture
    ├── MODEL_CARD.md                # HuggingFace model card
    ├── L3_FIX_SUMMARY.md            # L3 failure analysis and lessons
    └── DEPLOYMENT_CHECKLIST.md      # Remote instance setup checklist
```

## Hardware Requirements

- **Training**: H200 (141GB) or H100 (80GB) for full pipeline
- **Inference**: RTX 4090 (24GB) sufficient with QLoRA
- **Local experiments**: Can run on consumer GPUs with reduced batch sizes

## Philosophy

Adam is not trying to replace GPT-4 or Claude. It's exploring a different design space:
- **No knowledge**: Adam knows nothing until you tell it
- **No chat**: Adam is a function, not a conversationalist
- **No web**: Adam operates on what you upload (PDFs, repos, contracts)

The goal is a reasoning core that never hallucinates because it has nothing to hallucinate about.

If you give Adam a physics textbook that says water boils at 50°C, Adam will compute steam tables at 50°C. It won't "correct" you with its pretrained knowledge of 100°C because it has no opinion on the matter.

## Citation

If you use Adam or the Parametric Ignorance methodology:

```bibtex
@software{adam2026,
  title={Adam: A Context-First Reasoning Core via Parametric Ignorance},
  author={Catbelly Studio},
  year={2026},
  url={https://github.com/catbelly-studio/adam}
}
```

## License

MIT License. Use it, break it, fix it, publish it.

---

*Developed with obsession by Catbelly Studio*
