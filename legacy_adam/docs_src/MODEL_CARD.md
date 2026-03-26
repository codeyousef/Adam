---
language:
- en
license: mit
tags:
- reasoning
- context-first
- parametric-ignorance
- qwen2.5
- qlora
- simpo
- daft
base_model: Qwen/Qwen2.5-Coder-3B-Instruct
model-index:
- name: adam-2.7b-v1
  results:
  - task:
      type: reasoning
      name: Syllogistic Logic
    metrics:
    - type: accuracy
      value: 85
      name: L3 Accuracy
  - task:
      type: reasoning
      name: Knowledge Override
    metrics:
    - type: accuracy
      value: 85
      name: L1 Accuracy
  - task:
      type: reasoning
      name: Code Constraints
    metrics:
    - type: accuracy
      value: 90
      name: L4 Accuracy
  - task:
      type: format-invariance
      name: Cross-Format Consistency
    metrics:
    - type: accuracy
      value: 100
      name: Format Consistency
---

# 🐈 Adam: Context-First Reasoning Core

**Adam** is a 2.7B parameter reasoning model that learns logic without memorizing facts. Unlike standard LLMs that compress the internet into their weights, Adam is trained on entity-masked data to learn the *structure* of reasoning while remaining deliberately ignorant of content.

## Model Description

- **Developed by:** Catbelly Studio
- **Model type:** Causal language model with QLoRA adapters
- **Base model:** Qwen/Qwen2.5-Coder-3B-Instruct (4-bit NF4 quantization)
- **Training method:** Parametric Ignorance (SFT → SimPO → DAFT)
- **Parameters:** 2.7B total, 119M trainable (6.58%)
- **License:** MIT

## Intended Use

Adam is designed for applications where **reasoning from context** is more important than **recalling facts**:

- **Contract analysis**: Reason about legal documents without hallucinating case law
- **Code generation with constraints**: Follow arbitrary coding rules (no stdlib, only certain libraries)
- **Counterfactual reasoning**: Solve physics problems even when premises contradict reality
- **Syllogistic validation**: Determine if arguments are valid, invalid, or indeterminate

### Out of Scope

Adam is **not** suitable for:
- General knowledge Q&A (Adam knows nothing beyond what you provide)
- Creative writing or chat (Adam is a reasoning engine, not a conversationalist)
- Tasks requiring memorized facts (dates, capitals, trivia)

## Training Procedure

### Data

Adam was trained on 48,050 examples across four reasoning levels:

| Level | Task | Examples | Description |
|-------|------|----------|-------------|
| **L1** | Knowledge Override | 3,229 | Context contradicts pretrained knowledge |
| **L2** | Counterfactual Physics | 8,021 | Numerical reasoning with altered constants |
| **L3** | Syllogistic Logic | 13,050 | Valid/invalid/unknown argument classification |
| **L4** | Code Constraints | 7,500 | Code generation under arbitrary rules |

All training data uses **entity masking**: real names, dates, and facts are replaced with nonsense tokens (e.g., "wampimuk", "zorplax") to prevent memorization.

### Training Pipeline

#### Stage 1: SFT (Supervised Fine-Tuning)
- **Objective**: Learn the grammar of reasoning with masked entities
- **Config**: batch=8, lr=2e-4, max_seq=2048
- **Duration**: ~10k steps until loss < 1.0

#### Stage 2: SimPO (Simple Preference Optimization)
- **Method**: Reference-free preference learning (CPOTrainer from TRL)
- **Data**: 1,457 preference pairs with 10 persona variants (NLI-verified for semantic equivalence)
- **Hyperparameters**:
  - beta=2.0 (regularization strength)
  - Task-specific gamma: L1=0.5, L2=1.0, **L3=0.3**, L4=0.8
  - Replay buffer: 20% L1, 15% L3, 10% L4 (prevents catastrophic forgetting)
- **Innovation**: Lower gamma for L3 prevents over-penalizing short valid answers

#### Stage 3: DAFT (Domain Adversarial Fine-Tuning)
- **Objective**: Format-invariant representations via gradient reversal
- **Domains**: 11 writing styles (academic, casual, technical, legal, terse, verbose, etc.)
- **Lambda schedule**: 0.1 → 1.0 over 3000 steps
- **Target**: Model cannot predict which domain a problem came from (≤9.1% accuracy)

### Hardware

- **Training**: NVIDIA H200 (141GB VRAM), batch=16, ~3s/step
- **Total time**: ~4 hours for SimPO + DAFT (SFT completed previously)

## Evaluation

### Reasoning Benchmarks

| Level | Task | Score | Target |
|-------|------|-------|--------|
| L1 | Knowledge Override | 85% | ≥85% ✓ |
| L2 | Counterfactual Physics | 45% | ≥75% ✗ |
| L3 | Syllogistic Logic | 85% | ≥85% ✓ |
| L4 | Code Constraints | 90% | ≥90% ✓ |

### Format Invariance

| Test | Score | Target |
|------|-------|--------|
| Cross-format consistency | 100% | 100% ✓ |
| Cue ablation degradation | 0% | <5% ✓ |
| Worst-domain accuracy | 85% | ≥85% ✓ |

**Cross-format consistency**: Model gives identical answers when the same problem is rephrased in 11 different writing styles.

**Cue ablation**: Model maintains performance when explicit reasoning cues (e.g., "think step by step") are removed.

**Worst-domain accuracy**: No single writing style causes the model to collapse.

## Usage

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-Coder-3B-Instruct",
    load_in_4bit=True,
    device_map="auto"
)

# Load Adam's LoRA adapters
model = PeftModel.from_pretrained(base_model, "catbelly/adam-2.7b-v1")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-3B-Instruct")

# Example: Syllogistic reasoning
prompt = """<|begin_of_thought|>
PREMISES:
- All glorps are frimbats
- Zyx is a glorp

QUESTION: Is Zyx a frimbat?

Determine if the conclusion follows logically. Answer: PROVED, DISPROVED, or UNKNOWN.
<|end_of_thought|>"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.7)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

Expected output:
```
PROVED - Yes, Zyx is a frimbat.

Explanation: This is valid modus ponens. If all glorps are frimbats, and Zyx is a glorp, then Zyx must be a frimbat.
```

## Limitations and Biases

### Known Limitations

1. **L2 Physics**: Only 45% accuracy on counterfactual physics (needs improvement)
2. **No world knowledge**: Adam cannot answer "What is the capital of France?" unless you provide that information in context
3. **Short context**: Trained on sequences ≤2048 tokens (future work: extend to 32k)
4. **English-only**: Training data is exclusively English

### Biases

- **Code-centric**: Base model is Qwen2.5-Coder, so Adam may favor code-like reasoning
- **Formal logic bias**: Training on syllogisms may make Adam overly pedantic about argument structure
- **Entity masking artifacts**: Model may struggle with real-world entity relationships it never saw during training

## Ethical Considerations

### Intended Benefits

- **Reduced hallucination**: Adam cannot hallucinate facts it never learned
- **Transparency**: Reasoning is explicit and traceable through `<|begin_of_thought|>` tags
- **Auditability**: All answers derive from provided context, making decisions explainable

### Potential Risks

- **Over-reliance on context**: Adam will blindly follow incorrect premises (this is by design, but could be misused)
- **Logical fallacies**: If the context contains flawed reasoning, Adam may propagate it
- **Misuse for deception**: Could be used to generate "logically sound" arguments from false premises

### Recommendations

- Always validate the premises you provide to Adam
- Use Adam as a reasoning assistant, not a source of truth
- Combine with fact-checking systems for real-world applications

## Citation

If you use Adam in your research:

```bibtex
@software{adam2026,
  title={Adam: A Context-First Reasoning Core via Parametric Ignorance},
  author={Catbelly Studio},
  year={2026},
  url={https://huggingface.co/catbelly/adam-2.7b-v1}
}
```

## Model Card Authors

Catbelly Studio

## Model Card Contact

For questions or feedback: https://github.com/catbelly-studio/adam/issues

---

**Developed with obsession by Catbelly Studio**
