# Research Brief: Adam v2 Architecture & Training Design

## CRITICAL DIRECTIVE — READ FIRST

**Do NOT ask any clarifying questions. Do NOT request more information. Start researching and writing immediately.**

All context you need is provided below. The answers to any question you might ask are:

| Question you might ask | Answer |
|------------------------|--------|
| Deployment target? | Cloud API inference, no edge constraints |
| Model size limits? | 1B–14B acceptable, optimize for quality first |
| Latency constraints? | None — batch inference only |
| Prioritize efficiency or performance? | **Performance first.** Budget: <$20/run, <8hrs on H200 |
| Framework? | PyTorch + HuggingFace TRL 0.28+. No JAX, no TensorFlow |
| Specific papers to prioritize? | DeepSeek-R1/R1-Zero (GRPO), rStar-Math, STILL-2, Quiet-STaR, ORPO, Dr. GRPO — but search for others too |

Your job is to **produce the document**, not to gather requirements. Begin with Section 1 immediately.

---

## Role

You are a senior ML research engineer tasked with producing a **concrete implementation plan** for Adam v2. Your output must be a design document with specific, justified recommendations — not a list of open questions. Every recommendation must end in a clear decision ("use X, not Y, because Z"). You will search recent papers (2023–2026), compare approaches, and synthesize findings into an actionable spec.

---

## Background: Adam v1 Status

Adam is a 2.7B parameter reasoning model (Qwen2.5-Coder-3B base, QLoRA 4-bit NF4) trained to exhibit **Parametric Ignorance**: prioritize in-context instructions over pretrained knowledge.

### Current Empirical Results

| Task | Description | v1 Accuracy | Target |
|------|-------------|-------------|--------|
| L1 | Knowledge override (follow false premises) | 85% | ≥90% |
| L2 | Counterfactual physics (altered constants) | 45% | ≥75% |
| L3 | Syllogistic logic (PROVED/DISPROVED/UNKNOWN) | unstable (oscillates 10%→85%→0%) | ≥90% stable |
| L4 | Code under constraints (no built-ins) | 90% | ≥95% |

### Current Training Pipeline (v1)
```
SFT (10k steps, causal LM)
  → SimPO / CPO (2k steps, reference-free preference optimization, TRL)
    → DAFT (3k steps, gradient reversal for domain/style invariance)
```

### Known Failure Modes
- **L3 instability**: Model oscillates between "always UNKNOWN" and "always PROVED" across training runs. Root causes: data imbalance, catastrophic forgetting during SimPO, wrong gamma for short-answer tasks
- **L2 at 45%**: Multi-step numerical reasoning under counterfactual physics fails. Unclear if this is a model size issue, data quality issue, or inference pattern issue
- **Catastrophic forgetting**: SimPO preference training reliably degrades SFT task performance without replay injection
- **Replay buffer fragility**: Pre-mixing replay into dataset is brittle vs. dynamic curriculum

---

## Deliverables Required

Produce a complete Adam v2 design document covering all six sections below. For each section, state a clear **RECOMMENDATION** with justification citing specific papers or empirical evidence.

---

## Section 1: Base Architecture Selection

### 1a. Model Family & Pretraining Corpus

Evaluate and recommend between:
- Dense transformer (Qwen2.5, Llama-3.x, Phi-4, Mistral/Mixtral, Gemma-2)
- State-space / linear-attention alternatives (Mamba-2, RWKV-6, Griffin)
- Hybrid architectures (Jamba, Zamba, RecurrentGemma)

For each family, assess:
1. **Pretraining corpus composition** — does heavy code/math pretraining help or hurt L1 (knowledge override) and L2 (counterfactual arithmetic)?
2. **Attention pattern suitability** — sliding window vs. full attention vs. linear for multi-hop reasoning chains
3. **KV cache efficiency** — matters for long CoT traces in L2

**Specific question**: Is Qwen2.5-Coder-3B optimal, or does its code-heavy pretraining make L1 harder (model "knows" facts as code comments)?

**RECOMMENDATION REQUIRED**: Single model family + size + justification

---

### 1b. Model Size vs. Reasoning Capacity

Analyze the scaling relationship between model size and reasoning stability for L1-L4. Key questions:
- Larger models have more parametric knowledge — does this make L1 override *harder* or *easier* to train?
- Is L2's 45% a parameter count problem or a training methodology problem? Cite evidence.
- What is the smallest model that can reliably do multi-step syllogistic reasoning (L3)?

Reference: Explore literature on "reasoning emergence thresholds" and small model reasoning (e.g., Phi-4 3.8B outperforming 7B models, rStar-Math findings on small model reasoning)

**RECOMMENDATION REQUIRED**: Optimal parameter count (1B / 3B / 7B / 14B) with size-accuracy tradeoff analysis

---

### 1c. PEFT Strategy

Compare:
- QLoRA 4-bit NF4 (current) — r=64, alpha=128
- QLoRA 8-bit — better numerical precision for L2?
- Full fine-tuning (small model, e.g., 1.5B)
- LoRA without quantization (bf16)
- DoRA (Weight-Decomposed LoRA) — improved fine-tuning stability
- IA3 — fewer parameters, different inductive bias

**Specific question**: Does 4-bit NF4 quantization degrade L2's numerical precision enough to explain the 45% floor? Cite evidence from quantization literature.

**RECOMMENDATION REQUIRED**: Exact PEFT configuration (method + bits + rank + alpha)

---

## Section 2: Training Loop Design

This is the most critical section. Design the complete training loop for Adam v2, including novel training objectives not used in v1.

### 2a. Stage 1 Replacement: What Replaces SFT?

Evaluate alternatives to vanilla SFT for initial skill learning:

**Option A: SFT with curriculum ordering**
- Easy examples first (modus ponens L3) → hard examples last (multi-step counterfactual L2)
- Within each task level, sort by reasoning chain length

**Option B: SFT + Self-Play from day 1**
- Generate model outputs, verify with rule-based checker (for L3: valid/invalid syllogisms are decidable)
- Train on (prompt, correct_answer) pairs that the model *almost* gets right
- Reference: STaR (Self-Taught Reasoner), rStar, Quiet-STaR

**Option C: Online SFT with verifier**
- For L3 and L4 (rule-checkable tasks), generate candidate outputs during training
- Accept only verified-correct completions into training batch
- Reference: DeepSeek-R1, STILL-2, OpenR

**Option D: Direct Skill Injection (no SFT)**
- Skip SFT entirely; use RL from base model (like DeepSeek-R1-Zero approach)
- Use GRPO (Group Relative Policy Optimization) with rule-based reward
- Only works if base model already has latent reasoning capability

For each option:
1. State whether it's feasible for each of L1/L2/L3/L4 (L3 and L4 are rule-verifiable; L1 and L2 are not)
2. Estimate compute overhead vs SFT baseline
3. Cite relevant papers

**RECOMMENDATION REQUIRED**: Stage 1 training method with pseudocode for the training loop

---

### 2b. Stage 2: Preference Optimization

Evaluate and compare modern preference optimization methods for the parametric ignorance use case:

**Current**: SimPO / CPO (reference-free, gamma length penalty, beta=2.0)

**Alternatives to evaluate**:
- **DPO** (Direct Preference Optimization) — requires reference model; more stable?
- **ORPO** (Odds Ratio Preference Optimization) — single-stage SFT+preference, no reference model needed
- **KTO** (Kahneman-Tversky Optimization) — works on unpaired feedback, just (prompt, output, label) format
- **IPO** (Identity Preference Optimization) — addresses DPO overfitting
- **TDPO** (Token-level DPO) — token-level rather than sequence-level reward
- **GRPO** (Group Relative Policy Optimization) — used in DeepSeek-R1; generates multiple completions per prompt, uses group mean as baseline instead of reference model
- **Online DPO / Iterative DPO** — generate fresh preference pairs during training using current policy
- **RLOO** (REINFORCE Leave One Out) — lower variance RL baseline
- **Dr. GRPO** — fixes GRPO's known bias issues (length bias, minority sample advantage)

For each, assess:
1. Does it require a reference model? (matters for our setup)
2. How does it handle the "answer length varies by task" problem (L3 says "UNKNOWN", L2 needs multi-paragraph math)?
3. Does it work well with small datasets (~1500 preference pairs)?
4. Can it handle multi-task preference data (L1+L2+L3+L4 mixed)?

**Critical question**: GRPO generates K completions per prompt and uses within-group statistics. For L3 (decidable syllogisms), we could verify correctness programmatically. Does this make GRPO with rule-based reward the right choice for Stage 2 L3 specifically?

**RECOMMENDATION REQUIRED**: Exact preference optimization method per task level (can be different for L3 vs L2), with training loop pseudocode

---

### 2c. Catastrophic Forgetting Mitigation

Design the complete strategy for maintaining L1-L4 performance throughout all training stages.

**Current**: Pre-mix 15% replay samples into dataset before training (fragile, static)

**Evaluate these approaches**:

1. **Dynamic replay buffer with curriculum**
   - Maintain per-task accuracy tracker using validation probes every N steps
   - Dynamically increase replay ratio for tasks that are degrading
   - Pseudocode: `replay_ratio[task] = base_ratio + k * max(0, target_acc - current_acc[task])`

2. **Elastic Weight Consolidation (EWC)**
   - Compute Fisher information matrix after SFT
   - Penalize changes to parameters important for each task during SimPO
   - Tradeoff: memory overhead, complex implementation

3. **Gradient Episodic Memory (GEM) / A-GEM**
   - Store exemplar gradients from each task
   - Project updates to not increase loss on stored tasks
   - Works at gradient level, not data level

4. **Multi-task learning (MTL) instead of sequential**
   - Merge all stages: SFT objective + preference objective + domain invariance in single loop
   - Loss: `L_total = λ₁·L_sft + λ₂·L_preference + λ₃·L_domain_inv`
   - Dynamic λ annealing: ramp up preference loss as SFT converges
   - Reference: Examine if this removes need for separate stages entirely

5. **Task-Specific LoRA heads**
   - Separate LoRA adapters per task level (L1, L2, L3, L4)
   - Shared base, task-specific delta weights
   - Zero forgetting by construction (tasks can't interfere)
   - Routing: task classifier or explicit task token at inference

6. **Model merging / soup**
   - Train separate models for each task
   - Merge using TIES-merging, DARE, or linear interpolation
   - Reference: Model Merging literature (2024)

**RECOMMENDATION REQUIRED**: Single catastrophic forgetting strategy with implementation spec. If multi-task, provide loss weighting schedule.

---

### 2d. Novel Training Loop for L3 (Syllogistic Logic)

L3 is the hardest task. Design a specialized training loop to solve it.

Key constraints:
- L3 is fully rule-verifiable (syllogisms are decidable in propositional logic)
- 3-class output: PROVED / DISPROVED / UNKNOWN
- Current problem: model collapses to one class

**Design a training loop that**:
1. Uses the verifier (programmatic syllogism checker) to generate training signal
2. Handles the 3-class imbalance problem explicitly
3. Prevents the "collapse to UNKNOWN" failure mode
4. Ensures stable training across runs (not sensitive to data ordering)

**Candidates**:
- GRPO with programmatic reward function
- STaR-style self-play with correctness filter
- RL with shaped reward: +2 for correct label, -1 for wrong label, 0 for "UNKNOWN when answer is determinable"
- Contrastive learning: pull together (valid_syllogism → PROVED), push apart from (invalid_syllogism → PROVED)

Provide the complete training loop pseudocode.

**RECOMMENDATION REQUIRED**: Pseudocode for L3-specific training loop

---

## Section 3: Data Architecture

### 3a. Preference Data Format

Evaluate data formats for the preference optimization stage:

- **Paired (current)**: `(prompt, chosen, rejected)` — standard DPO/SimPO format
- **Best-of-N**: Multiple rejected responses at different quality levels — richer signal
- **Process-level**: Reward at each reasoning step, not just final answer (PRM — Process Reward Model approach)
- **Constitutional**: Multiple criteria per example (correct label + correct reasoning chain)

For L3 specifically:
- Does the model need to see WHY the syllogism is invalid (reasoning chain), or just WHAT the label is?
- Research: Does process-level reward (step-by-step verification) outperform outcome-level reward for logical tasks?

**RECOMMENDATION REQUIRED**: Data format per task level

---

### 3b. L2 Data Redesign (45% → 75%)

L2 (counterfactual physics) is at 45% — the biggest gap. Design a data curriculum:

1. Analyze failure modes: is the model failing at (a) understanding the counterfactual setup, (b) numerical computation, or (c) multi-step reasoning chains?
2. Propose a difficulty ladder from trivial to complex counterfactual reasoning
3. Should we add Chain-of-Thought (CoT) traces to L2 data? (Model generates reasoning steps before final answer)
4. Research: Do small models (<4B) struggle with multi-step arithmetic even with CoT? What does the literature say about numerical reasoning in small models?

**RECOMMENDATION REQUIRED**: L2 data redesign with difficulty ladder and CoT strategy

---

### 3c. Format Invariance (DAFT Replacement)

Current DAFT uses gradient reversal. Evaluate alternatives:

- **DAFT (current)**: Gradient reversal layer, domain classifier, adversarial training
- **Contrastive style-invariance**: Pull same question in different styles together in embedding space
- **Data augmentation only**: Generate 10 persona variants per example, train on all — does this make DAFT unnecessary?
- **Style normalization at inference**: Preprocess input to canonical form before model sees it

**Research question**: Does DAFT actually improve generalization or just overfit to 10 specific personas? Cite any ablations from style-invariant training literature.

**RECOMMENDATION REQUIRED**: Keep DAFT or replace it with what?

---

## Section 4: Reasoning Chain Design

### 4a. Chain-of-Thought Integration

Design the exact CoT format for Adam v2:

Current L3 training data format (mixed — some have CoT, some don't):
```
<|begin_of_thought|>
Step 1: IDENTIFY LOGICAL FORM
Step 2: CHECK VALIDITY
Step 3: CONCLUSION
<|end_of_thought|>
UNKNOWN - Cannot be determined
```

**Questions**:
1. Should ALL tasks use CoT, or only L2 (numerical) and L3 (logical)?
2. Is the current `<|begin_of_thought|>` format optimal, or should we use structured tags per reasoning step?
3. Does CoT help L1 (knowledge override) or hurt it (model might "reason" its way back to pretrained knowledge)?
4. What does the literature say about CoT for small models (<4B)? Does it help or just add noise?

Reference: Quiet-STaR (thinking before speaking), Coconut (continuous CoT in embedding space), DeepSeek-R1 long CoT

**RECOMMENDATION REQUIRED**: Exact CoT format and which tasks use it

---

### 4b. Structured Output Format

Define the exact output schema for each task level. Example:

```
L1: CONTEXT SAYS: [X]. PRETRAINED KNOWLEDGE SAYS: [Y]. USING CONTEXT: [answer]
L2: MODIFIED CONSTANTS: [list]. DERIVATION: [steps]. RESULT: [answer with units]
L3: FORM: [syllogism type]. VALIDITY: [PROVED|DISPROVED|UNKNOWN]. REASON: [one sentence]
L4: CONSTRAINT: [restate]. APPROACH: [alt method]. CODE: [implementation]
```

**Question**: Does enforcing structured output at training time improve accuracy? Or does it constrain the model too much?

Reference: Research on structured output training vs free-form generation

**RECOMMENDATION REQUIRED**: Output format per task level, or justify using free-form

---

## Section 5: Evaluation & Debugging Framework

### 5a. Training-Time Monitoring

Design the complete monitoring system to catch failures during training (not just after):

Required monitors (propose implementation for each):
1. **Per-task accuracy tracker**: Run validation probes every K steps, log per-task accuracy, detect if any task drops >10% in one window
2. **Label distribution tracker**: Detect if model is collapsing to single label (the UNKNOWN collapse problem)
3. **Gradient flow analyzer**: Detect vanishing/exploding gradients per LoRA layer
4. **Replay buffer utilization verifier**: Confirm replay samples are actually being used in training batches (the critical bug from v1)
5. **Forgetting rate measurer**: For each stage transition, measure accuracy delta per task

**RECOMMENDATION REQUIRED**: Complete monitoring spec with trigger conditions and automatic responses (e.g., "if L3 drops >15%, increase L3 replay ratio to 30%")

---

### 5b. Parametric Ignorance Verification

Design a test suite to *prove* the model is using context, not pretrained knowledge.

Current problem: we can't distinguish between "model correctly used context" and "model happened to output the context answer because it was also in pretrained knowledge"

**Design**:
1. Adversarial probe set: Famous facts with altered details (e.g., "According to this document, Newton discovered penicillin in 1687")
2. Zero-knowledge probe set: Novel facts the model cannot have seen in pretraining (randomly generated entity-fact pairs)
3. Contradiction stress test: Context directly contradicts pretrained knowledge → model must pick context
4. Graduated difficulty: Facts with low vs. high pretrained confidence (obscure facts vs. "Einstein won Nobel Prize")

**RECOMMENDATION REQUIRED**: Exact test protocol with examples

---

## Section 6: Final Training Recipe Specification

Based on all sections above, produce the complete Adam v2 training recipe:

```
Base Model:        [FILL FROM SECTION 1a]
Size:              [FILL FROM SECTION 1b]
PEFT:              [FILL FROM SECTION 1c]

Stage 1 (Skill acquisition):
  Method:          [FILL FROM SECTION 2a]
  Steps:           [specify]
  Data:            [specify: what tasks, how many examples, CoT format]
  Loss:            [exact formula]
  Stopping criterion: [what metric, what threshold]

Stage 2 (Preference alignment):
  Method:          [FILL FROM SECTION 2b]
  L1/L2:           [method + hyperparams]
  L3:              [FILL FROM SECTION 2d — may differ from L1/L2]
  L4:              [method + hyperparams]
  Replay strategy: [FILL FROM SECTION 2c]
  Steps:           [specify]

Stage 3 (Format invariance):
  Method:          [FILL FROM SECTION 3c]
  Steps:           [specify, or "eliminated"]

Inference:
  CoT:             [FILL FROM SECTION 4a]
  Output format:   [FILL FROM SECTION 4b]
  Decoding:        [greedy / beam / temperature]
```

Include estimated training cost on H200 ($3.14/hr) and total time.

---

## Output Format Requirements

Structure your response as:

```
## SECTION 1: BASE ARCHITECTURE
**Recommendation**: [1-2 sentences, concrete decision]
**Justification**: [cite papers, explain tradeoffs]
**Rejected alternatives**: [why not X, why not Y]

## SECTION 2: TRAINING LOOP
**Stage 1 Recommendation**: [method name + config]
**Stage 2 Recommendation**: [method name + config, per task]
**Forgetting mitigation**: [method]
**L3 loop pseudocode**:
```python
for batch in dataloader:
    ...
```

## SECTION 3: DATA ARCHITECTURE
...

[continue for all sections]

## FINAL RECIPE
[fill in the complete spec from Section 6]
```

---

## Constraints

- **Compute**: H200 141GB available, target <$20 per training run
- **Time**: Target <8 hours total pipeline
- **Library**: TRL 0.28+ (CPOTrainer, GRPOTrainer, DPOTrainer available), PyTorch 2.x, Transformers 5.x
- **Must be implementable**: No architectural changes that require custom CUDA kernels or new framework builds
- **Must be empirically grounded**: Every recommendation must cite at least one paper published 2023-2026 or our own v1 empirical results

---

## Priority Order

If time-constrained, research sections in this order:
1. **Section 2d** (L3 training loop — highest impact, currently at 0-60% oscillation)
2. **Section 2b** (preference optimization method — replaces SimPO?)
3. **Section 2c** (catastrophic forgetting — systemic problem)
4. **Section 1b** (model size — affects all tasks)
5. **Section 3b** (L2 data redesign — 45% is the biggest gap)
6. **Sections 4, 5, 6** (format, monitoring, recipe)
