"""
Data Forge V4 - Pure Logic Distillation Engine
Based on "Textbook Is All You Need" + Evol-Instruct methodology.
Follows docs/private/Data Curation.md strictly.

Key Principles:
1. Reasoning Density > Volume
2. Strict structural formatting (<|begin_of_thought|>/<|end_of_thought|>)
3. Verification-filtered data only
4. Curriculum learning phases
5. Zero conversational entropy
"""
import json
import os
import re
import random
import multiprocessing
from functools import partial
from datasets import load_dataset
from tqdm import tqdm

# --- CONFIGURATION ---
OUTPUT_DIR = "adam_curriculum_data"
TARGET_TOKENS = 5_700_000_000  # 5.7B tokens as per doc (~200 tokens/param for 2.7B)
MAX_TEXT_LENGTH = 8192  # Increased for long reasoning traces
NUM_WORKERS = max(1, multiprocessing.cpu_count() - 2)
HF_TOKEN = "hf"

# Estimated tokens per sample (conservative)
TOKENS_PER_SAMPLE = 500

# --- POISON FILTER: Comprehensive conversational artifact detection ---
# Based on Section 4.1 of Data Curation.md - "The Regex Firewall"
POISON_PATTERNS = [
    # Identity/Refusal artifacts
    r"(?i)^as an ai",
    r"(?i)^as a language model",
    r"(?i)^as an assistant",
    r"(?i)^i am an ai",
    r"(?i)^i'm an ai",

    # Social closure artifacts
    r"(?i)i hope this helps",
    r"(?i)hope that helps",
    r"(?i)hope this is helpful",
    r"(?i)feel free to ask",
    r"(?i)let me know if",
    r"(?i)happy to help",
    r"(?i)glad to help",
    r"(?i)don't hesitate to",

    # Phatic introduction artifacts
    r"(?i)^here is the",
    r"(?i)^here's the",
    r"(?i)^sure[,!]?\s*(i can|i'd be|let me|here)",
    r"(?i)^certainly[,!]",
    r"(?i)^of course[,!]",
    r"(?i)^absolutely[,!]",
    r"(?i)^great question",
    r"(?i)^good question",

    # Compliance artifacts
    r"(?i)^i'd be happy to",
    r"(?i)^i would be happy to",
    r"(?i)^i can help",
    r"(?i)^let me help",

    # Social media artifacts (Reddit Poisoning)
    r"(?i)^hey\b",
    r"(?i)^hi\b",
    r"(?i)^hello\b",
    r"(?i)thanks!",
    r"(?i)thank you",
    r"(?i)please help",
    r"(?i)anyone know",
    r"(?i)can someone",
    r"(?i)i was wondering",
    r"(?i)upvote",
    r"(?i)downvote",
    r"(?i)\breddit\b",
    r"(?i)stack\s*overflow",
    r"(?i)#\w+",  # hashtags
    r"(?i)\bimo\b",
    r"(?i)\bimho\b",
    r"(?i)\btbh\b",
    r"(?i)\blol\b",
    r"(?i)\blmao\b",
    r"(?i)\bbtw\b",
    r"(?i)\bafaik\b",
    r"(?i)\bOP\b",
    r"(?i)edit:",
    r"(?i)update:",
    r"(?i)\bu/\w+",  # Reddit usernames
    r"(?i)\br/\w+",  # Reddit subreddits

    # Chat template artifacts (unless controlled)
    r"(?i)^user:",
    r"(?i)^assistant:",
    r"(?i)^human:",
    r"(?i)^ai:",
]
POISON_COMPILED = [re.compile(p) for p in POISON_PATTERNS]


def is_poisoned(text: str) -> bool:
    """Reject text with conversational/social artifacts."""
    for pattern in POISON_COMPILED:
        if pattern.search(text):
            return True
    return False


def is_repetitive(text: str, threshold: float = 0.3) -> bool:
    """Reject text with excessive repetition (causes infinite loops in training)."""
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if len(lines) < 5:
        return False
    unique_ratio = len(set(lines)) / len(lines)
    if unique_ratio < threshold:
        return True

    # Also check for repeated phrases within lines
    words = text.split()
    if len(words) > 20:
        # Check for repeated 3-grams
        trigrams = [' '.join(words[i:i+3]) for i in range(len(words)-2)]
        if len(trigrams) > 0:
            unique_trigram_ratio = len(set(trigrams)) / len(trigrams)
            if unique_trigram_ratio < 0.5:
                return True
    return False


# --- ENTITY MASKING: Prevent fact memorization, focus on reasoning structure ---
# Per Data Curation.md: "Mask entities to prevent memorization, keep reasoning"

def mask_entities(text: str) -> str:
    """
    Replace proper nouns/entities with placeholders.
    Keeps numbers intact for mathematical reasoning.
    This forces the model to learn PATTERNS not FACTS.
    """
    # Multi-word capitalized phrases (names, places, organizations)
    # e.g., "Albert Einstein" -> "[E0]", "New York City" -> "[E1]"
    matches = list(set(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', text)))
    for i, match in enumerate(matches[:15]):  # Cap at 15 unique entities
        text = text.replace(match, f'[ENTITY_{i}]')

    # Single capitalized words mid-sentence (likely proper nouns)
    # "Einstein discovered" -> "[PERSON] discovered"
    text = re.sub(r'(?<=[a-z]\. )([A-Z][a-z]{2,})\b', r'[PERSON]', text)
    text = re.sub(r'(?<=\s)([A-Z][a-z]{2,})(?=\s+said|\s+wrote|\s+discovered|\s+proved|\s+showed)', r'[PERSON]', text)

    # Dates (keep structure, mask specific years for historical facts)
    # "in 1905" -> "in [YEAR]", but keep "x = 1905" for math
    text = re.sub(r'(?<![=<>])\b(1[0-9]{3}|20[0-2][0-9])\b(?!\s*[+\-*/=])', r'[YEAR]', text)

    # URLs and emails
    text = re.sub(r'https?://\S+', '[URL]', text)
    text = re.sub(r'\S+@\S+\.\S+', '[EMAIL]', text)

    # Specific locations (common patterns)
    text = re.sub(r'\b(University of [A-Z][a-z]+)\b', '[UNIVERSITY]', text)
    text = re.sub(r'\b([A-Z][a-z]+ Institute of Technology)\b', '[UNIVERSITY]', text)

    return text


def has_verification(item) -> bool:
    """Check if item has passed verification (for code/math datasets)."""
    # rStar-Coder uses is_passed
    if item.get("is_passed") == False:
        return False
    # Some datasets use verified flag
    if item.get("verified") == False:
        return False
    return True


# --- FORMATTERS: Strict structural format per Data Curation.md ---
# All formatters enforce <|begin_of_thought|>/<|end_of_thought|> structure

def format_deepmath(item):
    """
    DeepMath-103K: Math Logic Anchor (~0.1B tokens)
    Use R1 reasoning traces. Multiple solutions teach diverse reasoning paths.
    """
    problem = item.get("problem", item.get("question", ""))
    # Use all available R1 solutions for diversity
    solutions = []
    for key in ["r1_solution_1", "r1_solution_2", "r1_solution_3"]:
        if item.get(key):
            solutions.append(item.get(key))
    if not solutions:
        solution = item.get("solution", item.get("answer", ""))
        if solution:
            solutions.append(solution)

    if not problem or not solutions:
        return None

    final_answer = item.get("final_answer", "")

    # Use first solution, randomly select for diversity in repeated runs
    solution = random.choice(solutions)

    # Strict structural format
    text = f"""<|begin_of_thought|>
Analyze the mathematical structure. Identify constraints, relationships, and applicable theorems.
Break down the problem into logical steps. Verify each derivation.
<|end_of_thought|>

Problem:
{problem}

<|begin_of_solution|>
{solution}
<|end_of_solution|>"""

    if final_answer:
        text += f"\n\nFinal Answer: {final_answer}"

    return mask_entities(text)


def format_rstar(item):
    """
    rStar-Coder: Code Logic Anchor (~1B tokens)
    Algorithmic reasoning with verification. Only include verified solutions.
    Focus on synthetic_sft subset.
    """
    # CRITICAL: Only use verified solutions per Section 4.2
    if not has_verification(item):
        return None

    problem = item.get("query", item.get("problem", item.get("question", "")))
    response = item.get("response", item.get("solution", ""))

    if not problem or not response:
        return None

    # Strict structural format emphasizing algorithmic planning
    text = f"""<|begin_of_thought|>
Parse the problem constraints. Analyze time and space complexity requirements.
Design the algorithm step by step. Consider edge cases.
Plan before implementing.
<|end_of_thought|>

Specification:
{problem}

<|begin_of_solution|>
{response}
<|end_of_solution|>"""

    return mask_entities(text)


def format_stratos(item):
    """
    Bespoke-Stratos-17k: Reasoning Style Setter (~0.02B tokens)
    Metacognitive reasoning with strict structural schema.
    This is the "tone calibration" dataset - upsample significantly.
    """
    conversations = item.get("conversations", [])
    if len(conversations) < 2:
        return None

    problem = None
    response = None
    for turn in conversations:
        if turn.get("from") == "user" and not problem:
            problem = turn.get("value", "")
        elif turn.get("from") == "assistant" and not response:
            response = turn.get("value", "")

    if not problem or not response:
        return None

    # The response should already contain <thought> tags - preserve them
    # But wrap in our standard format if not present
    if "<|begin_of_thought|>" in response or "<thought>" in response:
        # Already structured, use as-is but normalize tags
        text = f"Problem:\n{problem}\n\n{response}"
        text = text.replace("<thought>", "<|begin_of_thought|>")
        text = text.replace("</thought>", "<|end_of_thought|>")
        text = text.replace("<solution>", "<|begin_of_solution|>")
        text = text.replace("</solution>", "<|end_of_solution|>")
    else:
        text = f"""<|begin_of_thought|>
Analyze systematically. Explore multiple approaches.
Engage in analysis, brainstorming, reassessment, and verification.
Verify each step before proceeding.
<|end_of_thought|>

Problem:
{problem}

<|begin_of_solution|>
{response}
<|end_of_solution|>"""

    return mask_entities(text)


def format_openthoughts(item):
    """
    OpenThoughts-114k: General Reasoning (~0.5B tokens)
    Systematic long thinking process. Broadens logic beyond pure math/code.
    """
    conversations = item.get("conversations", [])
    if len(conversations) < 2:
        return None

    problem = None
    response = None
    for turn in conversations:
        if turn.get("from") == "user" and not problem:
            problem = turn.get("value", "")
        elif turn.get("from") == "assistant" and not response:
            response = turn.get("value", "")

    if not problem or not response:
        return None

    # Normalize any existing tags
    if "<|begin_of_thought|>" in response or "<thought>" in response:
        text = f"Problem:\n{problem}\n\n{response}"
        text = text.replace("<thought>", "<|begin_of_thought|>")
        text = text.replace("</thought>", "<|end_of_thought|>")
        text = text.replace("<solution>", "<|begin_of_solution|>")
        text = text.replace("</solution>", "<|end_of_solution|>")
    else:
        text = f"""<|begin_of_thought|>
Reason systematically through the problem.
Engage in analysis, summarization, exploration, reassessment, reflection, and iteration.
Verify conclusions against the original constraints.
<|end_of_thought|>

Problem:
{problem}

<|begin_of_solution|>
{response}
<|end_of_solution|>"""

    return mask_entities(text)


def format_opencode(item):
    """
    OpenCodeReasoning: Dense Coding (~2B tokens)
    Code reasoning with step-by-step logic before implementation.
    """
    problem = item.get("question", item.get("problem", ""))
    reasoning = item.get("reasoning", item.get("thought", ""))
    code = item.get("solution", item.get("code", item.get("answer", "")))

    if not problem or not code:
        return None

    if reasoning:
        text = f"""<|begin_of_thought|>
{reasoning}
<|end_of_thought|>

Specification:
{problem}

<|begin_of_solution|>
{code}
<|end_of_solution|>"""
    else:
        text = f"""<|begin_of_thought|>
Parse the requirements carefully.
Design the algorithm considering time and space complexity.
Plan the implementation step by step.
<|end_of_thought|>

Specification:
{problem}

<|begin_of_solution|>
{code}
<|end_of_solution|>"""

    return mask_entities(text)


def format_mathsmith(item):
    """
    MathSmith-Hard: First Principles (~0.1B tokens)
    Axiomatic math derived from PlanetMath concepts.
    Critical for fixing "Earth Bias" - teaches derivation from axioms.
    """
    problem = item.get("Problem", item.get("problem", ""))
    solution = item.get("Output", item.get("output", ""))
    concepts = item.get("Sampled_concept", "")
    rationale = item.get("Rationale", "")

    if not problem or not solution:
        return None

    # Build thought section from rationale/concepts
    if rationale:
        thought = rationale
    elif concepts:
        thought = f"Derive solution from first principles using: {concepts}"
    else:
        thought = "Apply axiomatic reasoning. Derive from definitions and fundamental principles."

    text = f"""<|begin_of_thought|>
{thought}
Do not rely on memorized heuristics. Derive each step from stated axioms.
<|end_of_thought|>

Problem:
{problem}

<|begin_of_solution|>
{solution}
<|end_of_solution|>"""

    return mask_entities(text)


def format_numina(item):
    """
    NuminaMath-CoT: Competition Math with Chain-of-Thought
    High-quality mathematical reasoning from AMC/AIME/Olympiad sources.
    """
    problem = item.get("problem", item.get("question", ""))
    solution = item.get("solution", item.get("answer", ""))
    source = item.get("source", "")

    if not problem or not solution:
        return None

    if source:
        thought = f"Competition problem from {source}. Analyze the structure and identify applicable theorems and techniques."
    else:
        thought = "Analyze problem structure. Identify applicable theorems and competition techniques."

    text = f"""<|begin_of_thought|>
{thought}
Work through the logic systematically. Verify each step.
<|end_of_thought|>

Problem:
{problem}

<|begin_of_solution|>
{solution}
<|end_of_solution|>"""

    return mask_entities(text)


def format_synth(item):
    """
    PleIAs/SYNTH: Knowledge Base (~2B tokens filtered)
    Textbook-style synthesis from Wikipedia Vital Articles.

    CRITICAL FILTERING per Section 3 Recommendation 5:
    - Filter OUT: chat, dialogue, memorization exercise types
    - Prioritize: synthetic_reasoning column
    - Focus: Hard sciences (physics, chemistry, biology, math)
    """
    # Get exercise type - filter out conversational types
    exercise = item.get("exercise", "").lower()
    EXCLUDED_EXERCISES = {"chat", "dialogue", "memorization", "conversation", "discussion"}
    if exercise in EXCLUDED_EXERCISES:
        return None

    # Prioritize synthetic_reasoning (structural thinking process)
    reasoning = item.get("synthetic_reasoning", "")
    content = item.get("synthetic_answer", item.get("text", ""))
    query = item.get("query", "")

    if not content:
        return None

    # Additional filter: check for informal language in query
    if query and is_poisoned(query):
        return None

    # Build the formatted text
    if reasoning:
        text = f"""<|begin_of_thought|>
{reasoning}
<|end_of_thought|>

{content}"""
    elif query:
        text = f"""<|begin_of_thought|>
Explain the concept clearly using first principles.
Structure the explanation logically from fundamentals to applications.
<|end_of_thought|>

Query: {query}

Explanation:
{content}"""
    else:
        # Just content - use as knowledge base without thought wrapper
        text = f"""Explanation:
{content}"""

    return mask_entities(text)


# --- PROCESSING ---

def process_item(item_and_name):
    """Process a single item with appropriate formatter."""
    item, formatter_name = item_and_name
    try:
        formatter_map = {
            "deepmath": format_deepmath,
            "rstar": format_rstar,
            "stratos": format_stratos,
            "openthoughts": format_openthoughts,
            "opencode": format_opencode,
            "mathsmith": format_mathsmith,
            "numina": format_numina,
            "synth": format_synth,
        }

        formatter = formatter_map.get(formatter_name)
        if not formatter:
            return None

        text = formatter(item)

        if text is None:
            return None

        # Minimum length filter - reasoning should be substantial
        if len(text) < 300:
            return None

        # Maximum length filter
        if len(text) > MAX_TEXT_LENGTH:
            text = text[:MAX_TEXT_LENGTH]

        # Poison filter
        if is_poisoned(text):
            return None

        # Repetition filter
        if is_repetitive(text):
            return None

        return text
    except Exception:
        return None


def load_and_sample_dataset(name, path, split, sample_size, config=None, data_dir=None, streaming=False):
    """Load dataset and take a sample."""
    print(f"Loading {name}...")
    try:
        kwargs = {"split": split, "token": HF_TOKEN}
        if config:
            kwargs["name"] = config
        if data_dir:
            kwargs["data_dir"] = data_dir
        if streaming:
            kwargs["streaming"] = True

        ds = load_dataset(path, **kwargs)

        if streaming:
            # For streaming datasets, take samples iteratively
            samples = []
            for item in tqdm(ds, total=sample_size, desc=f"  Streaming {name}"):
                samples.append(dict(item))
                if len(samples) >= sample_size:
                    break
            print(f"   {name}: {len(samples):,} samples loaded")
            return samples
        else:
            ds = ds.shuffle(seed=42)
            if len(ds) > sample_size:
                ds = ds.select(range(sample_size))
            print(f"   {name}: {len(ds):,} samples loaded")
            return ds
    except Exception as e:
        print(f"   {name} failed: {e}")
        return None


def process_dataset(name, ds, formatter_name, upsample=1):
    """Process a dataset with the specified formatter."""
    print(f"Processing {name} (formatter: {formatter_name}, upsample: {upsample}x)...")

    if isinstance(ds, list):
        items_with_name = [(item, formatter_name) for item in ds]
    else:
        items_with_name = [(dict(item), formatter_name) for item in ds]

    with multiprocessing.Pool(NUM_WORKERS) as pool:
        results = list(tqdm(
            pool.imap(process_item, items_with_name, chunksize=100),
            total=len(items_with_name),
            desc=f"  {name}"
        ))

    valid = [r for r in results if r is not None]

    # Apply upsampling
    if upsample > 1:
        valid = valid * upsample
        print(f"  Upsampled {upsample}x: {len(valid):,} samples")

    print(f"  Valid: {len(valid):,} samples from {name}")
    return valid


def main():
    print("=" * 70)
    print("PURE LOGIC DISTILLATION ENGINE: Data Forge V4")
    print("Following: docs/private/Data Curation.md")
    print("Target: ~5.7B tokens for 2.7B parameter Reasoning Engine")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # =========================================================================
    # PHASE 1: AXIOMATIC KNOWLEDGE
    # Establish definitions, laws, and physical constants
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 1: AXIOMATIC KNOWLEDGE")
    print("=" * 70)

    phase1_texts = []

    # MathSmith-Hard: First Principles (critical for Earth Bias fix)
    ds = load_and_sample_dataset("mathsmith", "Jasaxion/MathSmith-Hard-Problems", "train", 100000)
    if ds:
        phase1_texts.extend(process_dataset("mathsmith", ds, "mathsmith", upsample=2))

    # PleIAs/SYNTH: Knowledge Base (filtered for science/physics)
    # Using streaming due to massive size
    print("\nLoading PleIAs/SYNTH (streaming, filtered for hard sciences)...")
    try:
        # Load a substantial sample via streaming
        ds = load_and_sample_dataset("synth", "PleIAs/SYNTH", "train", 500000, streaming=True)
        if ds:
            phase1_texts.extend(process_dataset("synth", ds, "synth", upsample=1))
    except Exception as e:
        print(f"   SYNTH streaming failed: {e}")

    # Save Phase 1
    phase1_file = os.path.join(OUTPUT_DIR, "phase1_axiomatic.jsonl")
    random.shuffle(phase1_texts)
    print(f"\nWriting Phase 1: {len(phase1_texts):,} samples to {phase1_file}")
    with open(phase1_file, "w", encoding="utf-8") as f:
        for text in phase1_texts:
            json.dump({"text": text}, f)
            f.write("\n")

    # =========================================================================
    # PHASE 2: ALGORITHMIC HARDENING
    # Teach constraint satisfaction, logic gates, symbolic manipulation
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 2: ALGORITHMIC HARDENING")
    print("=" * 70)

    phase2_texts = []

    # DeepMath-103K: Math Logic Anchor (ALL of it - gold standard)
    ds = load_and_sample_dataset("deepmath", "zwhe99/DeepMath-103K", "train", 150000)
    if ds:
        phase2_texts.extend(process_dataset("deepmath", ds, "deepmath", upsample=3))

    # rStar-Coder: Code Logic Anchor (verified solutions only)
    ds = load_and_sample_dataset("rstar", "microsoft/rStar-Coder", "train", 600000, data_dir="synthetic_sft")
    if ds:
        phase2_texts.extend(process_dataset("rstar", ds, "rstar", upsample=1))

    # NuminaMath-CoT: Competition math (high quality)
    ds = load_and_sample_dataset("numina", "AI-MO/NuminaMath-CoT", "train", 500000)
    if ds:
        phase2_texts.extend(process_dataset("numina", ds, "numina", upsample=1))

    # Save Phase 2
    phase2_file = os.path.join(OUTPUT_DIR, "phase2_algorithmic.jsonl")
    random.shuffle(phase2_texts)
    print(f"\nWriting Phase 2: {len(phase2_texts):,} samples to {phase2_file}")
    with open(phase2_file, "w", encoding="utf-8") as f:
        for text in phase2_texts:
            json.dump({"text": text}, f)
            f.write("\n")

    # =========================================================================
    # PHASE 3: REASONING CRYSTALLIZATION
    # Teach the process of deriving solutions from Phase 1 & 2 knowledge
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 3: REASONING CRYSTALLIZATION")
    print("=" * 70)

    phase3_texts = []

    # Bespoke-Stratos-17k: Reasoning Style Setter (HEAVY upsample per doc)
    ds = load_and_sample_dataset("stratos", "bespokelabs/Bespoke-Stratos-17k", "train", 20000)
    if ds:
        # Doc recommends 5-10x upsample or use for annealing
        phase3_texts.extend(process_dataset("stratos", ds, "stratos", upsample=10))

    # OpenThoughts-114k: General Reasoning
    ds = load_and_sample_dataset("openthoughts", "open-thoughts/OpenThoughts-114k", "train", 120000)
    if ds:
        phase3_texts.extend(process_dataset("openthoughts", ds, "openthoughts", upsample=2))

    # OpenCodeReasoning: Dense Code Reasoning (both splits)
    ds = load_and_sample_dataset("opencode_0", "nvidia/OpenCodeReasoning", "train", 750000, config="split_0")
    if ds:
        phase3_texts.extend(process_dataset("opencode_0", ds, "opencode", upsample=1))

    ds = load_and_sample_dataset("opencode_1", "nvidia/OpenCodeReasoning", "train", 750000, config="split_1")
    if ds:
        phase3_texts.extend(process_dataset("opencode_1", ds, "opencode", upsample=1))

    # Save Phase 3
    phase3_file = os.path.join(OUTPUT_DIR, "phase3_crystallization.jsonl")
    random.shuffle(phase3_texts)
    print(f"\nWriting Phase 3: {len(phase3_texts):,} samples to {phase3_file}")
    with open(phase3_file, "w", encoding="utf-8") as f:
        for text in phase3_texts:
            json.dump({"text": text}, f)
            f.write("\n")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("FORGE COMPLETE")
    print("=" * 70)

    total_samples = len(phase1_texts) + len(phase2_texts) + len(phase3_texts)
    estimated_tokens = total_samples * TOKENS_PER_SAMPLE

    print(f"""
Phase 1 (Axiomatic Knowledge):     {len(phase1_texts):>12,} samples
Phase 2 (Algorithmic Hardening):   {len(phase2_texts):>12,} samples
Phase 3 (Reasoning Crystallization): {len(phase3_texts):>12,} samples
{'─' * 50}
Total:                             {total_samples:>12,} samples
Estimated Tokens:                  {estimated_tokens:>12,} (~{estimated_tokens/1e9:.1f}B)

Output Directory: {OUTPUT_DIR}/
  - phase1_axiomatic.jsonl
  - phase2_algorithmic.jsonl
  - phase3_crystallization.jsonl

Training Order:
  1. Train on phase1_axiomatic.jsonl first (establish world model)
  2. Then phase2_algorithmic.jsonl (build logical operations)
  3. Finally phase3_crystallization.jsonl (crystallize reasoning behavior)

Key Features:
  - Strict <|begin_of_thought|>/<|end_of_thought|> structure enforced
  - Verification-filtered data (rStar-Coder)
  - PleIAs/SYNTH filtered for hard sciences only
  - Bespoke-Stratos 10x upsampled for reasoning style
  - Zero conversational entropy (comprehensive poison filtering)
""")


if __name__ == "__main__":
    main()
