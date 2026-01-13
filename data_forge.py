"""
Data Forge V3 - Pure Logic Distillation Engine
Based on "Textbook Is All You Need" + Evol-Instruct methodology.
Uses high-fidelity reasoning datasets, masks entities to prevent memorization.
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
OUTPUT_FILE = "adam_skeleton_data.jsonl"
TARGET_SAMPLES = 5_000_000  # Aim for 5M, expect 3-4M after filtering
MAX_TEXT_LENGTH = 4096
NUM_WORKERS = max(1, multiprocessing.cpu_count() - 2)
HF_TOKEN = "hf"

# --- POISON FILTER: Conversational artifacts (Reddit Poisoning) ---
POISON_PATTERNS = [
    r"(?i)^as an ai",
    r"(?i)i hope this helps",
    r"(?i)^sure[,!]?\s*(i can|i'd be|let me)",
    r"(?i)^here is the",
    r"(?i)^certainly[,!]",
    r"(?i)^of course[,!]",
    r"(?i)hope that helps",
    r"(?i)feel free to ask",
    r"(?i)let me know if",
    r"(?i)happy to help",
    r"(?i)^hey\b",
    r"(?i)^hi\b",
    r"(?i)thanks!",
    r"(?i)thank you",
    r"(?i)please help",
    r"(?i)anyone know",
    r"(?i)can someone",
    r"(?i)i was wondering",
    r"(?i)upvote",
    r"(?i)downvote",
    r"(?i)reddit",
    r"(?i)stack\s*overflow",
    r"(?i)#\w+",  # hashtags
    r"(?i)\bimo\b",
    r"(?i)\bimho\b",
    r"(?i)\btbh\b",
    r"(?i)\blol\b",
    r"(?i)\blmao\b",
]
POISON_COMPILED = [re.compile(p) for p in POISON_PATTERNS]

def is_poisoned(text: str) -> bool:
    """Reject text with conversational/social artifacts."""
    for pattern in POISON_COMPILED:
        if pattern.search(text):
            return True
    return False

# --- ENTITY MASKING: Prevent fact memorization, keep reasoning ---
def mask_entities(text):
    """Replace proper nouns/entities with placeholders. Keep numbers for math."""
    # Multi-word capitalized phrases (names, places, organizations)
    matches = list(set(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', text)))
    for i, match in enumerate(matches[:10]):
        text = text.replace(match, f'[E{i}]')
    # Single capitalized words mid-sentence (likely proper nouns)
    text = re.sub(r'(?<=[a-z]\. )([A-Z][a-z]{2,})\b', r'[E]', text)
    text = re.sub(r'(?<=\s)([A-Z][a-z]{2,})(?=\s+said|\s+wrote|\s+discovered)', r'[E]', text)
    return text

def is_repetitive(text: str, threshold: float = 0.3) -> bool:
    """Reject text with excessive repetition (causes infinite loops)."""
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if len(lines) < 5:
        return False
    return len(set(lines)) / len(lines) < threshold

# --- FORMATTERS: Structured reasoning format ---

def format_deepmath(item):
    """DeepMath-103K: Use R1 reasoning traces."""
    problem = item.get("problem", item.get("question", ""))
    # Try to get R1 solution (they have multiple)
    solution = item.get("r1_solution_1", item.get("solution", item.get("answer", "")))
    if not problem or not solution:
        return None
    text = f"<|begin_of_thought|>\nAnalyze the mathematical structure. Identify constraints and relationships.\n<|end_of_thought|>\n\nProblem:\n{problem}\n\nReasoning:\n{solution}"
    return mask_entities(text)

def format_rstar(item):
    """rStar-Coder: Algorithmic reasoning with verification."""
    problem = item.get("query", item.get("problem", item.get("question", "")))
    response = item.get("response", item.get("solution", ""))
    # Only use verified solutions
    if item.get("is_passed") == False:
        return None
    if not problem or not response:
        return None
    text = f"<|begin_of_thought|>\nParse constraints. Analyze time/space complexity. Design algorithm.\n<|end_of_thought|>\n\nSpecification:\n{problem}\n\nSolution:\n{response}"
    return mask_entities(text)

def format_stratos(item):
    """Bespoke-Stratos: Metacognitive reasoning - uses conversations format."""
    # Schema: {system: str, conversations: [{from: user/assistant, value: str}]}
    conversations = item.get("conversations", [])
    if len(conversations) < 2:
        return None
    # Extract user question (first turn) and assistant response
    problem = None
    response = None
    for turn in conversations:
        if turn.get("from") == "user":
            problem = turn.get("value", "")
        elif turn.get("from") == "assistant":
            response = turn.get("value", "")
    if not problem or not response:
        return None
    # The response often contains \boxed{} for final answer - this is metacognitive
    text = f"<|begin_of_thought|>\nAnalyze systematically. Explore multiple approaches. Verify each step.\n<|end_of_thought|>\n\nProblem:\n{problem}\n\n<|begin_of_solution|>\n{response}\n<|end_of_solution|>"
    return mask_entities(text)

def format_openthoughts(item):
    """OpenThoughts: General reasoning - uses conversations format."""
    # Schema: {system: str, conversations: [{from: user/assistant, value: str}]}
    conversations = item.get("conversations", [])
    if len(conversations) < 2:
        return None
    problem = None
    response = None
    for turn in conversations:
        if turn.get("from") == "user":
            problem = turn.get("value", "")
        elif turn.get("from") == "assistant":
            response = turn.get("value", "")
    if not problem or not response:
        return None
    text = f"<|begin_of_thought|>\nReason systematically. Explore approaches. Verify each step.\n<|end_of_thought|>\n\nProblem:\n{problem}\n\n<|begin_of_solution|>\n{response}\n<|end_of_solution|>"
    return mask_entities(text)

def format_opencode(item):
    """OpenCodeReasoning: Code with reasoning traces."""
    problem = item.get("question", item.get("problem", ""))
    reasoning = item.get("reasoning", item.get("thought", ""))
    code = item.get("solution", item.get("code", item.get("answer", "")))
    if not problem or not code:
        return None
    if reasoning:
        text = f"<|begin_of_thought|>\n{reasoning}\n<|end_of_thought|>\n\nSpecification:\n{problem}\n\nImplementation:\n{code}"
    else:
        text = f"<|begin_of_thought|>\nParse requirements. Design algorithm. Implement.\n<|end_of_thought|>\n\nSpecification:\n{problem}\n\nImplementation:\n{code}"
    return mask_entities(text)

def format_mathsmith(item):
    """MathSmith: First-principles axiomatic problems."""
    # Schema: Sampled_concept, Rationale, Problem, Output
    problem = item.get("Problem", item.get("problem", ""))
    solution = item.get("Output", item.get("output", ""))
    concepts = item.get("Sampled_concept", "")
    rationale = item.get("Rationale", "")
    if not problem or not solution:
        return None
    # Use rationale as the thinking process if available
    if rationale:
        text = f"<|begin_of_thought|>\n{rationale}\n<|end_of_thought|>\n\nProblem:\n{problem}\n\n<|begin_of_solution|>\n{solution}\n<|end_of_solution|>"
    elif concepts:
        text = f"<|begin_of_thought|>\nDeriving from axioms: {concepts}\n<|end_of_thought|>\n\nProblem:\n{problem}\n\n<|begin_of_solution|>\n{solution}\n<|end_of_solution|>"
    else:
        text = f"<|begin_of_thought|>\nApply first principles. Derive from definitions.\n<|end_of_thought|>\n\nProblem:\n{problem}\n\n<|begin_of_solution|>\n{solution}\n<|end_of_solution|>"
    return mask_entities(text)

def format_numina(item):
    """NuminaMath-CoT: Competition math with chain-of-thought."""
    problem = item.get("problem", item.get("question", ""))
    solution = item.get("solution", item.get("answer", ""))
    source = item.get("source", "")
    if not problem or not solution:
        return None
    # Source often indicates difficulty (amc, aime, olympiad)
    if source:
        text = f"<|begin_of_thought|>\nCompetition problem ({source}). Analyze structure. Apply relevant theorems.\n<|end_of_thought|>\n\nProblem:\n{problem}\n\n<|begin_of_solution|>\n{solution}\n<|end_of_solution|>"
    else:
        text = f"<|begin_of_thought|>\nAnalyze problem structure. Identify applicable theorems.\n<|end_of_thought|>\n\nProblem:\n{problem}\n\n<|begin_of_solution|>\n{solution}\n<|end_of_solution|>"
    return mask_entities(text)

def format_metamath(item):
    """MetaMathQA: Augmented math reasoning."""
    query = item.get("query", item.get("question", ""))
    response = item.get("response", item.get("answer", ""))
    qtype = item.get("type", "")
    if not query or not response:
        return None
    text = f"<|begin_of_thought|>\nSolve step by step. Verify arithmetic.\n<|end_of_thought|>\n\nProblem:\n{query}\n\n<|begin_of_solution|>\n{response}\n<|end_of_solution|>"
    return mask_entities(text)

def format_synth(item):
    """PleIAs/SYNTH: Textbook-style knowledge (filtered)."""
    # Use synthetic_reasoning if available (structured thinking)
    reasoning = item.get("synthetic_reasoning", "")
    content = item.get("synthetic_answer", item.get("text", ""))
    exercise = item.get("exercise", "")
    
    # Filter out chat/dialogue exercises
    if exercise and exercise.lower() in ["chat", "dialogue", "memorization", "conversation"]:
        return None
    if not content:
        return None
    if reasoning:
        text = f"<|begin_of_thought|>\n{reasoning}\n<|end_of_thought|>\n\nExplanation:\n{content}"
    else:
        text = f"Explanation:\n{content}"
    return mask_entities(text)

# --- PROCESSING ---

def process_item(item_and_name):
    """Process a single item with appropriate formatter."""
    item, formatter_name = item_and_name
    try:
        if formatter_name == "deepmath":
            text = format_deepmath(item)
        elif formatter_name == "rstar":
            text = format_rstar(item)
        elif formatter_name == "stratos":
            text = format_stratos(item)
        elif formatter_name == "openthoughts":
            text = format_openthoughts(item)
        elif formatter_name == "opencode":
            text = format_opencode(item)
        elif formatter_name == "mathsmith":
            text = format_mathsmith(item)
        elif formatter_name == "synth":
            text = format_synth(item)
        elif formatter_name == "numina":
            text = format_numina(item)
        elif formatter_name == "metamath":
            text = format_metamath(item)
        else:
            return None
        
        if text is None:
            return None
        if len(text) < 200:
            return None
        if is_poisoned(text):
            return None
        if is_repetitive(text):
            return None
        
        return text[:MAX_TEXT_LENGTH]
    except Exception:
        return None

def load_and_sample_dataset(name, path, split, sample_size, config=None, data_dir=None):
    """Load dataset and take a sample."""
    print(f"📥 Downloading {name}...")
    try:
        kwargs = {"split": split, "token": HF_TOKEN}
        if config:
            kwargs["name"] = config
        if data_dir:
            kwargs["data_dir"] = data_dir
        ds = load_dataset(path, **kwargs)
        ds = ds.shuffle(seed=42)
        if len(ds) > sample_size:
            ds = ds.select(range(sample_size))
        print(f"   ✓ {name}: {len(ds):,} samples")
        return ds
    except Exception as e:
        print(f"   ⚠ {name} failed: {e}")
        return None

def main():
    print("=" * 70)
    print("🔧 PURE LOGIC DISTILLATION ENGINE: Data Forge V3 (MAXED)")
    print("   Based on: Textbook Is All You Need + Evol-Instruct")
    print("   Target: 3-5M samples for B200 production run")
    print("=" * 70)
    
    datasets_to_process = []
    
    # ===== HIGH PRIORITY: Logic Anchors =====
    
    # 1. DeepMath-103K - Math Logic Anchor (take ALL of it)
    ds = load_and_sample_dataset("deepmath", "zwhe99/DeepMath-103K", "train", 150000)
    if ds: datasets_to_process.append(("deepmath", ds))
    
    # 2. rStar-Coder - Code Logic Anchor (take ALL ~580k)
    ds = load_and_sample_dataset("rstar", "microsoft/rStar-Coder", "train", 600000, data_dir="synthetic_sft")
    if ds: datasets_to_process.append(("rstar", ds))
    
    # 3. OpenCodeReasoning - Dense Code Reasoning (split_0 + split_1)
    ds = load_and_sample_dataset("opencode", "nvidia/OpenCodeReasoning", "train", 750000, config="split_0")
    if ds: datasets_to_process.append(("opencode", ds))
    ds = load_and_sample_dataset("opencode2", "nvidia/OpenCodeReasoning", "train", 750000, config="split_1")
    if ds: datasets_to_process.append(("opencode", ds))  # Same formatter
    
    # ===== REASONING STYLE: Upsample these heavily =====
    
    # 4. Bespoke-Stratos-17k - Metacognitive reasoning (will 15x upsample = ~250k)
    ds = load_and_sample_dataset("stratos", "bespokelabs/Bespoke-Stratos-17k", "train", 20000)
    if ds: datasets_to_process.append(("stratos", ds))
    
    # 5. OpenThoughts-114k - General Reasoning (take all, will 3x = ~340k)
    ds = load_and_sample_dataset("openthoughts", "open-thoughts/OpenThoughts-114k", "train", 120000)
    if ds: datasets_to_process.append(("openthoughts", ds))
    
    # ===== KNOWLEDGE BASE: First Principles =====
    
    # 6. MathSmith-Hard - Axiomatic math (Earth Bias fix)
    ds = load_and_sample_dataset("mathsmith", "Jasaxion/MathSmith-Hard-Problems", "train", 100000)
    if ds: datasets_to_process.append(("mathsmith", ds))
    
    # 7. SKIP PleIAs/SYNTH - Too large (41B tokens), causes disk issues
    # We have enough textbook-style data from other sources
    
    # ===== BONUS: More math reasoning =====
    
    # 8. NuminaMath-CoT - Competition math with CoT (HIGH QUALITY)
    ds = load_and_sample_dataset("numina", "AI-MO/NuminaMath-CoT", "train", 500000)
    if ds: datasets_to_process.append(("numina", ds))
    
    # 9. MetaMathQA - Augmented math reasoning
    ds = load_and_sample_dataset("metamath", "meta-math/MetaMathQA", "train", 400000)
    if ds: datasets_to_process.append(("metamath", ds))
    
    if not datasets_to_process:
        print("❌ No datasets loaded!")
        return
    
    print("\n" + "=" * 70)
    print("⚒️  Processing with entity masking (preserving numbers for math)...")
    print("=" * 70)
    
    all_texts = []
    
    for name, ds in datasets_to_process:
        print(f"Processing {name}...")
        items_with_name = [(dict(item), name) for item in ds]
        
        with multiprocessing.Pool(NUM_WORKERS) as pool:
            results = list(tqdm(
                pool.imap(process_item, items_with_name, chunksize=100),
                total=len(items_with_name),
                desc=f"  {name}"
            ))
        
        valid = [r for r in results if r is not None]
        
        # Strategic upsampling based on dataset importance
        if name == "stratos":
            valid = valid * 5  # 5x - important for reasoning but 15x was too much
            print(f"  ↑ Upsampled stratos 5x: {len(valid):,} samples")
        elif name == "openthoughts":
            valid = valid * 3  # 3x - good general reasoning
            print(f"  ↑ Upsampled openthoughts 3x: {len(valid):,} samples")
        elif name == "deepmath":
            valid = valid * 5  # 5x - math anchor is critical
            print(f"  ↑ Upsampled deepmath 5x: {len(valid):,} samples")
        
        all_texts.extend(valid)
        print(f"  ✓ {len(valid):,} valid samples from {name}")
    
    print(f"\n📊 Total collected: {len(all_texts):,} samples")
    random.shuffle(all_texts)
    
    if len(all_texts) > TARGET_SAMPLES:
        all_texts = all_texts[:TARGET_SAMPLES]
    
    print(f"💾 Writing {len(all_texts):,} samples to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for text in tqdm(all_texts, desc="Writing"):
            json.dump({"text": text}, f)
            f.write("\n")
    
    print("\n" + "=" * 70)
    print(f"✅ FORGE COMPLETE: {len(all_texts):,} high-fidelity logic samples")
    print("   Entity masking: ON (prevents fact memorization)")
    print("   Number masking: OFF (preserves arithmetic learning)")
    print("=" * 70)

if __name__ == "__main__":
    main()