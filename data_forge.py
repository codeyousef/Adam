"""
Data Forge V2 - Pure Logic Engine (FAST VERSION)
Downloads datasets first, then processes with multiprocessing.
"""
import json
import os
import random
import multiprocessing
from functools import partial
from datasets import load_dataset
from tqdm import tqdm

# --- CONFIGURATION ---
OUTPUT_FILE = "adam_skeleton_data.jsonl"
TARGET_SAMPLES = 2_000_000
MAX_TEXT_LENGTH = 4096
BATCH_SIZE = 1000
NUM_WORKERS = max(1, multiprocessing.cpu_count() - 2)
HF_TOKEN = "hf"

# --- POISON FILTER ---
POISON_PATTERNS = {
    "i think", "i believe", "in my opinion", "user:", "reddit", "stack overflow",
    "upvote", "downvote", "comment", "reply", "thanks!", "thank you", "please help",
    "anyone know", "can someone", "i was wondering", "hey guys", "hi everyone",
    "#", "hashtag", "follow me", "subscribe", "like and share", "imo", "imho",
    "lol", "lmao", "haha", "xd", "btw", "tbh", "ngl", "edit:", "update:",
    "tldr", "tl;dr", "source:", "via @", "retweet", "shared", "posted by",
    "imagine you", "let's say", "picture this", "once upon", "one day",
}

def is_poisoned(text: str) -> bool:
    text_lower = text.lower()
    return any(p in text_lower for p in POISON_PATTERNS)

def is_repetitive(text: str, threshold: float = 0.3) -> bool:
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if len(lines) < 5:
        return False
    return len(set(lines)) / len(lines) < threshold

def format_physics(item):
    q = item.get("message_1", item.get("question", ""))
    a = item.get("message_2", item.get("answer", ""))
    return f"<THOUGHT>Analyzing physics problem.</THOUGHT>\nProblem: {q}\n\nDerivation:\n{a}"

def format_math(item):
    q = item.get("instruction", item.get("question", item.get("problem", "")))
    a = item.get("output", item.get("answer", item.get("solution", "")))
    return f"<THOUGHT>Mathematical reasoning step by step.</THOUGHT>\nStatement: {q}\n\nSolution:\n{a}"

def format_code(item):
    problem = item.get("question", item.get("problem_statement", item.get("description", "")))
    solution = item.get("solutions", item.get("solution", item.get("code", "")))
    if isinstance(solution, list):
        solution = solution[0] if solution else ""
    return f"<THOUGHT>Parsing constraints carefully.</THOUGHT>\nSpecification:\n{problem}\n\nImplementation:\n{solution}"

def format_orca(item):
    return f"<THOUGHT>Word problem: extracting quantities.</THOUGHT>\nQuestion: {item.get('question', '')}\n\nReasoning:\n{item.get('answer', '')}"

def process_item(item_and_name):
    """Process a single item - used by multiprocessing."""
    item, formatter_name = item_and_name
    try:
        if formatter_name == "physics":
            text = format_physics(item)
        elif formatter_name in ("math", "math_hard", "metamath"):
            text = format_math(item)
        elif formatter_name in ("code", "apps"):
            text = format_code(item)
        elif formatter_name == "orca":
            text = format_orca(item)
        else:
            return None
        
        if len(text) < 200:
            return None
        if is_poisoned(text):
            return None
        if is_repetitive(text):
            return None
        
        return text[:MAX_TEXT_LENGTH]
    except:
        return None

def load_and_sample_dataset(name, path, split, sample_size, config=None):
    """Load dataset fully (no streaming) and take a sample."""
    print(f"📥 Downloading {name}...")
    try:
        if config:
            ds = load_dataset(path, config, split=split, token=HF_TOKEN)
        else:
            ds = load_dataset(path, split=split, token=HF_TOKEN)
        
        ds = ds.shuffle(seed=42)
        if len(ds) > sample_size:
            ds = ds.select(range(sample_size))
        print(f"   ✓ {name}: {len(ds):,} samples")
        return ds
    except Exception as e:
        print(f"   ⚠ {name} failed: {e}")
        return None

def main():
    print("=" * 60)
    print("🔧 PURE LOGIC ENGINE: Data Forge V2 (FAST)")
    print("   Downloads datasets first, then batch processes")
    print("=" * 60)
    
    # Calculate samples per source (weighted)
    weights = {
        "physics": 0.15,
        "math": 0.20, 
        "math_hard": 0.10,
        "metamath": 0.20,
        "orca": 0.20,
        "code": 0.08,
        "apps": 0.07,
    }
    
    datasets_to_process = []
    
    # Physics
    ds = load_and_sample_dataset("physics", "camel-ai/physics", "train", 
                                  int(TARGET_SAMPLES * weights["physics"] * 1.5))
    if ds: datasets_to_process.append(("physics", ds))
    
    # MathInstruct
    ds = load_and_sample_dataset("math", "TIGER-Lab/MathInstruct", "train",
                                  int(TARGET_SAMPLES * weights["math"] * 1.5))
    if ds: datasets_to_process.append(("math", ds))
    
    # MATH (competition)
    ds = load_and_sample_dataset("math_hard", "lighteval/MATH", "train",
                                  int(TARGET_SAMPLES * weights["math_hard"] * 1.5), config="all")
    if ds: datasets_to_process.append(("math_hard", ds))
    
    # MetaMathQA  
    ds = load_and_sample_dataset("metamath", "meta-math/MetaMathQA", "train",
                                  int(TARGET_SAMPLES * weights["metamath"] * 1.5))
    if ds: datasets_to_process.append(("metamath", ds))
    
    # Orca Math
    ds = load_and_sample_dataset("orca", "microsoft/orca-math-word-problems-200k", "train",
                                  int(TARGET_SAMPLES * weights["orca"] * 1.5))
    if ds: datasets_to_process.append(("orca", ds))
    
    # Code Contests
    ds = load_and_sample_dataset("code", "deepmind/code_contests", "train",
                                  int(TARGET_SAMPLES * weights["code"] * 1.5))
    if ds: datasets_to_process.append(("code", ds))
    
    # APPS
    ds = load_and_sample_dataset("apps", "codeparrot/apps", "train",
                                  int(TARGET_SAMPLES * weights["apps"] * 1.5))
    if ds: datasets_to_process.append(("apps", ds))
    
    if not datasets_to_process:
        print("❌ No datasets loaded!")
        return
    
    print("\n" + "=" * 60)
    print("⚒️  Processing and filtering...")
    print("=" * 60)
    
    all_texts = []
    
    for name, ds in datasets_to_process:
        print(f"Processing {name}...")
        
        # Prepare items with their formatter name
        items_with_name = [(dict(item), name) for item in ds]
        
        with multiprocessing.Pool(NUM_WORKERS) as pool:
            results = list(tqdm(
                pool.imap(process_item, items_with_name, chunksize=100),
                total=len(items_with_name),
                desc=f"  {name}"
            ))
        
        valid = [r for r in results if r is not None]
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
    
    print("\n" + "=" * 60)
    print(f"✅ FORGE COMPLETE: {len(all_texts):,} samples")
    print("=" * 60)

if __name__ == "__main__":
    main()