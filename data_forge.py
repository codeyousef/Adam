import spacy
import json
import os
import multiprocessing
import random
from datasets import load_dataset
from tqdm import tqdm

# --- CONFIGURATION ---
OUTPUT_FILE = "adam_skeleton_data.jsonl"
TARGET_SAMPLES = 5000000  # 5 Million samples (Approx 1 Epoch of high-quality data)
BATCH_SIZE = 100
NUM_PROCESSES = max(1, multiprocessing.cpu_count() - 2)
MAX_TEXT_LENGTH = 10000

# Logic Languages for Polyglot Reasoning
CODE_LANGUAGES = ["python", "rust", "c++", "javascript", "go"]

# --- THEORY: DRI (DATA REASONING INTENSITY) ---
LOGIC_MARKERS = {
    "because", "therefore", "consequently", "implies", "thus", "however",
    "although", "unless", "furthermore", "hence", "conceptually",
    "theoretically", "derive", "calculate", "if", "then", "result",
    "evidence", "hypothesis", "conclusion",
}

def calculate_dri_score(text):
    """Calculates the ratio of logical connectives to total words."""
    words = text.lower().split()
    if not words:
        return 0.0
    logic_count = sum(1 for w in words if w in LOGIC_MARKERS)
    return (logic_count / len(words)) * 100

def is_reasoning_dense(text, threshold=0.4):
    """Rejects flat factual text; accepts text explaining 'why' or 'how'."""
    return calculate_dri_score(text) >= threshold

def inject_search_trajectories(text, nlp_doc):
    """
    Transforms factual sentences into 'Search-Action' trajectories.
    Example: 
    "Elon Musk owns SpaceX." -> 
    "Elon Musk owns <SEARCH>companies owned by Elon Musk</SEARCH> <RESULT>SpaceX</RESULT> SpaceX."
    """
    # Filter for interesting entities to search about
    ents = [e for e in nlp_doc.ents if e.label_ in ["ORG", "PERSON", "GPE", "DATE", "EVENT", "WORK_OF_ART"]]
    
    if not ents:
        return text

    # Pick ONE entity to "forget" and search for per chunk
    target_ent = random.choice(ents)
    
    # Heuristic: Use surrounding words to form a natural context query
    start = max(0, target_ent.start - 4)
    end = min(len(nlp_doc), target_ent.end + 4)
    context_left = nlp_doc[start:target_ent.start].text
    context_right = nlp_doc[target_ent.end:end].text
    
    clean_query = f"{context_left} {target_ent.label_} {context_right}".strip()
    clean_query = " ".join(clean_query.split())

    # Construct the Tool-Use Pattern
    tool_sequence = f" <SEARCH>{clean_query}</SEARCH> <RESULT>{target_ent.text}</RESULT> {target_ent.text}"
    
    new_text = text[:target_ent.start_char] + tool_sequence + text[target_ent.end_char:]
    return new_text

def process_batch(batch_data):
    """Handles masking, search injection, and polyglot interleaving."""
    try:
        nlp = spacy.load("en_core_web_sm", disable=["parser", "tagger", "lemmatizer"])
    except OSError:
        return [item["content"] for item in batch_data]

    processed_results = []
    text_contents = []
    
    # Separate Code (keep as is) from Text (needs processing)
    for item in batch_data:
        if item["type"] == "text":
            text_contents.append(item["content"])
        else:
            processed_results.append(item["content"]) # Code

    if text_contents:
        for doc in nlp.pipe(text_contents, batch_size=50):
            # DECISION: Search Training vs. Parametric Ignorance
            # 30% Chance: Train to use SEARCH tools (Action-Native)
            # 70% Chance: Train to ignore entities (Parametric Ignorance)
            
            if random.random() < 0.3:
                # --- STRATEGY A: SEARCH INJECTION (Active Tool Use) ---
                new_text = inject_search_trajectories(doc.text, doc)
                processed_results.append(new_text)
            else:
                # --- STRATEGY B: HARD MASKING (Parametric Ignorance) ---
                new_text = doc.text
                for ent in reversed(doc.ents):
                    label = f"<{ent.label_}>"
                    new_text = new_text[: ent.start_char] + label + new_text[ent.end_char :]
                processed_results.append(new_text)

    random.shuffle(processed_results)
    return processed_results


def main():
    print(f"🐈 Catbelly Studio: Igniting DRI-Optimized Forge (Target: {TARGET_SAMPLES})...")
    print("   Mode: 40% Synthetic Reasoning | 30% Wiki (Search) | 30% Code")
    
    try:
        # 1. Wikipedia (Factual Backbone)
        wiki_ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
        
        # 2. Cosmopedia (Synthetic Textbooks - High Logic)
        cosmo_ds = load_dataset("HuggingFaceTB/cosmopedia", "stories", split="train", streaming=True)
        
        # 3. Orca Math (Step-by-Step Reasoning Chains)
        orca_ds = load_dataset("microsoft/orca-math-word-problems-200k", split="train", streaming=True)
        
        # 4. The Stack (Code Structure)
        code_datasets = [
            load_dataset("bigcode/the-stack-smol", data_dir=f"data/{l}", split="train", streaming=True)
            for l in CODE_LANGUAGES
        ]
        
        # Create Iterators
        wiki_iter = iter(wiki_ds)
        cosmo_iter = iter(cosmo_ds)
        orca_iter = iter(orca_ds)
        code_iters = [iter(ds) for ds in code_datasets]
        
    except Exception as e:
        print(f"❌ Dataset Error: {e}")
        return

    buffer = []
    processed_count = 0
    mode = "a" if os.path.exists(OUTPUT_FILE) else "w"

    with open(OUTPUT_FILE, mode, encoding="utf-8") as f:
        with multiprocessing.Pool(processes=NUM_PROCESSES) as pool:
            pbar = tqdm(total=TARGET_SAMPLES, desc="Forging Adam's Mind")

            while processed_count < TARGET_SAMPLES:
                try:
                    for _ in range(BATCH_SIZE * NUM_PROCESSES):
                        rng = random.random()
                        
                        # --- 40% SYNTHETIC REASONING (Textbooks & Math) ---
                        if rng < 0.4:
                            try:
                                if random.random() > 0.5:
                                    item = next(cosmo_iter)
                                    text = item["text"]
                                else:
                                    item = next(orca_iter)
                                    text = f"Question: {item['question']}\nReasoning:\n{item['answer']}"
                                buffer.append({"type": "text", "content": text[:MAX_TEXT_LENGTH]})
                            except StopIteration:
                                continue # If one stream ends, just skip
                                
                        # --- 30% WIKIPEDIA (Search Targets) ---
                        elif rng < 0.7:
                            try:
                                item = next(wiki_iter)
                                text = item["text"]
                                if len(text) > 1000 and is_reasoning_dense(text, threshold=0.4):
                                    buffer.append({"type": "text", "content": text[:MAX_TEXT_LENGTH]})
                            except StopIteration:
                                wiki_iter = iter(wiki_ds)
                                continue
                                
                        # --- 30% CODE (Structural Logic) ---
                        else:
                            lang_idx = random.randint(0, len(code_iters) - 1)
                            try:
                                item = next(code_iters[lang_idx])
                                if len(item["content"]) > 500:
                                    buffer.append({"type": "code", "content": item["content"]})
                            except StopIteration:
                                code_iters[lang_idx] = iter(code_datasets[lang_idx])
                                continue
                                
                except StopIteration:
                    break

                if not buffer:
                    break
                    
                chunk_size = max(1, len(buffer) // NUM_PROCESSES)
                chunks = [buffer[i : i + chunk_size] for i in range(0, len(buffer), chunk_size)]
                results = pool.map(process_batch, chunks)

                for batch_result in results:
                    for t in batch_result:
                        json.dump({"text": t}, f)
                        f.write("\n")
                        processed_count += 1
                        pbar.update(1)
                        if processed_count >= TARGET_SAMPLES:
                            break
                    if processed_count >= TARGET_SAMPLES:
                        break
                buffer = []

    print(f"✅ Reasoning-Dense Data saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()