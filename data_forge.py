import spacy
import json
import os
import multiprocessing
import random
from datasets import load_dataset
from tqdm import tqdm

# --- CONFIGURATION ---
OUTPUT_FILE = "adam_skeleton_data.jsonl"
TARGET_SAMPLES = 5000000 
BATCH_SIZE = 100
NUM_PROCESSES = max(1, multiprocessing.cpu_count() - 2)
MAX_TEXT_LENGTH = 4096 

# AUTHENTICATION (Required for The Stack)
HF_TOKEN = "hf"  # Using the token you provided

# Logic Languages
CODE_LANGUAGES = ["python", "rust", "c++", "javascript", "go"]

# --- THEORY: DRI (DATA REASONING INTENSITY) ---
LOGIC_MARKERS = {
    "because", "therefore", "consequently", "implies", "thus", "however",
    "although", "unless", "furthermore", "hence", "conceptually",
    "theoretically", "derive", "calculate", "if", "then", "result",
    "evidence", "hypothesis", "conclusion", "since", "leads to", "due to",
    "assuming", "given", "structure", "function", "relationship"
}

def calculate_dri_score(text):
    words = text.lower().split()
    if not words: return 0.0
    logic_count = sum(1 for w in words if w in LOGIC_MARKERS)
    return (logic_count / len(words))

def is_reasoning_dense(text, threshold=0.03): 
    return calculate_dri_score(text) >= threshold

def inject_search_trajectories(text, nlp_doc):
    ents = [e for e in nlp_doc.ents if e.label_ in ["ORG", "PERSON", "GPE", "EVENT", "WORK_OF_ART"]]
    if not ents: return text
    target_ent = random.choice(ents)
    tool_sequence = f"<THOUGHT>I need to verify {target_ent.label_}.</THOUGHT> <SEARCH>{target_ent.text}</SEARCH> <RESULT>{target_ent.text}</RESULT> {target_ent.text}"
    new_text = text.replace(target_ent.text, tool_sequence, 1)
    return new_text

def process_batch(batch_data):
    try:
        nlp = spacy.load("en_core_web_sm", disable=["parser", "tagger", "lemmatizer"])
    except OSError:
        return [item["content"] for item in batch_data]

    processed_results = []
    text_contents = []
    
    for item in batch_data:
        if item["type"] == "text": text_contents.append(item["content"])
        else: processed_results.append(item["content"])

    if text_contents:
        for doc in nlp.pipe(text_contents, batch_size=50):
            rng = random.random()
            if rng < 0.2:
                try: processed_results.append(inject_search_trajectories(doc.text, doc))
                except: processed_results.append(doc.text)
            elif rng < 0.5:
                new_text = doc.text
                for ent in reversed(doc.ents):
                    if ent.label_ in ["ORG", "PERSON", "GPE"]:
                        label = f"<{ent.label_}>"
                        new_text = new_text[:ent.start_char] + label + new_text[ent.end_char:]
                processed_results.append(new_text)
            else:
                processed_results.append(doc.text)

    random.shuffle(processed_results)
    return processed_results

def main():
    print(f"🐈 Catbelly Studio: Igniting DRI-Optimized Forge (Target: {TARGET_SAMPLES})...")
    
    try:
        # 1. Wikipedia 
        wiki_ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True, token=HF_TOKEN)
        
        # 2. Cosmopedia 
        cosmo_ds = load_dataset("HuggingFaceTB/cosmopedia", "stories", split="train", streaming=True, token=HF_TOKEN)
        
        # 3. Orca Math 
        orca_ds = load_dataset("microsoft/orca-math-word-problems-200k", split="train", streaming=True, token=HF_TOKEN)
        
        # 4. The Stack (REQUIRES TOKEN)
        code_datasets = []
        for l in CODE_LANGUAGES:
            try:
                ds = load_dataset("bigcode/the-stack-smol", data_dir=f"data/{l}", split="train", streaming=True, token=HF_TOKEN)
                code_datasets.append(iter(ds))
            except Exception as e:
                print(f"⚠️ Code load error ({l}): {e}")
        
        wiki_iter = iter(wiki_ds)
        cosmo_iter = iter(cosmo_ds)
        orca_iter = iter(orca_ds)
        
    except Exception as e:
        print(f"❌ Dataset Init Error: {e}")
        return

    buffer = []
    processed_count = 0
    mode = "a" if os.path.exists(OUTPUT_FILE) else "w"

    with open(OUTPUT_FILE, mode, encoding="utf-8") as f:
        with multiprocessing.Pool(processes=NUM_PROCESSES) as pool:
            pbar = tqdm(total=TARGET_SAMPLES, desc="Forging Data")

            while processed_count < TARGET_SAMPLES:
                while len(buffer) < (BATCH_SIZE * NUM_PROCESSES):
                    rng = random.random()
                    try:
                        if rng < 0.4: # Cosmopedia/Orca
                            if random.random() > 0.5:
                                item = next(cosmo_iter)
                                text = item["text"]
                            else:
                                item = next(orca_iter)
                                text = f"Question: {item['question']}\nReasoning:\n{item['answer']}"
                            buffer.append({"type": "text", "content": text[:MAX_TEXT_LENGTH]})
                                
                        elif rng < 0.7: # Wiki
                            item = next(wiki_iter)
                            text = item["text"]
                            if len(text) > 500 and is_reasoning_dense(text, threshold=0.03):
                                buffer.append({"type": "text", "content": text[:MAX_TEXT_LENGTH]})
                                
                        else: # Code
                            if code_datasets:
                                lang_idx = random.randint(0, len(code_datasets) - 1)
                                item = next(code_datasets[lang_idx])
                                if len(item.get("content", "")) > 200:
                                    buffer.append({"type": "code", "content": item["content"][:MAX_TEXT_LENGTH]})
                                
                    except (StopIteration, Exception):
                        continue

                chunk_size = max(1, len(buffer) // NUM_PROCESSES)
                chunks = [buffer[i : i + chunk_size] for i in range(0, len(buffer), chunk_size)]
                results = pool.map(process_batch, chunks)

                for batch_result in results:
                    for t in batch_result:
                        if processed_count >= TARGET_SAMPLES: break
                        json.dump({"text": t}, f)
                        f.write("\n")
                        processed_count += 1
                        pbar.update(1)
                buffer = []

    print(f"✅ Data saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()