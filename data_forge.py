import spacy
import json
import os
import multiprocessing
import random
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm

# --- CONFIGURATION ---
OUTPUT_FILE = "adam_skeleton_data.jsonl"
# 5 Million Samples: ~2.5M General Knowledge + ~2.5M Polyglot Logic
TARGET_SAMPLES = 5000000
BATCH_SIZE = 100
NUM_PROCESSES = max(1, multiprocessing.cpu_count() - 2)

# Logic Languages: A mix ensures Adam learns different "styles" of reasoning.
# Python (Algorithmic), Rust (Strict/Safety), C++ (Low Level), JS (Event Driven)
CODE_LANGUAGES = ["python", "rust", "c++", "javascript", "go"]


def process_batch(batch_data):
    """
    Handles a mixed batch of text and code.
    - Text: Masked to enforce Parametric Ignorance.
    - Code: Kept raw to ensure syntax acquisition (Logic).
    """
    try:
        # Load spacy only for text processing, keep it lightweight
        nlp = spacy.load("en_core_web_sm", disable=["parser", "tagger", "lemmatizer"])
    except OSError:
        # Fallback if spacy isn't installed/downloaded, just return raw content
        return [item["content"] for item in batch_data]

    processed_results = []

    text_contents = []
    # Map original indices to preserve order if needed, or just append distinct types
    # Here we simply process text and append code directly

    for item in batch_data:
        if item["type"] == "text":
            text_contents.append(item["content"])
        else:
            # Code is appended directly (Adam needs to learn raw logic/syntax)
            processed_results.append(item["content"])

    # Batch mask the text
    if text_contents:
        # Batch size for spacy pipe
        for doc in nlp.pipe(text_contents, batch_size=50):
            new_text = doc.text
            # Process entities in reverse to handle offset changes cleanly
            for ent in reversed(doc.ents):
                label = f"<{ent.label_}>"
                new_text = new_text[: ent.start_char] + label + new_text[ent.end_char :]
            processed_results.append(new_text)

    # Shuffle results so a batch isn't just block of code then block of text
    random.shuffle(processed_results)
    return processed_results


def main():
    print(f"🐈 Catbelly Studio: Igniting Hybrid Forge (Text + Multi-Lang Code)...")

    try:
        # 1. Logic Source: Wikipedia (The "Why" - General Reasoning)
        print("   - Loading Wikipedia...")
        wiki_ds = load_dataset(
            "wikimedia/wikipedia", "20231101.en", split="train", streaming=True
        )

        # 2. Logic Source: The Stack Smol (The "How" - Formal Logic)
        # We interleave multiple languages to create a robust logic dataset
        print(f"   - Loading Code ({', '.join(CODE_LANGUAGES)})...")
        code_datasets = []
        for lang in CODE_LANGUAGES:
            # Load specific language subset
            ds = load_dataset(
                "bigcode/the-stack-smol",
                data_dir=f"data/{lang}",
                split="train",
                streaming=True,
            )
            code_datasets.append(ds)

        # We will cycle through these iterators
        code_iters = [iter(ds) for ds in code_datasets]

    except Exception as e:
        print(f"❌ Network/Dataset Error: {e}")
        return

    buffer = []
    processed_count = 0
    mode = "a" if os.path.exists(OUTPUT_FILE) else "w"

    wiki_iter = iter(wiki_ds)

    with open(OUTPUT_FILE, mode, encoding="utf-8") as f:
        with multiprocessing.Pool(processes=NUM_PROCESSES) as pool:
            pbar = tqdm(total=TARGET_SAMPLES, desc="Forging Hybrid Mind")

            while processed_count < TARGET_SAMPLES:
                try:
                    for _ in range(BATCH_SIZE * NUM_PROCESSES):
                        # 50/50 Split between Text and Code
                        if random.random() > 0.5:
                            # --- Process Wikipedia ---
                            try:
                                item = next(wiki_iter)
                                if len(item["text"]) > 1000:
                                    buffer.append(
                                        {"type": "text", "content": item["text"]}
                                    )
                            except StopIteration:
                                continue  # If wiki runs out (unlikely with streaming), skip
                        else:
                            # --- Process Code (Random Language) ---
                            # Pick a random language iterator
                            lang_idx = random.randint(0, len(code_iters) - 1)
                            try:
                                item = next(code_iters[lang_idx])
                                if len(item["content"]) > 500:
                                    buffer.append(
                                        {"type": "code", "content": item["content"]}
                                    )
                            except StopIteration:
                                # If one language runs out, just try another next time
                                continue

                except StopIteration:
                    break

                if not buffer:
                    # If buffer is empty after trying to fill, we might be out of data
                    # But with streaming, this is rare. Break to avoid infinite loop if network fails.
                    break

                # Chunk and process
                chunk_size = max(1, len(buffer) // NUM_PROCESSES)
                chunks = [
                    buffer[i : i + chunk_size]
                    for i in range(0, len(buffer), chunk_size)
                ]

                results = pool.map(process_batch, chunks)

                for batch_result in results:
                    for text in batch_result:
                        json.dump({"text": text}, f)
                        f.write("\n")
                        processed_count += 1
                        pbar.update(1)
                        if processed_count >= TARGET_SAMPLES:
                            break
                    if processed_count >= TARGET_SAMPLES:
                        break
                buffer = []

    print(f"✅ Data saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
