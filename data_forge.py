import spacy
import json
import os
import multiprocessing
import random
from datasets import load_dataset
from tqdm import tqdm

# --- CONFIGURATION ---
OUTPUT_FILE = "adam_skeleton_data.jsonl"
TARGET_SAMPLES = 5000000  # 5 Million samples for the 4-month project duration
BATCH_SIZE = 100
NUM_PROCESSES = max(1, multiprocessing.cpu_count() - 2)
MAX_TEXT_LENGTH = 10000

# Logic Languages for Polyglot Reasoning
CODE_LANGUAGES = ["python", "rust", "c++", "javascript", "go"]

# --- THEORY: DRI (DATA REASONING INTENSITY) ---
# Derived from "Transition to Reasoning-Centric Data Engineering"
# Words that act as proxies for logical causality and structural transitions.
LOGIC_MARKERS = {
    "because",
    "therefore",
    "consequently",
    "implies",
    "thus",
    "however",
    "although",
    "unless",
    "furthermore",
    "hence",
    "conceptually",
    "theoretically",
    "derive",
    "calculate",
    "if",
    "then",
    "result",
    "evidence",
    "hypothesis",
    "conclusion",
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


def process_batch(batch_data):
    """Handles masking and polyglot interleaving."""
    try:
        nlp = spacy.load("en_core_web_sm", disable=["parser", "tagger", "lemmatizer"])
    except OSError:
        return [item["content"] for item in batch_data]

    processed_results = []
    text_contents = []

    for item in batch_data:
        if item["type"] == "text":
            text_contents.append(item["content"])
        else:
            processed_results.append(item["content"])

    if text_contents:
        for doc in nlp.pipe(text_contents, batch_size=50):
            new_text = doc.text
            # Hard Masking: Replacing entities to enforce Parametric Ignorance
            for ent in reversed(doc.ents):
                label = f"<{ent.label_}>"
                new_text = new_text[: ent.start_char] + label + new_text[ent.end_char :]
            processed_results.append(new_text)

    random.shuffle(processed_results)
    return processed_results


def main():
    print(
        f"🐈 Catbelly Studio: Igniting DRI-Optimized Forge (Target: {TARGET_SAMPLES})..."
    )
    try:
        wiki_ds = load_dataset(
            "wikimedia/wikipedia", "20231101.en", split="train", streaming=True
        )
        code_datasets = [
            load_dataset(
                "bigcode/the-stack-smol",
                data_dir=f"data/{l}",
                split="train",
                streaming=True,
            )
            for l in CODE_LANGUAGES
        ]
        code_iters = [iter(ds) for ds in code_datasets]
    except Exception as e:
        print(f"❌ Dataset Error: {e}")
        return

    buffer = []
    processed_count = 0
    mode = "a" if os.path.exists(OUTPUT_FILE) else "w"
    wiki_iter = iter(wiki_ds)

    with open(OUTPUT_FILE, mode, encoding="utf-8") as f:
        with multiprocessing.Pool(processes=NUM_PROCESSES) as pool:
            pbar = tqdm(total=TARGET_SAMPLES, desc="Forging Adam's Mind")

            while processed_count < TARGET_SAMPLES:
                try:
                    for _ in range(BATCH_SIZE * NUM_PROCESSES):
                        if random.random() > 0.5:
                            # --- Wikipedia: Reasoning Filtered ---
                            try:
                                item = next(wiki_iter)
                                text = item["text"]
                                # Filter for Reasoning Density and length
                                if len(text) > 1000 and is_reasoning_dense(
                                    text, threshold=0.4
                                ):
                                    buffer.append(
                                        {
                                            "type": "text",
                                            "content": text[:MAX_TEXT_LENGTH],
                                        }
                                    )
                            except StopIteration:
                                continue
                        else:
                            # --- Polyglot Code ---
                            lang_idx = random.randint(0, len(code_iters) - 1)
                            try:
                                item = next(code_iters[lang_idx])
                                if len(item["content"]) > 500:
                                    buffer.append(
                                        {"type": "code", "content": item["content"]}
                                    )
                            except StopIteration:
                                continue
                except StopIteration:
                    break

                if not buffer:
                    break
                chunk_size = max(1, len(buffer) // NUM_PROCESSES)
                chunks = [
                    buffer[i : i + chunk_size]
                    for i in range(0, len(buffer), chunk_size)
                ]
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
