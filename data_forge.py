import spacy
import json
import os
import multiprocessing
from datasets import load_dataset
from tqdm import tqdm

# --- CONFIGURATION ---
OUTPUT_FILE = "adam_skeleton_data.jsonl"
# UPDATED: 5 Million samples.
# At ~2 samples per second training speed, this provides ~700 hours (1 month) per Epoch.
# For a 4-month project, Adam will read this dataset about 3-4 times.
TARGET_SAMPLES = 5000000
BATCH_SIZE = 100
NUM_PROCESSES = max(1, multiprocessing.cpu_count() - 2)


def mask_entities(text_batch):
    try:
        # Disable heavy pipeline components for speed
        nlp = spacy.load("en_core_web_sm", disable=["parser", "tagger", "lemmatizer"])
    except OSError:
        return text_batch

    masked_texts = []
    for doc in nlp.pipe(text_batch, batch_size=50):
        new_text = doc.text
        # We process entities in reverse to not mess up indices
        for ent in reversed(doc.ents):
            label = f"<{ent.label_}>"
            new_text = new_text[: ent.start_char] + label + new_text[ent.end_char :]
        masked_texts.append(new_text)
    return masked_texts


def main():
    print(
        f"🐈 Catbelly Studio: Igniting Data Forge for Adam (Target: {TARGET_SAMPLES})..."
    )
    try:
        # Using the Parquet-based Wikipedia dataset (Stable)
        dataset = load_dataset(
            "wikimedia/wikipedia", "20231101.en", split="train", streaming=True
        )
    except Exception as e:
        print(f"❌ Network/Dataset Error: {e}")
        return

    buffer = []
    processed_count = 0

    # Open in 'append' mode ('a') so you can stop/start this script if needed over days
    mode = "a" if os.path.exists(OUTPUT_FILE) else "w"

    with open(OUTPUT_FILE, mode, encoding="utf-8") as f:
        with multiprocessing.Pool(processes=NUM_PROCESSES) as pool:
            iterator = iter(dataset)
            pbar = tqdm(total=TARGET_SAMPLES, desc="Forging Adam's Mind")

            # If resuming, update pbar
            if mode == "a":
                # Quick line count estimate (optional, can be slow on massive files)
                pass

            while processed_count < TARGET_SAMPLES:
                try:
                    for _ in range(BATCH_SIZE * NUM_PROCESSES):
                        item = next(iterator)
                        # Filter for high-quality, medium-length articles
                        if len(item["text"]) > 1000:
                            buffer.append(item["text"])
                except StopIteration:
                    break

                if not buffer:
                    break

                chunk_size = max(1, len(buffer) // NUM_PROCESSES)
                chunks = [
                    buffer[i : i + chunk_size]
                    for i in range(0, len(buffer), chunk_size)
                ]

                results = pool.map(mask_entities, chunks)

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
