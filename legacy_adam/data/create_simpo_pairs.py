#!/usr/bin/env python3
"""Convert persona-augmented data to SimPO preference pairs.

Takes verified persona-augmented data and creates preference pairs
suitable for SimPO training.

Usage:
    python create_simpo_pairs.py \
        --input hope/adam_training_data/adam_persona_verified.jsonl \
        --output hope/adam_training_data/adam_simpo_ready.jsonl
"""

import argparse
import json
import random
from pathlib import Path
from typing import Optional


def create_preference_pair(example: dict) -> dict:
    """Convert a single example to SimPO preference pair format.

    The train_adam_simpo.py script expects: instruction, input, preferred, rejected.
    We preserve this format while adding persona metadata.

    Args:
        example: Training example with instruction, input, preferred, rejected

    Returns:
        SimPO-compatible dict preserving original field names
    """
    return {
        "instruction": example.get("instruction", ""),
        "input": example.get("input", ""),
        "preferred": example.get("preferred", ""),
        "rejected": example.get("rejected", ""),
        # Preserve metadata for analysis
        "category": example.get("category", ""),
        "persona": example.get("persona", "original"),
        "original_id": example.get("original_id", ""),
        "metadata": example.get("metadata", {}),
    }


def validate_pair(pair: dict) -> bool:
    """Validate that a preference pair is well-formed."""
    # All fields must be strings
    instruction = pair.get("instruction", "")
    preferred = pair.get("preferred", "")
    rejected = pair.get("rejected", "")

    if not isinstance(instruction, str) or not isinstance(preferred, str) or not isinstance(rejected, str):
        return False

    # Must have non-empty instruction and preferred/rejected
    if not instruction.strip():
        return False
    if not preferred.strip():
        return False
    if not rejected.strip():
        return False

    # Preferred and rejected must be different
    if preferred.strip() == rejected.strip():
        return False

    return True


def run_conversion(
    input_path: str,
    output_path: str,
    shuffle: bool = True,
    seed: int = 42,
    max_examples: Optional[int] = None
):
    """Convert verified augmented data to SimPO pairs.

    Args:
        input_path: Path to verified JSONL input
        output_path: Path for SimPO-format output
        shuffle: Whether to shuffle output
        seed: Random seed for shuffling
        max_examples: Optional limit on output examples
    """
    print(f"Loading data from {input_path}")
    examples = []
    with open(input_path) as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))

    print(f"Loaded {len(examples)} examples")

    # Convert to SimPO format
    pairs = []
    skipped = 0

    for ex in examples:
        pair = create_preference_pair(ex)
        if validate_pair(pair):
            pairs.append(pair)
        else:
            skipped += 1

    print(f"Created {len(pairs)} valid preference pairs")
    if skipped > 0:
        print(f"Skipped {skipped} invalid examples")

    # Shuffle if requested
    if shuffle:
        random.seed(seed)
        random.shuffle(pairs)

    # Apply max limit
    if max_examples and len(pairs) > max_examples:
        pairs = pairs[:max_examples]
        print(f"Limited to {max_examples} examples")

    # Save output
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")

    print(f"Saved {len(pairs)} preference pairs to {output_path}")

    # Print statistics
    persona_counts = {}
    category_counts = {}
    for p in pairs:
        persona = p.get("persona", "unknown")
        category = p.get("category", "unknown")
        persona_counts[persona] = persona_counts.get(persona, 0) + 1
        category_counts[category] = category_counts.get(category, 0) + 1

    print("\nPairs per persona:")
    for persona, count in sorted(persona_counts.items()):
        print(f"  {persona}: {count}")

    print("\nPairs per category:")
    for category, count in sorted(category_counts.items()):
        print(f"  {category}: {count}")

    return pairs


def main():
    parser = argparse.ArgumentParser(description="Convert to SimPO Preference Pairs")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to verified augmented JSONL"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path for SimPO-format output JSONL"
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Don't shuffle the output"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling"
    )
    parser.add_argument(
        "--max",
        type=int,
        default=None,
        help="Maximum number of examples to output"
    )
    args = parser.parse_args()

    run_conversion(
        input_path=args.input,
        output_path=args.output,
        shuffle=not args.no_shuffle,
        seed=args.seed,
        max_examples=args.max
    )


if __name__ == "__main__":
    main()
