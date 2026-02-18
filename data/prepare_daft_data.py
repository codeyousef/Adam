#!/usr/bin/env python3
"""Prepare DAFT training data from persona-augmented SimPO data.

Converts persona labels to domain_id for DAFT training.
"""

import argparse
import json
from pathlib import Path

# Map persona names to domain IDs
PERSONA_TO_DOMAIN = {
    "original": 0,
    "academic": 1,
    "casual": 2,
    "technical": 3,
    "legal": 4,
    "socratic": 5,
    "journalistic": 6,
    "bullet": 7,
    "narrative": 8,
    "terse": 9,
    "verbose": 10,
}

def prepare_daft_data(input_path: str, output_path: str):
    """Convert SimPO preference data to DAFT format with domain labels."""
    print(f"Loading data from {input_path}")

    examples = []
    domain_counts = {d: 0 for d in range(11)}

    with open(input_path) as f:
        for line in f:
            if line.strip():
                ex = json.loads(line)

                # Get domain ID from persona
                persona = ex.get("persona", "original")
                domain_id = PERSONA_TO_DOMAIN.get(persona, 0)

                # Add domain_id to example
                ex["domain_id"] = domain_id
                examples.append(ex)
                domain_counts[domain_id] += 1

    print(f"Loaded {len(examples)} examples")
    print("\nDomain distribution:")
    for d, count in sorted(domain_counts.items()):
        persona_name = [k for k, v in PERSONA_TO_DOMAIN.items() if v == d][0]
        print(f"  D{d} ({persona_name}): {count}")

    # Save output
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    print(f"\nSaved {len(examples)} examples to {output_path}")

    return examples


def main():
    parser = argparse.ArgumentParser(description="Prepare DAFT data from SimPO data")
    parser.add_argument("--input", type=str, required=True, help="Input SimPO JSONL")
    parser.add_argument("--output", type=str, required=True, help="Output DAFT JSONL")
    args = parser.parse_args()

    prepare_daft_data(args.input, args.output)


if __name__ == "__main__":
    main()
