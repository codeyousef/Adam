#!/usr/bin/env python3
"""Automatic format-domain labeling for DAFT training.

Assigns domain labels (0-5) to training examples based on their format characteristics.
Used to prepare data for Domain Adversarial Fine-Tuning.
"""

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from collections import Counter
from typing import Optional


@dataclass
class DomainPattern:
    """Pattern for detecting a format domain."""
    domain_id: int
    name: str
    patterns: list[str]  # Regex patterns
    description: str
    priority: int = 0  # Higher = check first


# Domain definitions matching the 6 format families
DOMAIN_PATTERNS = [
    DomainPattern(
        domain_id=0,
        name="training_format",
        patterns=[
            r"Premise\s+1:",
            r"Premise\s+\d+:",
        ],
        description="Training format: 'Premise 1:... Is X a B?'",
        priority=10
    ),
    DomainPattern(
        domain_id=1,
        name="bulleted",
        patterns=[
            r"^[•\-\*]\s",
            r"HYPOTHESIS:",
            r"^\s*•",
        ],
        description="Bulleted premises: '• All A are B HYPOTHESIS: X is B'",
        priority=9
    ),
    DomainPattern(
        domain_id=2,
        name="formal",
        patterns=[
            r"P\d+:",
            r"[⊢⊨→]",
            r"\|[-=]",
            r"∀|∃",
        ],
        description="Formal notation: 'P1: A⊂B, P2: X∈A ⊢ X∈B?'",
        priority=8
    ),
    DomainPattern(
        domain_id=3,
        name="conditional",
        patterns=[
            r"Given\s+that",
            r"If\s+.+\s+and\s+.+,\s+then",
            r"Assuming\s+that",
        ],
        description="Conditional phrasing: 'Given that..., is X B?'",
        priority=7
    ),
    DomainPattern(
        domain_id=4,
        name="question_first",
        patterns=[
            r"^Is\s+\w+\s+(?:a\s+)?\w+\?",
            r"^(?:Is|Are|Does|Do)\s+.+\?\s*\n.*Given:",
        ],
        description="Question-first ordering: 'Is X a B? Given: ...'",
        priority=6
    ),
    DomainPattern(
        domain_id=5,
        name="minimal",
        patterns=[
            r"^[A-Z][a-z]+\s+[a-z]+\s+[A-Z][a-z]+\s+[A-Z][a-z]+\s+[a-z]+",
            # Minimal punctuation - few special characters
        ],
        description="Minimal punctuation: 'All A are B X is A is X a B'",
        priority=5
    ),
]


class DomainLabeler:
    """Assign format domain labels to training examples."""

    def __init__(self, default_domain: int = 0):
        """Initialize labeler.

        Args:
            default_domain: Domain to assign when no pattern matches
        """
        self.patterns = sorted(DOMAIN_PATTERNS, key=lambda x: -x.priority)
        self.default_domain = default_domain

    def _text_from_example(self, example: dict) -> str:
        """Extract text content from example for pattern matching."""
        instruction = example.get("instruction", "")
        input_text = example.get("input", "")
        return f"{instruction}\n{input_text}"

    def _count_punctuation(self, text: str) -> float:
        """Count punctuation density as a feature."""
        punct_chars = set(".,;:!?\"'()-[]{}•*")
        punct_count = sum(1 for c in text if c in punct_chars)
        return punct_count / max(len(text), 1)

    def label_example(self, example: dict) -> int:
        """Assign a domain label to an example.

        Args:
            example: Training example dict

        Returns:
            Domain ID (0-5)
        """
        # If already labeled, return existing label
        if "domain_id" in example:
            return example["domain_id"]

        text = self._text_from_example(example)

        # Check each pattern in priority order
        for domain in self.patterns:
            for pattern in domain.patterns:
                if re.search(pattern, text, re.MULTILINE | re.IGNORECASE):
                    return domain.domain_id

        # Special case: minimal punctuation detection
        punct_density = self._count_punctuation(text)
        if punct_density < 0.02:  # Very low punctuation
            return 5  # minimal domain

        return self.default_domain

    def label_dataset(self, examples: list[dict]) -> list[dict]:
        """Add domain labels to all examples.

        Args:
            examples: List of training examples

        Returns:
            Examples with domain_id added
        """
        labeled = []
        for ex in examples:
            domain_id = self.label_example(ex)
            labeled_ex = {**ex, "domain_id": domain_id}

            # Also add domain name to metadata
            domain_name = DOMAIN_PATTERNS[domain_id].name if domain_id < len(DOMAIN_PATTERNS) else "unknown"
            if "metadata" not in labeled_ex:
                labeled_ex["metadata"] = {}
            labeled_ex["metadata"]["domain"] = domain_name

            labeled.append(labeled_ex)

        return labeled

    def analyze_distribution(self, examples: list[dict]) -> dict:
        """Analyze domain distribution in dataset.

        Args:
            examples: List of examples (with or without labels)

        Returns:
            Statistics dictionary
        """
        domain_counts = Counter()
        category_domain_counts = {}

        for ex in examples:
            domain_id = ex.get("domain_id", self.label_example(ex))
            domain_counts[domain_id] += 1

            category = ex.get("category", "unknown")
            if category not in category_domain_counts:
                category_domain_counts[category] = Counter()
            category_domain_counts[category][domain_id] += 1

        return {
            "total": len(examples),
            "domain_counts": dict(domain_counts),
            "domain_percentages": {
                d: count / len(examples) * 100
                for d, count in domain_counts.items()
            },
            "category_domain_counts": {
                cat: dict(counts)
                for cat, counts in category_domain_counts.items()
            }
        }


def verify_balanced_domains(examples: list[dict], tolerance: float = 0.1) -> bool:
    """Check if domains are reasonably balanced.

    Args:
        examples: Labeled examples
        tolerance: Acceptable deviation from uniform distribution

    Returns:
        True if balanced within tolerance
    """
    labeler = DomainLabeler()
    stats = labeler.analyze_distribution(examples)

    expected = 100 / 6  # ~16.7% per domain
    for domain_id, pct in stats["domain_percentages"].items():
        if abs(pct - expected) > expected * tolerance * 5:  # Allow 5x tolerance
            return False

    return True


def main():
    parser = argparse.ArgumentParser(description="Domain Labeler for DAFT")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input JSONL file"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output JSONL file (defaults to input with _labeled suffix)"
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only analyze distribution, don't modify"
    )
    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.input}")
    examples = []
    with open(args.input) as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))

    print(f"Loaded {len(examples)} examples")

    labeler = DomainLabeler()

    if args.analyze_only:
        # Just analyze
        stats = labeler.analyze_distribution(examples)

        print("\n" + "=" * 60)
        print("DOMAIN DISTRIBUTION ANALYSIS")
        print("=" * 60)

        print(f"\nTotal examples: {stats['total']}")

        print("\nDomain distribution:")
        for d in range(6):
            count = stats["domain_counts"].get(d, 0)
            pct = stats["domain_percentages"].get(d, 0)
            name = DOMAIN_PATTERNS[d].name if d < len(DOMAIN_PATTERNS) else "unknown"
            print(f"  D{d} ({name}): {count} ({pct:.1f}%)")

        print("\nBy category:")
        for cat, counts in sorted(stats["category_domain_counts"].items()):
            print(f"\n  {cat}:")
            for d, count in sorted(counts.items()):
                print(f"    D{d}: {count}")

    else:
        # Label and save
        labeled = labeler.label_dataset(examples)

        output_path = args.output or args.input.replace(".jsonl", "_labeled.jsonl")
        print(f"Writing labeled data to {output_path}")

        with open(output_path, "w") as f:
            for ex in labeled:
                f.write(json.dumps(ex) + "\n")

        # Print stats
        stats = labeler.analyze_distribution(labeled)
        print("\nDomain distribution:")
        for d in range(6):
            count = stats["domain_counts"].get(d, 0)
            pct = stats["domain_percentages"].get(d, 0)
            print(f"  D{d}: {count} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
