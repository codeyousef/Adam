#!/usr/bin/env python3
"""NLI-based semantic equivalence verification for AMR-LDA augmentation.

Uses bidirectional entailment to verify that augmented variants preserve
the logical meaning of the original examples.
"""

import torch
from dataclasses import dataclass
from typing import Optional
from transformers import AutoModelForSequenceClassification, AutoTokenizer


@dataclass
class NLIResult:
    """Result of NLI verification between two texts."""
    forward_label: str  # ENTAILMENT, NEUTRAL, CONTRADICTION
    forward_score: float
    backward_label: str
    backward_score: float
    is_equivalent: bool
    confidence: float  # min of forward and backward scores


class NLIVerifier:
    """Verify semantic equivalence using bidirectional NLI.

    For two texts to be semantically equivalent:
    - Original must entail variant (forward)
    - Variant must entail original (backward)
    - Both directions must have scores above threshold
    """

    LABEL_MAP = {0: "CONTRADICTION", 1: "NEUTRAL", 2: "ENTAILMENT"}

    def __init__(
        self,
        model_name: str = "roberta-large-mnli",
        device: Optional[str] = None,
        threshold: float = 0.95
    ):
        """Initialize NLI verifier.

        Args:
            model_name: HuggingFace model for NLI
            device: Device to run on (auto-detect if None)
            threshold: Minimum entailment score for equivalence
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = threshold

        print(f"Loading NLI model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        print(f"NLI model loaded on {self.device}")

    def _predict(self, premise: str, hypothesis: str) -> tuple[str, float]:
        """Get NLI prediction for a single premise-hypothesis pair.

        Args:
            premise: The premise text
            hypothesis: The hypothesis text

        Returns:
            Tuple of (label, score) for the prediction
        """
        inputs = self.tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            pred_idx = probs.argmax(dim=-1).item()
            score = probs[0, pred_idx].item()

        return self.LABEL_MAP[pred_idx], score

    def verify_equivalence(
        self,
        original: str,
        variant: str,
        threshold: Optional[float] = None
    ) -> NLIResult:
        """Verify semantic equivalence between original and variant.

        Uses bidirectional entailment:
        - Forward: Does original entail variant?
        - Backward: Does variant entail original?

        Equivalence requires both directions to be ENTAILMENT with
        scores above the threshold.

        Args:
            original: Original text
            variant: Augmented variant text
            threshold: Override default threshold

        Returns:
            NLIResult with verification details
        """
        threshold = threshold or self.threshold

        # Forward: original entails variant?
        fwd_label, fwd_score = self._predict(original, variant)

        # Backward: variant entails original?
        bwd_label, bwd_score = self._predict(variant, original)

        # Check equivalence
        fwd_ok = fwd_label == "ENTAILMENT" and fwd_score >= threshold
        bwd_ok = bwd_label == "ENTAILMENT" and bwd_score >= threshold
        is_equivalent = fwd_ok and bwd_ok

        # Confidence is the minimum of both scores (only if both entailment)
        if fwd_label == "ENTAILMENT" and bwd_label == "ENTAILMENT":
            confidence = min(fwd_score, bwd_score)
        else:
            confidence = 0.0

        return NLIResult(
            forward_label=fwd_label,
            forward_score=fwd_score,
            backward_label=bwd_label,
            backward_score=bwd_score,
            is_equivalent=is_equivalent,
            confidence=confidence
        )

    def verify_batch(
        self,
        originals: list[str],
        variants: list[str],
        threshold: Optional[float] = None
    ) -> list[NLIResult]:
        """Verify equivalence for a batch of pairs.

        Args:
            originals: List of original texts
            variants: List of variant texts
            threshold: Override default threshold

        Returns:
            List of NLIResult for each pair
        """
        assert len(originals) == len(variants), "Mismatched list lengths"
        return [
            self.verify_equivalence(orig, var, threshold)
            for orig, var in zip(originals, variants)
        ]

    def filter_valid_variants(
        self,
        original: str,
        variants: list[str],
        threshold: Optional[float] = None
    ) -> list[tuple[str, NLIResult]]:
        """Filter variants to only those semantically equivalent to original.

        Args:
            original: Original text
            variants: List of candidate variants
            threshold: Override default threshold

        Returns:
            List of (variant, result) tuples for valid variants only
        """
        valid = []
        for variant in variants:
            result = self.verify_equivalence(original, variant, threshold)
            if result.is_equivalent:
                valid.append((variant, result))
        return valid


class FastNLIVerifier(NLIVerifier):
    """Faster NLI verifier using DistilRoBERTa for lower latency.

    Use this for rapid iteration; use full RoBERTa-large for final validation.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-distilroberta-base",
        device: Optional[str] = None,
        threshold: float = 0.90  # Slightly lower threshold for distilled model
    ):
        super().__init__(model_name, device, threshold)


def extract_semantic_content(example: dict) -> str:
    """Extract the key semantic content from an example for NLI comparison."""
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    content = instruction
    if input_text:
        content += " " + input_text
    return content


def run_verification_pipeline(
    input_path: str,
    output_path: str,
    rejected_path: Optional[str] = None,
    model_name: str = "microsoft/deberta-large-mnli",
    threshold: float = 0.95
):
    """Run verification on persona-augmented JSONL data.

    Args:
        input_path: Path to input JSONL with augmented data
        output_path: Path for verified output JSONL
        rejected_path: Optional path for rejected examples
        model_name: NLI model to use
        threshold: Entailment threshold
    """
    import json
    from pathlib import Path
    from tqdm import tqdm

    print(f"Loading data from {input_path}")
    examples = []
    with open(input_path) as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))

    print(f"Loaded {len(examples)} examples")

    # Group by original_id
    groups = {}
    for ex in examples:
        orig_id = ex.get("original_id", id(ex))
        if orig_id not in groups:
            groups[orig_id] = {"original": None, "variants": []}
        if ex.get("persona") == "original":
            groups[orig_id]["original"] = ex
        else:
            groups[orig_id]["variants"].append(ex)

    print(f"Found {len(groups)} original examples with variants")

    # Initialize verifier
    verifier = NLIVerifier(model_name=model_name, threshold=threshold)

    verified = []
    rejected = []

    for orig_id, group in tqdm(groups.items(), desc="Verifying"):
        original = group["original"]

        if original is None:
            for v in group["variants"]:
                v["nli_verified"] = False
                v["nli_reason"] = "no_original_for_comparison"
                verified.append(v)
            continue

        original["nli_verified"] = True
        original["nli_scores"] = {"forward_score": 1.0, "backward_score": 1.0}
        verified.append(original)

        original_content = extract_semantic_content(original)

        for variant in group["variants"]:
            variant_content = extract_semantic_content(variant)
            result = verifier.verify_equivalence(original_content, variant_content)

            variant["nli_verified"] = result.is_equivalent
            variant["nli_scores"] = {
                "forward_label": result.forward_label,
                "forward_score": result.forward_score,
                "backward_label": result.backward_label,
                "backward_score": result.backward_score,
                "confidence": result.confidence
            }

            if result.is_equivalent:
                verified.append(variant)
            else:
                rejected.append(variant)

    # Save outputs
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for ex in verified:
            f.write(json.dumps(ex) + "\n")
    print(f"\nSaved {len(verified)} verified examples to {output_path}")

    if rejected_path and rejected:
        with open(rejected_path, "w") as f:
            for ex in rejected:
                f.write(json.dumps(ex) + "\n")
        print(f"Saved {len(rejected)} rejected examples to {rejected_path}")

    # Statistics
    total_variants = sum(len(g["variants"]) for g in groups.values())
    rejection_rate = len(rejected) / max(total_variants, 1)

    print(f"\n{'='*50}")
    print("VERIFICATION STATISTICS")
    print(f"{'='*50}")
    print(f"Total variants checked: {total_variants}")
    print(f"Verified: {len(verified) - len(groups)}")
    print(f"Rejected: {len(rejected)}")
    print(f"Rejection rate: {rejection_rate*100:.1f}%")

    if rejection_rate < 0.03:
        print("\nWARNING: Rejection rate < 3% - variants may be too similar")
    elif rejection_rate > 0.15:
        print("\nWARNING: Rejection rate > 15% - check persona prompts")
    else:
        print("\nQuality check: PASSED")

    return verified, rejected


def main():
    """CLI for NLI verification pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description="NLI Semantic Verification")
    parser.add_argument("--input", type=str, help="Input JSONL file")
    parser.add_argument("--output", type=str, help="Output JSONL file")
    parser.add_argument("--rejected", type=str, default=None, help="Rejected examples file")
    parser.add_argument("--model", type=str, default="microsoft/deberta-large-mnli")
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--test", action="store_true", help="Run test examples")
    args = parser.parse_args()

    if args.test or not args.input:
        # Run test examples
        verifier = NLIVerifier()
        test_pairs = [
            ("All dogs are mammals.", "If something is a dog, then it is a mammal."),
            ("All A are B.", "Everything that is A is also B."),
            ("All dogs are mammals.", "All mammals are dogs."),
            ("All dogs are mammals.", "All non-mammals are non-dogs."),
        ]

        print("\n" + "=" * 60)
        print("NLI VERIFICATION TEST")
        print("=" * 60)

        for original, variant in test_pairs:
            result = verifier.verify_equivalence(original, variant)
            status = "EQUIVALENT" if result.is_equivalent else "NOT EQUIVALENT"
            print(f"\nOriginal: {original}")
            print(f"Variant:  {variant}")
            print(f"Result:   {status} (confidence: {result.confidence:.3f})")
    else:
        run_verification_pipeline(
            input_path=args.input,
            output_path=args.output,
            rejected_path=args.rejected,
            model_name=args.model,
            threshold=args.threshold
        )


if __name__ == "__main__":
    main()
