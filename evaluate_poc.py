"""
evaluate_poc.py — Evaluate from-scratch pretrained Adam PoC checkpoint.

Loads the checkpoint directly as a Qwen2ForCausalLM (no base model, no PEFT),
then runs the L1-L4 validation probes.

Usage:
    python evaluate_poc.py --checkpoint adam_poc_checkpoints/checkpoint-113000
"""

import argparse
import json
import torch
from transformers import AutoTokenizer, Qwen2ForCausalLM
import validation_probes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint directory")
    parser.add_argument("--output", default="poc_eval_results.json")
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-0.5B")
    args = parser.parse_args()

    print(f"Loading from-scratch model from {args.checkpoint}...")
    model = Qwen2ForCausalLM.from_pretrained(
        args.checkpoint,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Parameters: {n_params:.1f}M")

    print("\nRunning L1-L4 validation probes...")
    report = validation_probes.run_validation(
        model=model,
        tokenizer=tokenizer,
        step=0,
        levels=[1, 2, 3, 4],
        device="cuda",
    )

    results = {
        "checkpoint": args.checkpoint,
        "L1": report.level1_accuracy,
        "L2": report.level2_accuracy,
        "L3": report.level3_accuracy,
        "L4": report.level4_accuracy,
    }

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    targets = {"L1": 0.70, "L2": 0.35, "L3": 0.60, "L4": 0.75}
    print("\n" + "=" * 50)
    print("VALIDATION RESULTS — Adam PoC (from scratch)")
    print("=" * 50)
    all_pass = True
    for level, acc in results.items():
        if level == "checkpoint":
            continue
        target = targets[level]
        status = "PASS" if acc >= target else "FAIL"
        if acc < target:
            all_pass = False
        print(f"  {status}  {level}: {acc:.1%}  (target ≥ {target:.0%})")
    print("=" * 50)
    print(f"  Output: {args.output}")
    if all_pass:
        print("  ALL TARGETS MET")
    else:
        print("  SOME TARGETS MISSED")


if __name__ == "__main__":
    main()
