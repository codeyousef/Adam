#!/usr/bin/env python3
"""Quick validation script that outputs parseable results."""
import argparse
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import validation_probes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    print(f"Loading model from {args.checkpoint}...")

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-Coder-3B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa"
    )

    # Load adapter
    model = PeftModel.from_pretrained(base_model, args.checkpoint)
    model = model.merge_and_unload()
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)

    print("\nRunning validation probes...")
    report = validation_probes.run_validation(
        model=model,
        tokenizer=tokenizer,
        step=2000,
        levels=[1, 2, 3, 4],
        device="cuda"
    )

    # Extract accuracies from report
    results = {
        "L1": report.level1_accuracy,
        "L2": report.level2_accuracy,
        "L3": report.level3_accuracy,
        "L4": report.level4_accuracy,
    }

    # Save to JSON
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*60)
    print("VALIDATION RESULTS")
    print("="*60)
    for level, acc in results.items():
        status = "✓" if acc >= 0.85 else "✗"
        print(f"{status} {level}: {acc:.1%}")
    print("="*60)

    # Check L3 specifically
    l3_acc = results["L3"]
    if l3_acc >= 0.85:
        print(f"\n🎉 L3 FIX SUCCESSFUL! L3 accuracy = {l3_acc:.1%}")
        return 0
    else:
        print(f"\n❌ L3 still needs work: {l3_acc:.1%}")
        return 1

if __name__ == "__main__":
    exit(main())
