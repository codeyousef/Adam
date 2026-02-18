#!/usr/bin/env python3
"""Fix persona augmentation to ensure string responses, then regenerate data."""
import json
import sys
from pathlib import Path

def flatten_response(response):
    """Convert dict or any type to string."""
    if isinstance(response, str):
        return response
    elif isinstance(response, dict):
        # Extract meaningful content from structured response
        # Priority: result -> conclusion -> response -> JSON dump
        if "result" in response:
            result = response["result"]
            if isinstance(result, dict):
                answer = result.get("answer", "")
                justification = result.get("justification", "")
                return f"{answer}\n\n{justification}".strip()
            return str(result)
        elif "conclusion" in response:
            return str(response["conclusion"])
        elif "response" in response:
            return str(response["response"])
        else:
            # Fallback: serialize to readable format
            return json.dumps(response, indent=2)
    else:
        return str(response)

def fix_data_file(input_path: str, output_path: str):
    """Fix data file by ensuring preferred/rejected are strings."""
    fixed = 0
    total = 0

    with open(input_path) as f_in, open(output_path, 'w') as f_out:
        for line in f_in:
            if not line.strip():
                continue

            total += 1
            sample = json.loads(line)

            # Fix preferred field
            if "preferred" in sample:
                if not isinstance(sample["preferred"], str):
                    sample["preferred"] = flatten_response(sample["preferred"])
                    fixed += 1
            else:
                print(f"Warning: Sample {total} missing 'preferred' field")
                continue

            # Fix rejected field
            if "rejected" in sample:
                if not isinstance(sample["rejected"], str):
                    sample["rejected"] = flatten_response(sample["rejected"])
                    fixed += 1
            else:
                print(f"Warning: Sample {total} missing 'rejected' field")
                continue

            f_out.write(json.dumps(sample) + '\n')

    print(f"Fixed {fixed} fields in {total} samples")
    print(f"Output: {output_path}")
    return total

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python fix_and_regenerate_data.py <input.jsonl> <output.jsonl>")
        sys.exit(1)

    fix_data_file(sys.argv[1], sys.argv[2])
