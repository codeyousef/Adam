#!/usr/bin/env python3
"""
v417 Launch Script — Extended Hard Negatives for Same-Family Discrimination

AIM: The v415/v416 diagnosis showed the bi-encoder ceiling is real. This experiment
extends hard_negatives for hard families with SAME-FAMILY wrong patterns so the
reranker is trained on the actual wrong code patterns (throttle vs debounce, etc.).

This is a SAFE patch to data.py — only extends existing hard_negatives lists.
Does NOT add new corpus chunks (that caused a syntax error in v417 attempt 1).

v417 vs v419 difference:
  - v417: Only extends hard_negatives (safe data-only change)
  - v419: Scales late_interaction_verifier_weight AND extends hard_negatives

Both tests related hypotheses:
  - v417: More same-family negative signal helps the reranker
  - v419: Higher reranker weight + same-family negatives = better discrimination

v417 is standalone — no CORPUS_CHUNKS changes. If it helps, then
same-family hard negatives are the path. If not, the problem is deeper.
"""
import sys, os, json, subprocess, shutil, yaml
from pathlib import Path

ROOT = Path("/mnt/Storage/Projects/catbelly_studio/ignorance-1")
PY = str(ROOT / "../.venv/bin/python")
RUN_DIR = ROOT / "artifacts/strict_eval_autoresearch_v4/v417-hard-negatives-seed705"
RUN_DIR.mkdir(parents=True, exist_ok=True)

V378_CKPT = ROOT / "artifacts/strict_eval_autoresearch_v378/v378-late-inter-high-weight-seed511-seed514/model.pt"
V338_CKPT = ROOT / "artifacts/strict_eval_autoresearch_v338/v338-promoted-earlier-onset-tiny-mixed-bridge-seed504/model.pt"

# === STEP 1: Safely extend hard_negatives in data.py ===
data_py = ROOT / "src/utils/data.py"
data_content = data_py.read_text()

# Track if we made any changes
patches_applied = []

def extend_hard_negatives(content, family_name, wrong_code, wrong_label):
    """Safely extend the hard_negatives list for a specific family."""
    marker = f'"family": "{family_name}",'
    if marker not in content:
        print(f"  WARNING: family '{family_name}' not found, skipping")
        return content, False
    
    # Find the hard_negatives block for this family and extend it
    # The pattern is: hard_negatives: [...], then "family": "family_name"
    # We need to find the block between hard_negatives and the next family
    import re
    
    # Find the hard_negatives block for this family
    # Pattern: "family": "family_name", ... "hard_negatives": [...], "family":
    pattern = rf'("family": "{family_name}",.*?"hard_negatives": \[)([^\]]+)(\],.*?"family":)'
    
    def replacer(m):
        existing = m.group(2)
        # Check if wrong_code is already in the block
        escaped_wrong = wrong_code.replace('\\', '\\\\').replace('"', '\\"')
        if wrong_code.strip('" \n') in existing or escaped_wrong in existing:
            print(f"  {family_name}: already has this negative, skipping")
            return m.group(0)
        # Add the new negative
        new_block = existing.rstrip() + ',\n                        ' + json.dumps(wrong_code.strip('" \n'))
        patches_applied.append(f"  {family_name}: added {wrong_label}")
        return m.group(1) + new_block + m.group(3)
    
    new_content, count = re.subn(pattern, replacer, content, flags=re.DOTALL, count=1)
    if count == 0:
        print(f"  WARNING: Could not find hard_negatives block for '{family_name}', skipping")
    return new_content, count > 0

# Extend debounce with throttle
new_content, ok = extend_hard_negatives(
    data_content, "debounce",
    '"# task: Throttle a handler to fire at most once per interval\\nlast_fired = 0\\ndef throttle(fn, interval_ms):\\n    def wrapped(*args, **kwargs):\\n        global last_fired\\n        if Date.now() - last_fired >= interval_ms:\\n            last_fired = Date.now()\\n            fn(*args, **kwargs)\\n    return wrapped\\n"',
    "throttle (wrong timing)"
)
if ok:
    data_content = new_content

# Extend frequency with unique
new_content, ok = extend_hard_negatives(
    data_content, "frequency",
    '"# task: Get unique values preserving first-occurrence order\\nseen = set()\\nunique_result = []\\nfor item in items:\\n    if item not in seen:\\n        seen.add(item)\\n        unique_result.append(item)\\n"',
    "unique (ignores repeats)"
)
if ok:
    data_content = new_content

# Extend startswith_js with endswith
new_content, ok = extend_hard_negatives(
    data_content, "startswith_js",
    '"# task: Check if string ends with suffix\\nconst hasSuffix = text.endsWith(suffix);\\n"',
    "endswith (wrong direction)"
)
if ok:
    data_content = new_content

# Extend strip_lines with lstrip and splitlines
new_content, ok = extend_hard_negatives(
    data_content, "strip_lines",
    '"# task: Strip leading whitespace from each line\\nwith open(path) as handle:\\n    rows = [line.lstrip() for line in handle]\\n"',
    "lstrip (wrong direction)"
)
if ok:
    data_content = new_content

new_content, ok = extend_hard_negatives(
    data_content, "strip_lines",
    '"# task: Read file and return all lines including empty ones\\nwith open(path) as handle:\\n    rows = handle.read().splitlines()\\n"',
    "splitlines (includes empty lines)"
)
if ok:
    data_content = new_content

# Extend merge_dicts with left-biased merge
new_content, ok = extend_hard_negatives(
    data_content, "merge_dicts",
    '"# task: Merge two dicts keeping earlier keys on conflict\\nmerged = {}; merged.update(left); merged.update(right)\\n"',
    "left-first merge (wrong precedence)"
)
if ok:
    data_content = new_content

data_py.write_text(data_content)
for p in patches_applied:
    print(f"Applied: {p}")
if not patches_applied:
    print("WARNING: No patches applied — check data.py structure")

# === STEP 2: Build v417 config ===
config = yaml.safe_load((V378_CKPT.parent / "config.yaml").read_text())
phase4 = config.setdefault("phase4", {})

config["seed"] = 705
config["profile"] = "strict-eval-autoresearch-v4-v417-hard-negatives"
config["warm_start_phase3_only"] = False
config["warm_start_model_path"] = str(V378_CKPT)
config["base_model_path"] = str(V338_CKPT)

phase4["seed"] = 705
phase4["steps"] = 300
phase4["phase4_steps"] = 300
phase4["classifier_weight"] = 0.09
phase4["clf_weight"] = 0.09
phase4["query_multiview_weight"] = 1.0
phase4["warm_start_phase3_only"] = False
phase4["warm_start_model_path"] = str(V378_CKPT)
phase4["production_mode"] = False
phase4["production_steps"] = 0
phase4["production_phase4_repeats"] = 0
phase4["phase4_dataset"] = "behavioral_constraints_v2_taxonomy_support_discipline_v1"

config_path = RUN_DIR / "config.yaml"
config_path.write_text(yaml.safe_dump(config, sort_keys=False))
print(f"Config saved to {config_path}")

# === STEP 3: Train v417 ===
model_path = RUN_DIR / "model.pt"
tmp_ckpt = str(model_path) + ".tmp"
print(f"Copying warm_start from {V378_CKPT} -> {tmp_ckpt}")
shutil.copy2(V378_CKPT, tmp_ckpt)

train_cmd = [
    PY, str(ROOT / "train_production.py"),
    "--config", str(config_path),
    "--size", str(int((config.get("sizes") or config.get("phase4", {}).get("sizes", [15_000_000]))[0])),
    "--output", str(model_path),
    "--device", str(config.get("device", "cuda")),
]

print("\nTraining command:", " ".join(train_cmd))
result = subprocess.run(train_cmd, cwd=ROOT, timeout=36000)
print(f"\nReturn code: {result.returncode}")
if result.returncode != 0:
    print("STDERR:", result.stderr[-3000:] if result.stderr else "(none)")
    sys.exit(result.returncode)

print(f"\nModel saved to {model_path}")
print("Run strict eval next:")
print(f"  python test_2.7b.py 15000000 {model_path} 2>&1 | grep -E 'Objective|D=|A=|FP=|SF=|score='")
