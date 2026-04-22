#!/usr/bin/env python3
"""Inspect argparse arguments and main() function in run_strict_eval_autoresearch.py"""
import inspect
import sys

sys.path.insert(0, '/mnt/Storage/Projects/catbelly_studio/ignorance-1')
import experiments.run_strict_eval_autoresearch as m

src = inspect.getsource(m)

# Find the full argparse section
idx = src.find('parser = argparse.ArgumentParser')
if idx == -1:
    idx = src.find('ArgumentParser(')
print(f"argparse at idx {idx}")

# Get the argparse section - find the parser = line and print from there
parser_line_idx = src.find('parser = argparse.ArgumentParser')
if parser_line_idx >= 0:
    # Print 3000 chars of argparse section
    print("=== ARGPARSE SECTION ===")
    print(src[parser_line_idx:parser_line_idx+3000])

# Find main() function
main_idx = src.find('def main(')
if main_idx >= 0:
    print(f"\n=== MAIN at idx {main_idx} ===")
    print(src[main_idx:main_idx+500])
else:
    print("\nNO main() found!")

# Find the budget-hours argument
for line in src.split('\n'):
    if 'add_argument' in line and 'budget' in line.lower():
        print(f"BUDGET: {line.strip()}")
    if 'add_argument' in line and ('cycles' in line.lower() or 'auto' in line.lower()):
        print(f"OTHER: {line.strip()}")
