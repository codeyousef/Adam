#!/usr/bin/env python3
"""Inspect _maybe_chain_followup and why v363 recommended_followup is null"""
import inspect
import sys

sys.path.insert(0, '/mnt/Storage/Projects/catbelly_studio/ignorance-1')
import experiments.run_strict_eval_autoresearch as m

# Find _maybe_chain_followup
src = inspect.getsource(m)
idx = src.find('def _maybe_chain_followup')
if idx >= 0:
    print(f"=== _maybe_chain_followup at idx {idx} ===")
    print(src[idx:idx+2000])
else:
    print("_maybe_chain_followup NOT FOUND")

# Find _write_stop_notice
idx2 = src.find('def _write_stop_notice')
if idx2 >= 0:
    print(f"\n=== _write_stop_notice at idx {idx2} ===")
    print(src[idx2:idx2+2000])
else:
    print("_write_stop_notice NOT FOUND")
