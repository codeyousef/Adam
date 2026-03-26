# CLAUDE.md

This repository's primary active project is `ignorance-1/`.

## Main Project

- `ignorance-1/` is the active JEPA architecture workspace.
- `legacy_adam/` contains older Adam research code and archived artifacts.

When working in this repository:

1. Default to `ignorance-1/` unless the task explicitly mentions Adam.
2. Treat `legacy_adam/` as reference or archive material.
3. Keep root-level documentation aligned with the JEPA project, not the archived Adam pipeline.

## Key JEPA Files

- `ignorance-1/src/models/jepa.py`
- `ignorance-1/src/training/phase1.py`
- `ignorance-1/src/training/phase2.py`
- `ignorance-1/src/training/phase3.py`
- `ignorance-1/src/training/phase4.py`
- `ignorance-1/experiments/validate_phases.py`
- `ignorance-1/autorun.py`
- `ignorance-1/config/ignorance_1.yaml`

## Current Production-Candidate Recipe

- `phase1`: `384` embed dim, `10/12` encoder/predictor layers, `4096` projections, `192` steps
- `phase4`: `15M, 40M, 80M, 150M, 300M, 600M, 1.2B`
- `phase4.steps = 112`
- `phase4.num_splits = 7`
- `phase4.proxy_recipe = v5_distinct`
- `phase4.step_scale_power = 0.55`
- `phase4.max_step_multiplier = 5.0`
- `phase4.lr_scale_power = 0.2`
- `phase4.max_lr_divisor = 2.5`

## Commands

```bash
cd ignorance-1
../.venv/bin/python experiments/validate_phases.py --config config/ignorance_1.yaml
../.venv/bin/python autorun.py --strategy production_readiness --dry-run
```
