# Copilot Instructions — IGNORANCE-1 JEPA

The primary active project in this repository is `ignorance-1/`.

Legacy Adam research code was moved into `legacy_adam/` and should only be touched when a task explicitly asks for Adam-era work.

## Quick Reference

| Action | Command |
|--------|---------|
| Read beginner architecture guide | `open ignorance-1/BEGINNER_ARCHITECTURE.md` |
| Run single validation | `cd ignorance-1 && ../.venv/bin/python experiments/validate_phases.py --config config/ignorance_1.yaml --output artifacts/results.json --report REPORT.md` |
| Preview autorun queue | `cd ignorance-1 && ../.venv/bin/python autorun.py --dry-run` |
| Preview production-readiness queue | `cd ignorance-1 && ../.venv/bin/python autorun.py --strategy production_readiness --dry-run` |

## Main Layout

```
ignorance-1/
    src/models/jepa.py            → JEPA architecture
    src/training/phase1.py        → Representation quality
    src/training/phase2.py        → Retrieval-dependent ignorance
    src/training/phase3.py        → Planning probe
    src/training/phase4.py        → Scaling probe
    experiments/validate_phases.py → Single-run validation
    autorun.py                    → Experiment scheduler and retry wrapper
    config/ignorance_1.yaml       → Default production-candidate config
    artifacts/results.tsv         → Main JEPA ledger

legacy_adam/
    ...                           → Archived Adam pipeline and artifacts
```

## Conventions

- Default to `ignorance-1/` for all new work.
- Treat `legacy_adam/` as archive/reference unless the task explicitly says otherwise.
- Keep root-level docs and repo instructions aligned with the JEPA project.
- The current JEPA production-candidate recipe is the default config in `ignorance-1/config/ignorance_1.yaml`.

## Current JEPA Candidate

- `phase1.embed_dim = 384`
- `phase1.encoder_layers = 10`
- `phase1.predictor_layers = 12`
- `phase1.projections = 4096`
- `phase4.sizes = [15M, 40M, 80M, 150M, 300M, 600M, 1.2B]`
- `phase4.steps = 112`
- `phase4.num_splits = 7`
- `phase4.proxy_recipe = v5_distinct`
- `phase4.step_scale_power = 0.55`
- `phase4.max_step_multiplier = 5.0`
- `phase4.lr_scale_power = 0.2`
- `phase4.max_lr_divisor = 2.5`

## Reliability Note

`ignorance-1/autorun.py` retries once when a run fails with a transient CUDA launch failure signature. Treat repeated failures as infrastructure or kernel issues and investigate before trusting the run.
