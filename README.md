# Catbelly Studio

This repository now treats `ignorance-1/` as the main active project.

`ignorance-1` is a JEPA-style research system that validates latent-space learning with four phases:

- `L1`: representation quality
- `L2`: retrieval-dependent ignorance
- `L3`: latent planning
- `L4`: scaling behavior across proxy model sizes

The previous Adam work has been preserved under `legacy_adam/`.

## Main Project

- Primary workspace: `ignorance-1/`
- Beginner overview: `ignorance-1/BEGINNER_ARCHITECTURE.md`
- Default config: `ignorance-1/config/ignorance_1.yaml`
- Runner: `ignorance-1/autorun.py`
- Single validation entrypoint: `ignorance-1/experiments/validate_phases.py`
- Results ledger: `ignorance-1/artifacts/results.tsv`

## Current JEPA Candidate

The current production-candidate recipe uses:

- `phase1`: `384` dim, `10` encoder layers, `12` predictor layers, `4096` projections
- `phase4` sizes: `15M, 40M, 80M, 150M, 300M, 600M, 1.2B`
- `phase4` steps: `112`
- `phase4` splits: `7`
- `phase4` step scaling: `0.55`
- `phase4` LR scaling: `0.2`

## Typical Commands

Run a single validation:

```bash
cd ignorance-1
../.venv/bin/python experiments/validate_phases.py \
  --config config/ignorance_1.yaml \
  --output artifacts/results.json \
  --report REPORT.md
```

Preview autorun queue:

```bash
cd ignorance-1
../.venv/bin/python autorun.py --strategy production_readiness --dry-run
```

## Legacy Adam

All Adam-era training code, artifacts, and documentation were moved into:

- `legacy_adam/`

Nothing there was deleted; it was only reorganized so the JEPA work is the default surface.
