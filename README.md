# Chandrappan

Scientifically auditable lunar-image localization and verified dense registration for NASA LRO/LROC imagery.

Chandrappan operates in two modes: **G**, metadata-assisted georeferenced registration (the production priority), and **V**, visual retrieval over an explicitly indexed local NASA catalog. RoMa v2 is the baseline matcher; LunarRoMa is an experimental adaptation and never removes the RoMa fallback.

## Quick start

```bash
python scripts/system_check.py
python scripts/doctor.py
make test
```

For the current phase, see `PROJECT_STATE.md`, `TASKS.md`, and `docs/HANDOFF.md`. Detailed operational instructions are in `docs/RUNBOOK.md`.

Dataset/evaluation smoke commands and the immutable project-defined T0 limitations are documented in `docs/DATASET_EVALUATION.md`.

## Safety and scientific claims

The software returns `VERIFIED`, `UNCERTAIN`, or `REJECTED`; it must not turn a failed stage into a result. A verified registration requires metadata/geometry checks, bidirectional consistency, spatial coverage, and configured quality gates—not merely a low fit error.

## Dependencies

Python 3.10+ is required; use the host `python` interpreter (currently Python 3.10) rather than this host's incomplete `python3` build. `requirements.txt` contains runtime dependencies and `requirements-dev.txt` testing/linting tools. RoMa v2 is vendored only for source inspection at `vendor/romav2` and is excluded from this repository's Git history.
