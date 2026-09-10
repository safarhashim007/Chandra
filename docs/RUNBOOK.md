# Runbook

## Bootstrap

```bash
scripts/bootstrap.sh
make doctor
make test
```

## Required startup checks

```bash
git status
git branch --show-current
git log --oneline --decorate -15
python scripts/doctor.py
make test
```

## Failure policy

Capture the exact command, traceback, inputs, device, shapes, ranges, CRS, and configuration. Trace backward to the first invalid value, reduce to a minimal reproduction, add a regression test, then rerun unit, component, original, and integration checks.
