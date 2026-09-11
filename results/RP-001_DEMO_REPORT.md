# RP-001 hackathon demo report

## Status

- EXPERIMENTAL REGION-SPECIFIC ADAPTATION: `FAIL`
- RP-001 PRODUCTION GATE: `FAIL`
- Scope: Apollo 15 S-IVB Impact only; not evidence of global lunar generalization.

## Training provenance

- 12 TRAIN acquisitions → 19 GT-valid pairs → 97 640×640 crops.
- Validation used for training: 0; TEST used for training: 0.
- Selected preprocessing: `raw` (TRAIN/VALIDATION comparison only).
- Selected D1 checkpoint: LR `3e-06`, step `10`.

## Outcome

- The real D1 run completed, but it did not improve monitored TRAIN PCK@1 or preserve the selected validation criterion strongly enough for a demo PASS.
- TEST metrics were recorded after checkpoint freeze and were not used for selection.
- Static assets are labeled precomputed examples. The Mac profile is inference-only and requires an MPS smoke test on the presentation device.

## Paired base-640 metrics

| Split | Model | PCK@0.5 | PCK@1 | Median EPE (px) | VRR |
|---|---|---:|---:|---:|---:|
| TRAIN | Official | 0.421628 | 0.612918 | 0.656599 | 16/19 (0.842105) |
| TRAIN | D1 | 0.421307 | 0.612626 | 0.656844 | 16/19 (0.842105) |
| VALIDATION | Official | 0.544387 | 0.745905 | 0.477545 | 3/4 (0.750000) |
| VALIDATION | D1 | 0.544187 | 0.741748 | 0.476929 | 3/4 (0.750000) |
| TEST | Official | 0.799365 | 0.825662 | 0.347608 | 4/4 (1.000000) |
| TEST | D1 | 0.799228 | 0.825722 | 0.347805 | 4/4 (1.000000) |
