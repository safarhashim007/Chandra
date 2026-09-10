# Training objective correction

## Result

The historical T0 refiner experiment is no longer permitted as training evidence. Its collapse is explained by a normalized-unit Smooth-L1 objective, final-stage-only supervision across deliberate refiner detach boundaries, and invalid T0 use—not by evidence of reversed coordinates.

The corrected 640x640, non-T0 TRAIN experiment improves its train-pair EPE at every tested learning rate. The selected conservative run (robust loss, LR 1e-6, 50 steps) improves held-out validation median EPE 4.0009→3.8998 px, PCK@3 .3479→.3680, and PCK@5 .5226→.5253 while retaining VRR .5 and FAR .0. It decreases PCK@1 .10650→.10482, so the strict gate fails.

## Corrected objective

`cs^alpha * ((EPE / cs)^2 + 1)^(alpha / 2)` with `alpha=0.5`, `c=1e-3`, and separate supervised strides 4, 2, and 1. EMA uses decay .999 for refiner parameters only. Overlap and precision losses are deliberately disabled because the available GT has no nodata-aware overlap mask or verified covisible precision target.

## Validation gate

- median EPE: 4.0009 → 3.8998
- PCK@1: 0.106502 → 0.104818
- PCK@3: 0.347858 → 0.368035
- PCK@5: 0.522611 → 0.525300
- VRR/FAR: 0.500/0.000 → 0.500/0.000
- gate checks: {'median_epe_not_worse': True, 'pck1_not_materially_worse': False, 'vrr_not_materially_worse': True, 'far_not_materially_worse': True, 'pck3_or_pck5_improved': True}

FULL TRAINING GATE: FAIL
