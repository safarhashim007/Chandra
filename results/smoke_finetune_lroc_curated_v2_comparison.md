# LROC curated v2 smoke comparison

- curation_run: lroc_curated_v2
- curation_sha256: 80641bb801a51e628dbce5d5601e6755fc899f4fcea611e9271694ca7d884a99
- checkpoint_promoted: False
- decision: SMOKE FINE-TUNING GATE: FAIL

| Metric | Official RoMa v2 | Smoke refiner | Delta |
|---|---:|---:|---:|
| median_epe_px | 4.000859 | 3.899777 | -0.101082 |
| mean_epe_px | 94.633530 | 91.752541 | -2.880989 |
| pck_1 | 0.106502 | 0.104818 | -0.001684 |
| pck_3 | 0.347858 | 0.368035 | +0.020177 |
| pck_5 | 0.522611 | 0.525300 | +0.002689 |
| pck_10 | 0.543340 | 0.544783 | +0.001443 |
| vrr | 0.500000 | 0.500000 | +0.000000 |
| far | 0.000000 | 0.000000 | +0.000000 |

## Strict Gate Checks

- median_epe_not_worse: PASS
- pck1_not_materially_worse: FAIL
- vrr_not_materially_worse: PASS
- far_not_materially_worse: PASS
- pck3_or_pck5_improved: PASS
