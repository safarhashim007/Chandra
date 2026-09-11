# Region Pack candidate audit

This is a pre-selection audit. It ranks repeat-observation sites from the official ODE discovery response and measures only metadata present in the existing local corpus.

## Scientific limitations

- The local full-browse GeoTIFFs are one-band browse representations.
- Illumination/backplane statistics require the official four-band PDS IMG files.
- A browse-product creation time is not substituted for source acquisition time.
- Scores rank acquisition candidates only; they are not registration performance claims.

## Ranked candidates

| Rank | Site | Official repeat products | Local products | Median overlap | Dense-GT valid | GSD span (m/px) | Angle coverage | Score |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | E009S3481 | 83 | 4 | 1.000 | 0 | 0.000 | 0%/0%/0% | 77.650 |
| 2 | E186N1213 | 100 | 6 | 0.185 | 2 | 0.200 | 0%/0%/0% | 76.836 |
| 3 | E207N3357 | 80 | 6 | 1.000 | 0 | 0.000 | 0%/0%/0% | 76.000 |
| 4 | E090S0155 | 65 | 6 | 1.000 | 6 | 0.000 | 0%/0%/0% | 71.750 |
| 5 | E010N0230 | 99 | 2 | 0.117 | 0 | 0.000 | 0%/0%/0% | 64.366 |
| 6 | E199N0308 | 54 | 6 | 0.898 | 4 | 1.700 | 0%/0%/0% | 63.516 |
| 7 | E011N1499 | 88 | 2 | 0.180 | 0 | 0.000 | 0%/0%/0% | 59.903 |
| 8 | E064S3160 | 50 | 2 | 1.000 | 0 | 0.000 | 0%/0%/0% | 59.500 |
| 9 | E018N3346 | 54 | 6 | 0.468 | 12 | 0.500 | 0%/0%/0% | 56.906 |
| 10 | E041N1229 | 32 | 2 | 0.299 | 1 | 0.000 | 0%/0%/0% | 42.068 |
| 11 | E074N3010 | 20 | 2 | 0.784 | 0 | 0.000 | 0%/0%/0% | 37.609 |
| 12 | E223S3203 | 40 | 0 | n/a | 0 | unknown | 0%/0%/0% | 29.000 |

## Current curation contribution

- `E186N1213`: 2 selected / 2 dense-GT-valid local TRAIN pairs.
- `E090S0155`: 6 selected / 6 dense-GT-valid local TRAIN pairs.
- `E199N0308`: 4 selected / 4 dense-GT-valid local TRAIN pairs.
- `E018N3346`: 12 selected / 12 dense-GT-valid local TRAIN pairs.
- `E041N1229`: 1 selected / 1 dense-GT-valid local TRAIN pairs.

No site is selected by this report alone. The selected site's PDS3 source headers and angle backplanes must be acquired and validated before RP-001 can satisfy its acceptance gate.
