# Geometric registration verification

`chandrappan.geometry.verify_registration()` evaluates similarity, affine, and
homography models with OpenCV RANSAC. Models are tried in that order and the
first model satisfying every configured constraint is selected. A more complex
model is not preferred merely because its residual is smaller.

The verifier records match count, inlier count and ratio, mean/median/maximum
inlier reprojection error, local scale, rotation, shear, anisotropy,
determinant, grid coverage, hull coverage, degeneracy, and rejection reasons.
Grid coverage is the occupied fraction of an 8x8 source-image grid by inliers;
hull coverage is the clipped inlier convex-hull area divided by source-image
area. Both are required because a low-residual local cluster is not a reliable
registration.

The initial acceptance configuration is in
`configs/registration_acceptance.yaml`. It was selected only from the real
expanded validation split by maximizing VRR subject to the configured FAR
limit. `T0_v1` is rejected by the threshold-selection API and is never used for
this search.

Definitions:

- `VRR = accepted genuine positive registrations / eligible genuine positives`.
- `FAR = accepted negative registrations / all negative pairs`.
- A match is not a registration. Acceptance requires correspondence count,
  inliers, inlier ratio, reprojection error, coverage, plausible scale and
  rotation, and bounded shear/anisotropy.

## Real corpus

`data/expanded/images.parquet` contains 20 genuine LROC NAC GeoTIFF products
from seven archive photometric regions. The raw products remain outside Git in
`data/raw/`; the manifest retains the archive URL and explicit nulls for
metadata not present in the downloaded label. Split manifests and positive and
negative pair manifests are under `data/expanded/`.

Run the corpus build with:

```bash
/usr/bin/python3.10 scripts/build_lroc_corpus.py --output-root data/expanded
```

Run the untouched validation baseline and optional diagnostics with:

```bash
/usr/bin/python3.10 scripts/evaluate_validation.py \
  --diagnostics-dir results/validation/diagnostics
/usr/bin/python3.10 scripts/select_validation_thresholds.py
```

The current validation corpus has two positive and four easy geographic
negative pairs. It is sufficient to verify mechanics and report a project
baseline, but not a substitute for the unavailable official LunarMatch-NASA
split or benchmark.

## Outliers and confidence

T0 correspondence errors are highly skewed: the current one-pair baseline has
median EPE `1.5818 px`, mean EPE `38.0155 px`, P90 `124.1449 px`, and 41.5% of
valid correspondences above 5 px. The confidence analysis shows that RoMa's
overlap confidence is useful but not a complete registration decision rule:
the top 50% of T0 correspondences have median EPE `0.2559 px` and PCK@1
`0.9180`, while the full set has PCK@1 `0.4697`. Validation shows the same
pattern, with confidence filtering improving correspondence accuracy but not
replacing geometric coverage and plausibility checks.

Full RoMa fine-tuning remains blocked by the existing training quality gate.
