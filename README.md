# CHANDRA

### Multi-modal Lunar Image Correspondence & Registration

CHANDRA is a lunar image correspondence and registration system developed in response to **Smart India Hackathon Problem Statement 26166** from the Indian Space Research Organisation (ISRO). It is designed to establish reliable correspondences between Chandrayaan-2 optical imagery and reference lunar imagery despite changes in illumination, viewpoint, sensor characteristics, and scale.

The current implementation combines learned dense matching, consistency checks, robust geometric verification, spatial coverage analysis, and interpretable registration outputs. The production path retains official RoMa v2 inference; LunarRoMa remains experimental and is not promoted.

## SIH Problem Statement

| Field | Details |
|---|---|
| Problem Statement ID | **26166** |
| Title | **Multi-modal, Sun angle and scale invariant image correspondence using Chandrayaan-2 optical images (OHRC, TMC and IIRS)** |
| Organization | **Indian Space Research Organisation (ISRO)** |
| Department | **Department of Space / Indian Space Research Organisation** |
| Category | **Software** |
| Theme | **Space Technology** |

## The Problem

Image registration aligns a source (moving) image with a reference (fixed) image of the same scene in a common coordinate system. The source image is geometrically transformed; the reference image supplies the coordinate frame. For lunar imagery, reliable corresponding terrain points must be found before the transformation can be estimated and validated.

The official objective is a generic solution for finding correspondences between Chandrayaan-2 optical imagery and reference lunar imagery, targeting sub-pixel source-image registration accuracy and spatially uniform correspondence distribution. Expected products include match points, a registered image, and quantitative evaluation.

## Why Lunar Registration Is Difficult

### Illumination variation

Changes in Sun azimuth, Sun elevation, surface lighting, shadow direction, and shadow length can make the same crater or ridge look substantially different.

### Viewpoint variation

Different camera positions and orientations introduce translation, rotation, scale change, and perspective distortion.

### Scale variation

Orbital altitude, spatial resolution, and optical-system differences can create large scale changes, including cross-instrument comparisons.

## Our Approach

~~~text
Chandrayaan-2 or reference lunar image
                  |
                  v
            Preprocessing
                  |
                  v
       Candidate reference ranking
                  |
                  v
          RoMa v2 dense matching
                  |
                  v
     Bidirectional consistency checks
                  |
                  v
          Confidence filtering
                  |
                  v
       Geometric hypothesis generation
                  |
                  v
       RANSAC robust verification
                  |
                  v
        Transform and registration
                  |
                  v
       Match views, outputs, metrics
~~~

**Retrieval proposes; geometry decides.** A visually plausible match is not accepted from image similarity or RMSE alone.

## Processing Pipeline

The implemented local runtime validates a map-projected lunar raster, ranks eligible regional references with a deterministic normalized intensity descriptor, extracts a common valid crop, runs official RoMa v2 dense matching, filters by overlap confidence and bidirectional consistency, selects spatially distributed correspondences, and verifies the result geometrically.

OpenCV RANSAC evaluates similarity, affine, and homography hypotheses in that order. The acceptance policy checks inlier count, inlier ratio, reprojection error, scale, rotation, shear, anisotropy, determinant, grid coverage, and hull coverage before producing a registration verdict and artifacts.

## Core Technologies

- Python 3.10+
- PyTorch
- Official RoMa v2 for dense learned correspondence
- OpenCV for robust geometric estimation and image remapping
- FastAPI and Uvicorn for the local API
- Rasterio, PyProj, and Shapely for raster metadata and lunar geometry
- React/Vite for the local frontend

### RoMa v2

RoMa v2 is the primary learned correspondence model. It produces dense visual correspondences for terrain where conventional sparse descriptors may struggle. RoMa is an upstream model; CHANDRA supplies the lunar data contracts, runtime adapter, validation policy, registration logic, provenance, and inspection experience around it.

### Classical baselines

The current production matching path is RoMa-based. The geometry and evaluation modules are structured for classical feature-match comparisons such as SIFT or ORB where those baselines are added; they are not presented as the final matching engine.

## Geometric Verification

Visual matchers propose correspondences. Geometry determines whether those correspondences can describe the same physical lunar surface.

**RANSAC** means Random Sample Consensus. It rejects outliers while fitting a geometrically consistent transform. CHANDRA evaluates similarity transforms, affine transforms, and homographies, then applies physical-plausibility and spatial-coverage checks. A rejected candidate is a scientific outcome, not a software error.

Verdicts are **VERIFIED**, **UNCERTAIN**, or **REJECTED**.

## Outputs

Depending on the runtime profile and verdict, CHANDRA exposes:

- candidate correspondence points and confidence values;
- verified inlier match lines;
- selected reference and candidate rank;
- estimated transformation parameters;
- registered imagery;
- overlay, split, blink, and absolute-difference views;
- inlier count, inlier ratio, reprojection error, and coverage metrics;
- lunar crop centre and source/reference provenance.

## Evaluation

The repository calculates or records RMSE/error statistics where the protocol provides them, candidate correspondence count, inliers, inlier ratio, reprojection error, scale, rotation, coverage, PCK, EPE, verified registration rate (VRR), and false-accept rate (FAR).

The current recorded validation baseline uses **20 LRO NAC observations**, with **2 positive and 4 negative evaluated pairs**. Untouched official RoMa v2 recorded VRR **0.5** and FAR **0.0** under that project protocol. This is a small validation corpus, not an official SIH benchmark or evidence of global lunar performance.

Sub-pixel registration is an SIH target, not a current guarantee across the required Chandrayaan-2 dataset.

## Dataset

CHANDRA is being developed around the datasets defined by SIH Problem Statement 26166.

### Source imagery

The central source imagery is from Chandrayaan-2 optical payloads:

- OHRC
- TMC-2
- IIRS

Dataset portal: [Chandrayaan-2 image browser](https://chmapbrowse.issdc.gov.in/)

The checked-in repository does not include a Chandrayaan-2 source corpus. Raw imagery and model weights are excluded from version control.

### Reference imagery

Reference lunar imagery specified by the problem statement may include LRO NAC and SELENE imagery.

- [LRO image downloads](https://lroc.im-ldi.com/images/downloads/)
- [LRO QuickMap](https://quickmap.lroc.im-ldi.com/)

The available local validation/demo corpus is a small map-projected LRO NAC regional set used to verify mechanics, geometry, provenance, and the presentation path. It is not a substitute for the official Chandrayaan-2 evaluation set.

## Repository Structure

~~~text
api/             FastAPI metadata and local-demo API
Python package/  Lunar geometry, data, matching, evaluation, and runtime code
configs/         Region, acceptance, training, and demo configuration
data/            Manifests, splits, and ignored local data locations
docs/            Architecture, runbooks, reports, and technical notes
frontend/        React/Vite local mission console
results/         Evaluation reports and verification records
scripts/         Acquisition, dataset, evaluation, packaging, and demo tools
tests/           Regression and scientific-contract tests
Makefile         Formatting, lint, health, and test entry points
~~~

## Installation

Use a coherent Python 3.10+ environment with SQLite enabled:

~~~bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt -r requirements-dev.txt
.venv/bin/python scripts/doctor.py
~~~

## Running CHANDRA

### Local demo/API

~~~bash
./scripts/start_workstation.sh
~~~

This expects a separately prepared local demo bundle and serves the API at http://127.0.0.1:8000.

### Apple Silicon presentation bundle

~~~bash
./scripts/setup_mac_m4.sh
./scripts/start_demo_mac.sh
~~~

The macOS arm64 profile is local and offline after setup, uses the frozen official RoMa v2 checkpoint, and selects MPS when available with CPU fallback. Mac M4 performance, parity, memory, and browser rehearsal remain pending validation.

### Frontend development

~~~bash
cd frontend
npm install
npm run dev
~~~

Build the static frontend with npm run build.

## Verification Commands

~~~bash
make format
make lint
make test
python scripts/doctor.py
~~~

Focused checks are available through make test-geo, make test-matching, make test-training, and make test-api.

## Current Limitations

- The available evaluation corpus is small and does not represent the official Chandrayaan-2 benchmark.
- The current regional validation result is not a global lunar retrieval or localization claim.
- Held-out positive verification is incomplete under the strict acceptance protocol.
- Shadow and crater-semantic hard-negative coverage is incomplete.
- The experimental LunarRoMa/refiner path did not pass the strict no-regression gate and is not promoted.
- Official RoMa v2 remains the production fallback.

## Roadmap

Add validated Chandrayaan-2 OHRC, TMC-2, and IIRS products; expand geographically isolated positive and hard-negative evaluation; improve illumination and cross-sensor robustness without weakening acceptance thresholds; and revisit lunar-specific adaptation only after the required coordinate, ground-truth, loss, gradient, overfit, and immutable-baseline gates pass.

Broader visual localization, a larger reference catalogue, DEM-assisted verification, and active perception remain future research directions rather than current capabilities.

## Acknowledgements and References

CHANDRA is developed in response to ISRO’s Smart India Hackathon Problem Statement 26166. Chandrayaan-2 source imagery and the reference lunar imagery listed above are governed by their respective data portals and problem-statement terms.

The active dense matcher is the official RoMa v2 implementation. See the [RoMa v2 paper](https://arxiv.org/abs/2511.15706) and the repository’s [integration notes](docs/ROMAV2_INTERNALS.md).

See also the [registration verification notes](docs/REGISTRATION_VERIFICATION.md), [architecture](docs/ARCHITECTURE.md), [demo report](results/RP-001_DEMO_REPORT.md), and [Mac presentation readiness](results/MAC_M4_DEMO_READINESS.md).
