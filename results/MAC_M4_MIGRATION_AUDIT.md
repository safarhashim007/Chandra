# Mac M4 migration audit

Date: 2026-09-11
Scope: RP-001 offline presentation path only.

## Verdict

The old demo path was not presentation-safe: it loaded the D1 specialist state despite D1 being
`EXPERIMENTAL · NOT PROMOTED`, rebuilt reference descriptors per query, and relied on the host
Torch Hub cache for DINOv3 source. The new release path loads only the frozen Official RoMa v2
checkpoint, caches descriptors before presentation, and includes the DINOv3 source required by
the upstream official architecture.

## Dependency classification

| Class | Dependencies / components | Mac demo disposition |
|---|---|---|
| CROSS_PLATFORM | Python 3.10+, NumPy, Pillow, OpenCV, Rasterio, PyProj, Shapely, FastAPI, Uvicorn, YAML, Chandrappan geometry and verification | Included in `requirements-demo-mac.txt` or bundled source. |
| MAC_COMPATIBLE | PyTorch, TorchVision, Einops, Apple MPS | Installed as native Apple-Silicon wheels after transfer. Model runs float32 on MPS. |
| CUDA_ONLY | training loops, optimizer/EMA state, CUDA AMP, CUDA cache cleanup, NVIDIA monitoring, CUDA benchmark commands | Not copied into the presentation workflow; no CUDA is attempted on macOS. |
| LINUX_ONLY | Ubuntu virtual environment, `/proc` inspection, `nvidia-smi`, acquisition helpers, Linux package/cache assumptions | Excluded from the release workflow. |
| OPTIONAL | resolution benchmark, device-parity comparison, developer fallback logging, static precomputed page | Included as optional utilities. They do not block normal startup. |
| REMOVE_FROM_DEMO | 7.4 GiB raw PDS3, 247 MiB full GT archive, 97 training crops, training manifests, optimizer/D1 state, corpus acquisition, evaluation suites | Excluded by the bundle builder. |
| REQUIRES_FALLBACK | An MPS operator failure in the upstream PyTorch path | Central resolver uses `MPS → CPU` on macOS. `PYTORCH_ENABLE_MPS_FALLBACK=1` is optional; the runtime does not depend on it. Developer mode records the MPS float32 path. |

## Device audit

`chandrappan.runtime.resolve_device()` is the only presentation device resolver.

- Linux `auto`: CUDA when available, otherwise CPU.
- macOS `auto`: MPS when available, otherwise CPU.
- Explicit CUDA on macOS: hard error.
- All checkpoints use local `map_location` through upstream model construction; inference mode,
  `eval()`, and frozen parameters are enforced.
- The MPS presentation path disables upstream bfloat16 AMP. No approximation or custom kernel is
  substituted.

The upstream RoMa source still contains generic CUDA discovery and a network fallback for a
missing checkpoint. Neither is reachable in the release path: the release passes a verified local
checkpoint and the resolver rejects CUDA on macOS.

## MPS compatibility audit

RoMa local correlation and matching use portable PyTorch operators. No custom CUDA extension is
loaded by this demo path. The upstream descriptor constructor uses `torch.hub.load` for DINOv3
source code; the 1.5 MiB pinned runtime source tree is bundled under `models/torch_hub/`, and runtime sets
that directory before importing RoMa. The DINO weights are already in the official RoMa checkpoint.
If the bundled source is absent, startup fails before any remote request can occur.

## Offline audit

No frontend CDN, Google Font, remote image, NASA/PDS/ODE request, analytics request, Hugging Face
request, or model-weight download is in the runtime route. NASA URLs remain provenance metadata
only. The only upstream remote code branches discovered are disabled by the mandatory local
checkpoint and local DINO source checks.

## Relocation audit

The release runtime derives all paths from the release root. No `/home/tinkerspace` path is in the
release manifest or startup scripts. macOS shell scripts use POSIX `sh`, `pwd -P`, `uname`, and
local `curl`; they do not require `apt`, GNU `sed`, `/proc`, or NVIDIA utilities.

## Required Mac validation

Run after copying the release directory:

```sh
./scripts/setup_mac_m4.sh
./scripts/doctor_mac_demo.py
./scripts/benchmark_demo_resolution.py
./scripts/compare_demo_devices.py --device mps --baseline artifacts/ubuntu_cuda_heldout.json --output artifacts/mps_parity.json
./scripts/preflight_demo.sh
```

Do not claim MPS timing, memory, or parity until those commands pass on the M4.
