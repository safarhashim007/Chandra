# Architectural Decisions

## ADR-006 — Mac presentation uses frozen Official RoMa v2 with local DINO source

Date: 2026-09-11

Decision: The Mac demo loads only the official RoMa v2 checkpoint. D1 state is excluded from its runtime bundle and remains provenance-only because it did not improve held-out validation. The upstream RoMa descriptor construction requires DINOv3 source via Torch Hub; its pinned source cache is bundled locally and validated before model construction. This preserves official model behavior without network access. The device resolver selects CUDA only on non-macOS development hosts and MPS then CPU on macOS; MPS uses float32 because upstream bfloat16 AMP support is not assumed.

## ADR-001 — Moon-specific internal coordinates

Date: 2026-09-10

Decision: Chandrappan uses east-positive longitude and planetocentric latitude, with a 1,737,400 m mean lunar radius where spherical projection is needed. Earth EPSG:4326 is not the canonical internal CRS.

Reason: Earth CRS semantics and datum assumptions are inappropriate for lunar product geometry.

## ADR-002 — Official RoMa v2 remains fallback

Date: 2026-09-10

Decision: LunarRoMa cannot replace RoMa v2 until it improves held-out regional metrics at controlled false-accept rate.

Reason: Training loss is not evidence of better lunar registration.

## ADR-003 — Public source is the RoMa implementation authority

Date: 2026-09-10

Decision: RoMa v2 integrations are derived from the inspected public source checkout, not assumed from prompt terminology.

## ADR-004 — Refiner supervision is stage-wise and T0 is never optimized

Date: 2026-09-11

Decision: LunarRoMa supervision uses the RoMa robust Euclidean warp loss at strides 4, 2, and 1. Smooth-L1 remains an explicit legacy ablation only. T0 pair IDs are rejected before any training diagnostic.

Reason: `ConvRefiner` detaches the incoming warp at every stage; a final-stage-only loss leaves earlier refiners unsupervised. The previous diagnostic also used immutable T0, so its numeric collapse is useful historical evidence but invalid as a training procedure.

## ADR-005 — RP-001 hackathon D1 is an isolated experimental exception

Date: 2026-09-11

Decision: A user-authorized regional-specialist experiment may optimize only `refiners.1` from official RoMa v2 weights, using whole-acquisition TRAIN-only RP-001 data and corrected map-derived GT. The official inference path and production fallback remain untouched. Model/preprocessing/checkpoint selection must use TRAIN/VALIDATION only; TEST is held out for reporting. The experiment is labeled `EXPERIMENTAL_REGION_SPECIALIST` irrespective of outcome.

Reason: A hackathon demonstration can investigate small-region adaptation without changing the scientific standard for a production matcher. In the executed D1 run, no material monitored improvement occurred, so both the demo and production gates remain failed.
