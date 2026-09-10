# Architectural Decisions

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
