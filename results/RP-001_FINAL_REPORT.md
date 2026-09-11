# RP-001 final report — not accepted

## Site and provenance

- Region Pack: `RP-001`
- Site: Apollo 15 S-IVB Impact (`E009S3481`)
- Centre: 0.924287706°S, 348.110097572°E; planetocentric latitude and east-positive longitude.
- Canonical lunar radius: 1,737,400 m.
- Source: official NASA LROC SDPPHO four-band PDS3 map-projected NAC products.
- Selection basis: [candidate audit](region_pack_candidate_audit.md) ranked this site first: 83 official repeat products, identical verified selected map bounds, and a wide acquisition span.
- Dataset provenance manifest SHA256: `fea33eff03d21ac679eafa246e488052d7e1d6ed799d25694f621bc1f1cd2a32`.
- Frozen split SHA256: `4fe0c25a79c5986c61acccab216044cfaa8f8c5bf9e52faf5f664860265c089d`.

## Observation set

Twenty full observations were acquired, SHA256-recorded, PDS-header validated, and compared against their official map-projected browse TIFFs.

- TRAIN: 12 full acquisitions
- VALIDATION: 4 full acquisitions
- TEST: 4 full acquisitions, permanently held out from regional training and threshold tuning
- Split leakage check: PASS
- Exact selected product IDs, source URLs, per-file SHA256 values, and raw paths: [RP-001 configuration](../configs/regions/rp001.yaml)
- Local catalog: `data/region_packs/RP-001/observation_catalog.csv` (ignored raw-data artifact)

The selected product IDs are `M106949300R`, `M109311328R`, `M114030251R`, `M160030722R`, `M172995906R`, `M177711422R`, `M185962468R`, `M1096572724R`, `M1106002958L`, `M1111890780R`, `M1118958225R`, `M1129574893R`, `M1138987738R`, `M1160184176R`, `M1177841115R`, `M1213150577R`, `M1256675053R`, `M1291945613R`, `M1323677870R`, and `M1345996066R`; each has the `NAC_PHO_E009S3481_` prefix.

## Measured diversity and geometry

- Acquisition dates: 2009-09-07 through 2020-06-05.
- Incidence backplane-median range: 4.497–83.554°.
- Emission backplane-median range: 2.074–27.820°.
- Phase backplane-median range: 1.906–81.933°.
- GSD: 5.000 m/px for all 20 observations. This pack has no GSD diversity.
- Canonical-footprint overlap: 100% for every selected observation.
- Map-derived GT: 162 pair artifacts (TRAIN pairs plus held-out query-to-TRAIN references); maximum A→B→A/B→A→B numerical cycle error: `1.28e-11 px`.

Backplanes provide sampled product-level angle statistics. Map-derived GT is geospatial correspondence supervision, not surveyed landmark truth; relative registration accuracy is therefore reported separately from absolute lunar geolocation accuracy.

## Frozen official RoMa v2 baseline

Production inference used the untouched official RoMa v2 checkpoint in its `precise` bidirectional setting. Regional fine-tuning was not attempted.

- RoMa checkpoint SHA256: `1557dec0d21b62366465f7ff4d5fdf228cc695d0582e196ad2b80e05230828b7`.
- Retrieval: deterministic normalized intensity-histogram cosine ranking over TRAIN-only regional references; top K = 3. Retrieval is a candidate proposer only.
- Held-out positive TEST verification rate: 2/4 = 0.5.
- Evaluated hard-negative FAR: 0/3 = 0.0.
- Dense metrics on the top-ranked candidate's 16,000 filtered, overlap/cycle/precision-valid held-out correspondences: PCK@0.5 0.4252, PCK@1 0.7460, PCK@2/3/5 1.0; median EPE 0.5931 px, mean EPE 0.6553 px, P90/P95/P99 1.3421/1.4574/1.6544 px.

The low FAR does not offset the 0.5 positive verification rate. Registration acceptance requires the configured coverage, transform-plausibility, residual, and inlier gates; low residual by itself never accepts a match.

## Refinement and negatives

- ZNCC is run only after geometric verification, with border, ambiguity, flat-peak, and reverse-consistency rejection.
- No NCC proposal survived the non-degradation rule. Proposed refinements that increased geometric residual were rejected and are not reported as improvements.
- Three negative cases were evaluated and rejected: an unrelated lunar region, a coarse photometric hard candidate (not a semantic crater-similarity claim), and a same-region disjoint crop.
- A validated shadow mask and independent crater-semantic similarity labels are unavailable, so those required negative categories are explicitly incomplete.

## Demo and frontend artifacts

The local, ignored data artifact contains TRAIN/VALIDATION/TEST contact sheets, raw NASA files, the processed science rasters, metadata, GT, and the TRAIN-only reference bank. The read-only frontend endpoints expose region and observation metadata:

- `GET /api/frontend/regions`
- `GET /api/frontend/regions/RP-001`
- `GET /api/frontend/regions/RP-001/observations`

Per-query overlay, difference-image, and match-line demo artifacts are not produced for an accepted full suite because the acceptance gate is not met.

## Acceptance decision

**RP-001 is not complete and must not be presented as a successful flagship registration pack.**

It has a validated acquisition foundation, frozen split, reference bank, official bidirectional RoMa baseline, geometric safeguards, and a partial negative result. It still needs: improved held-out positive VRR, accepted non-degrading NCC controls or an explicit disabled-refinement policy, validated shadow/semantic hard negatives, per-query visual artifacts, and a full acceptance-gate rerun.

No LunarRoMa training was attempted or promoted. The production fallback remains untouched official RoMa v2.

## Reproducibility state

- Git HEAD at report time: `9e26c22`.
- The RP-001 implementation and existing curation expansion were uncommitted at report time; this is not a clean-release commit.
- Dataset version: RP-001 configuration SHA256 `0e0e4d4dd4a106fae0836d5a2ec8666bab85ca42f1b83757f2466473c79627d2`.
