import json
from pathlib import Path

from scripts.audit_region_pack_candidates import build_audit


def test_audit_does_not_claim_unknown_angles(tmp_path: Path) -> None:
    fixture = Path("data/expanded/images.parquet")
    discovery = Path("results/lroc_site_discovery.json")
    curation = Path("results/dataset_curation_lroc_curated_v2.json")
    audit = build_audit(fixture, Path("data/expanded/train.parquet"), discovery, curation)

    assert audit["candidates"]
    e009 = next(row for row in audit["candidates"] if row["site_id"] == "E009S3481")
    assert e009["local_corpus"]["incidence_deg"] is None
    assert json.loads(json.dumps(audit))["scientific_limitations"]
