"""Deterministic geographic splitting and cross-split leakage checks."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence

from shapely import wkt

from .manifest import ImageRecord


def geographic_split(
    records: Sequence[ImageRecord], *, seed: int = 0
) -> dict[str, list[ImageRecord]]:
    """Assign whole region groups to train/validation/test deterministically."""
    groups: dict[str, list[ImageRecord]] = {}
    for record in records:
        groups.setdefault(record.region_id, []).append(record)
    names = sorted(groups)
    random.Random(seed).shuffle(names)
    count = len(names)
    if count >= 3:
        # Keep at least two independent groups in validation/test when a corpus
        # is large enough to support both positive and negative evaluation.
        test_count = max(2 if count >= 6 else 1, round(count * 0.2))
        val_count = max(2 if count >= 6 else 1, round(count * 0.2))
        test_names = set(names[:test_count])
        val_names = set(names[test_count : test_count + val_count])
    elif count == 2:
        test_name = max(names, key=lambda name: (len(groups[name]), name))
        test_names, val_names = {test_name}, set()
    else:
        test_names, val_names = set(), set()
    result = {"train": [], "validation": [], "test": []}
    for name in names:
        split = "test" if name in test_names else "validation" if name in val_names else "train"
        result[split].extend(groups[name])
    return result


def validate_no_leakage(
    splits: Mapping[str, Sequence[ImageRecord]], *, min_overlap_ratio: float = 0.1
) -> None:
    assigned_ids: dict[str, str] = {}
    assigned_regions: dict[str, str] = {}
    for split, records in splits.items():
        for record in records:
            if record.image_id in assigned_ids:
                raise ValueError(f"image reused across splits: {record.image_id}")
            assigned_ids[record.image_id] = split
            previous = assigned_regions.get(record.region_id)
            if previous is not None and previous != split:
                raise ValueError(f"region reused across splits: {record.region_id}")
            assigned_regions[record.region_id] = split
    split_names = list(splits)
    for index, first_name in enumerate(split_names):
        for second_name in split_names[index + 1 :]:
            for first in splits[first_name]:
                first_geometry = wkt.loads(first.footprint_wkt)
                for second in splits[second_name]:
                    second_geometry = wkt.loads(second.footprint_wkt)
                    denominator = min(first_geometry.area, second_geometry.area)
                    if (
                        denominator
                        and first_geometry.intersection(second_geometry).area / denominator
                        >= min_overlap_ratio
                    ):
                        raise ValueError(
                            f"forbidden footprint overlap: {first.image_id} / {second.image_id}"
                        )
