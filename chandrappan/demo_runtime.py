"""Offline RP-001 presentation runtime using the frozen official RoMa v2.

The runtime consumes only the compact release manifest, processed science
rasters, local descriptors, local thumbnails, and the official checkpoint.
It deliberately does not read training crops, optimizer state, raw PDS3
products, or the experimental D1 checkpoint.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
import torch
import yaml

from chandrappan.geo.footprint import footprint_corners_world
from chandrappan.geo.metadata import LunarImageMetadata, metadata_from_mapping
from chandrappan.geometry.verification import VerificationConfig, verify_registration
from chandrappan.region_packs import intensity_descriptor, rank_references
from chandrappan.rp001_demo import as_roma_tensor, preprocess_science, sha256_file
from chandrappan.runtime import RuntimeDevice, load_official_roma, resolve_device
from chandrappan.training.coordinates import roma_warp_to_pixel


@dataclass(frozen=True)
class DemoPaths:
    """All runtime locations resolved from a relocatable release root."""

    root: Path
    config: Path
    checkpoint: Path
    data_root: Path
    manifest: Path
    artifacts: Path
    frontend: Path

    @classmethod
    def from_root(cls, root: Path) -> DemoPaths:
        release_root = root.resolve()
        config = release_root / "configs" / "demo_mac_m4.yaml"
        if not config.is_file():
            raise FileNotFoundError(f"Mac demo configuration is missing: {config}")
        payload = yaml.safe_load(config.read_text(encoding="utf-8"))
        paths = payload["paths"]
        data_root = release_root / paths["data_root"]
        return cls(
            root=release_root,
            config=config,
            checkpoint=release_root / paths["checkpoint"],
            data_root=data_root,
            manifest=data_root / "demo_manifest.json",
            artifacts=release_root / paths["artifacts"],
            frontend=release_root / paths["frontend"],
        )


def _crop_metadata(
    metadata: LunarImageMetadata, center_world: tuple[float, float], size: int
) -> tuple[LunarImageMetadata, tuple[int, int]]:
    center_x, center_y = metadata.transform.world_to_pixel(*center_world)
    origin_x = max(0, min(metadata.width - size, int(center_x - size / 2)))
    origin_y = max(0, min(metadata.height - size, int(center_y - size / 2)))
    world_origin = metadata.transform.pixel_to_world(origin_x, origin_y)
    return (
        replace(
            metadata,
            width=size,
            height=size,
            transform=replace(
                metadata.transform,
                x_origin=float(world_origin[0]),
                y_origin=float(world_origin[1]),
            ),
        ),
        (origin_x, origin_y),
    )


def _window_mean(mask: np.ndarray, size: int) -> np.ndarray:
    integral = np.pad(mask.astype(np.uint32), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    total = (
        integral[size:, size:]
        - integral[:-size, size:]
        - integral[size:, :-size]
        + integral[:-size, :-size]
    )
    return total / float(size * size)


def _read_window(path: Path, origin: tuple[int, int], size: int) -> tuple[np.ndarray, np.ndarray]:
    import rasterio
    from rasterio.windows import Window

    with rasterio.open(path) as dataset:
        window = Window(origin[0], origin[1], size, size)
        image = dataset.read(1, window=window).astype(np.float32)
        valid = dataset.read_masks(1, window=window) > 0
    return image, valid & np.isfinite(image)


def _descriptor(path: Path) -> np.ndarray:
    import rasterio
    from rasterio.enums import Resampling

    with rasterio.open(path) as dataset:
        image = dataset.read(1, out_shape=(1, 256, 256), resampling=Resampling.average)
    return intensity_descriptor(image)


def _candidate_center(
    source: dict[str, Any], target: dict[str, Any], size: int, minimum_coverage: float
) -> tuple[str, tuple[float, float]] | None:
    """Choose one deterministic, nodata-aware common crop without GT access."""
    import rasterio
    from shapely.geometry import Point, Polygon

    source_meta, target_meta = source["metadata"], target["metadata"]
    overlap = Polygon(footprint_corners_world(source_meta)).intersection(
        Polygon(footprint_corners_world(target_meta))
    )
    if overlap.is_empty or overlap.area <= 0:
        return None
    with rasterio.open(source["raster"]) as first, rasterio.open(target["raster"]) as second:
        source_mask = first.read_masks(1) > 0
        target_mask = second.read_masks(1) > 0
    if source_mask.shape != target_mask.shape:
        return None
    coverage = _window_mean(source_mask & target_mask, size)
    candidates: list[tuple[float, float, int, int, tuple[float, float]]] = []
    for y in range(0, coverage.shape[0], 80):
        for x in range(0, coverage.shape[1], 80):
            value = float(coverage[y, x])
            if value < minimum_coverage:
                continue
            center = source_meta.transform.pixel_to_world(x + size / 2, y + size / 2)
            if overlap.covers(Point(center)):
                distance = float(
                    np.hypot(
                        (x + size / 2) / source_mask.shape[1] - 0.5,
                        (y + size / 2) / source_mask.shape[0] - 0.5,
                    )
                )
                candidates.append((value, distance, y, x, (float(center[0]), float(center[1]))))
    if not candidates:
        return None
    _, _, _, _, center = min(candidates, key=lambda row: (-row[0], row[1], row[2], row[3]))
    return "central", center


def _geometry(
    prediction: dict[str, torch.Tensor], resolution: int, confidence_threshold: float, maximum: int
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    warp = roma_warp_to_pixel(prediction["warp_AB"], (resolution, resolution))[0]
    confidence = prediction["overlap_AB"][0, ..., 0]
    warp_np = warp.detach().cpu().numpy()
    confidence_np = confidence.detach().cpu().numpy()
    yy, xx = np.indices(warp_np.shape[:2], dtype=np.float32)
    source = np.stack((xx + 0.5, yy + 0.5), axis=-1)
    keep = np.isfinite(warp_np).all(axis=-1) & np.isfinite(confidence_np)
    keep &= confidence_np >= confidence_threshold
    keep &= (warp_np[..., 0] >= 0) & (warp_np[..., 0] < resolution)
    keep &= (warp_np[..., 1] >= 0) & (warp_np[..., 1] < resolution)
    indices = np.flatnonzero(keep)
    if len(indices) > maximum:
        indices = indices[np.argsort(-confidence_np.reshape(-1)[indices], kind="stable")[:maximum]]
    source_xy = source.reshape(-1, 2)[indices]
    target_xy = warp_np.reshape(-1, 2)[indices]
    selected_confidence = confidence_np.reshape(-1)[indices]
    verification = verify_registration(
        source_xy, target_xy, (resolution, resolution), config=VerificationConfig()
    )
    return verification.as_dict(), source_xy, target_xy, selected_confidence, warp_np


def _project(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack((points, np.ones(len(points), dtype=np.float32)))
    if matrix.shape == (2, 3):
        return homogeneous @ matrix.T
    projected = homogeneous @ matrix.T
    return projected[:, :2] / projected[:, 2:3]


def spatially_distributed_matches(
    source: np.ndarray,
    target: np.ndarray,
    confidence: np.ndarray,
    verification: dict[str, Any],
    resolution: int,
    limit: int,
) -> list[dict[str, Any]]:
    """Prefer verified inliers spread across occupied 8×8 terrain cells."""
    selected = verification.get("selected")
    if selected is None or not len(source):
        return []
    matrix = np.asarray(selected["matrix"], dtype=np.float32)
    errors = np.linalg.norm(_project(matrix, source) - target, axis=1)
    inliers = np.flatnonzero(errors <= 3.0)
    if not len(inliers):
        return []
    cells: dict[tuple[int, int], list[int]] = {}
    for index in inliers[np.argsort(-confidence[inliers], kind="stable")]:
        x, y = source[index]
        cell = (min(7, int(x * 8 / resolution)), min(7, int(y * 8 / resolution)))
        cells.setdefault(cell, []).append(int(index))
    ordered: list[int] = []
    while cells and len(ordered) < limit:
        for cell in sorted(tuple(cells)):
            ordered.append(cells[cell].pop(0))
            if not cells[cell]:
                del cells[cell]
            if len(ordered) >= limit:
                break
    return [
        {
            "id": f"M{number:02d}",
            "query": [round(float(source[index, 0]), 3), round(float(source[index, 1]), 3)],
            "reference": [round(float(target[index, 0]), 3), round(float(target[index, 1]), 3)],
            "confidence": round(float(confidence[index]), 6),
        }
        for number, index in enumerate(ordered, start=1)
    ]


class DemoRuntime:
    """Long-lived, single-model local presentation runtime."""

    def __init__(self, root: Path, *, device: str = "auto", developer_mode: bool = False) -> None:
        self.paths = DemoPaths.from_root(root)
        self.config = yaml.safe_load(self.paths.config.read_text(encoding="utf-8"))
        self.runtime_device: RuntimeDevice = resolve_device(device)
        self.developer_mode = developer_mode
        self.model: Any | None = None
        self.manifest: dict[str, Any] = {}
        self.records: dict[str, dict[str, Any]] = {}
        self.reference_descriptors: dict[str, np.ndarray] = {}
        self._lock = Lock()
        self.ready = False
        self.fallback_events: list[str] = []

    @property
    def resolution(self) -> int:
        return int(self.config["live_demo_resolution"])

    def _load_manifest(self) -> None:
        if not self.paths.manifest.is_file():
            raise FileNotFoundError(f"RP-001 release manifest is missing: {self.paths.manifest}")
        self.manifest = json.loads(self.paths.manifest.read_text(encoding="utf-8"))
        self.records = {}
        for payload in self.manifest["observations"]:
            record = dict(payload)
            record["raster"] = self.paths.data_root / record["raster"]
            record["metadata"] = metadata_from_mapping(record["metadata"])
            record["thumbnail"] = self.paths.data_root / record["thumbnail"]
            if not record["raster"].is_file():
                raise FileNotFoundError(f"release raster is missing: {record['raster']}")
            descriptor_path = self.paths.data_root / record["descriptor"]
            if not descriptor_path.is_file():
                raise FileNotFoundError(f"observation descriptor is missing: {descriptor_path}")
            record["descriptor_values"] = np.load(descriptor_path)
            self.records[record["product_id"]] = record
        self.reference_descriptors = {}
        for record in self.records.values():
            if record["role"] != "TRAIN":
                continue
            self.reference_descriptors[record["product_id"]] = record["descriptor_values"]
        if len(self.reference_descriptors) != 12:
            raise ValueError("RP-001 release must contain exactly 12 TRAIN reference descriptors")

    def preload(self) -> None:
        """Load metadata/descriptors/model once, then warm the actual local path."""
        self._load_manifest()
        expected_hash = self.manifest["model"]["sha256"]
        actual_hash = sha256_file(self.paths.checkpoint)
        if actual_hash != expected_hash:
            raise ValueError("official RoMa v2 checkpoint checksum mismatch")
        self.model = load_official_roma(self.paths.checkpoint, self.runtime_device)
        if self.runtime_device.torch_device.type == "mps":
            self.fallback_events.append("MPS float32 path: upstream AMP disabled")
        if self.config.get("warmup", True):
            case = self.manifest["known_cases"]["heldout"]
            self._warmup(case["query_product_id"])
        self.ready = True

    def _warmup(self, query_id: str) -> None:
        query = self.records[query_id]
        ranked = rank_references(query["descriptor_values"], self.reference_descriptors)
        reference = self.records[ranked[0].product_id]
        crop = _candidate_center(
            query,
            reference,
            self.resolution,
            float(self.config["matching"]["minimum_common_valid_coverage"]),
        )
        if crop is None:
            raise RuntimeError("warmup query has no valid common crop")
        _, center = crop
        _, source_origin = _crop_metadata(query["metadata"], center, self.resolution)
        _, target_origin = _crop_metadata(reference["metadata"], center, self.resolution)
        source, _ = _read_window(query["raster"], source_origin, self.resolution)
        target, _ = _read_window(reference["raster"], target_origin, self.resolution)
        assert self.model is not None
        with torch.inference_mode():
            self.model.match(
                as_roma_tensor(preprocess_science(source, "raw"), self.runtime_device.torch_device),
                as_roma_tensor(preprocess_science(target, "raw"), self.runtime_device.torch_device),
            )

    def bank_summary(self) -> list[dict[str, Any]]:
        return [
            self._record_summary(record)
            for record in sorted(
                self.records.values(), key=lambda item: (item["role"], item["product_id"])
            )
        ]

    def _record_summary(self, record: dict[str, Any]) -> dict[str, Any]:
        return {
            "product_id": record["product_id"],
            "role": record["role"],
            "acquisition_date": record["acquisition_date"],
            "incidence_deg": record["incidence_deg"],
            "emission_deg": record["emission_deg"],
            "phase_deg": record["phase_deg"],
            "gsd_m_per_px": record["gsd_m_per_px"],
            "sha256": record["sha256"],
            "thumbnail_url": f"/demo-data/{record['thumbnail'].relative_to(self.paths.data_root)}",
        }

    def status(self) -> dict[str, Any]:
        return {
            "server_ready": True,
            "model_ready": self.ready,
            "model": "Official RoMa v2",
            "model_status": "ACTIVE · FROZEN",
            "device": self.runtime_device.ui_label,
            "device_type": self.runtime_device.torch_device.type,
            "resolution": self.resolution,
            "network_required": False,
            "reference_candidates": len(self.reference_descriptors),
            "fallback_events": self.fallback_events if self.developer_mode else [],
        }

    def run_case(self, case_name: str) -> dict[str, Any]:
        if not self.ready or self.model is None:
            raise RuntimeError("model is not ready")
        if case_name not in self.manifest["known_cases"]:
            raise ValueError(f"unknown demo case: {case_name}")
        with self._lock:
            case = self.manifest["known_cases"][case_name]
            if case.get("kind", "observation") == "hard_negative":
                return self._run_hard_negative(case_name, case)
            return self._run_observation(case_name, case)

    def _run_observation(self, case_name: str, case: dict[str, Any]) -> dict[str, Any]:
        query = self.records[case["query_product_id"]]
        query_descriptor = query["descriptor_values"]
        return self._match_ranked(case_name, query, query_descriptor, held_out=True)

    def _run_hard_negative(self, case_name: str, case: dict[str, Any]) -> dict[str, Any]:
        path = self.paths.data_root / case["query_array"]
        if not path.is_file():
            raise FileNotFoundError(f"hard-negative fixture is missing: {path}")
        query_image = np.load(path).astype(np.float32)
        if query_image.shape != (self.resolution, self.resolution):
            raise ValueError("hard-negative fixture has unexpected resolution")
        descriptor = intensity_descriptor(query_image)
        reference = self.records[case["reference_product_id"]]
        target, _ = _read_window(
            reference["raster"], tuple(case["reference_origin"]), self.resolution
        )
        started = time.perf_counter()
        result = self._match_pair(query_image, target)
        result["timing"]["total_seconds"] = round(time.perf_counter() - started, 4)
        candidate = {
            **self._record_summary(reference),
            "rank": 1,
            "retrieval_score": float(
                np.dot(descriptor, self.reference_descriptors[reference["product_id"]])
            ),
            "status": "VERIFIED" if result["verification"]["accepted"] else "GEOMETRY FAILED",
            "verification": result["verification"],
            "geometric_match_count": result["geometric_match_count"],
        }
        artifacts = self._write_artifacts(case_name, query_image, target, result["warp"])
        return self._result_payload(
            case_name,
            query={
                "product_id": "OTHER_LUNAR_REGION",
                "role": "HARD_NEGATIVE",
                "acquisition_date": None,
                "incidence_deg": None,
                "emission_deg": None,
                "phase_deg": None,
                "gsd_m_per_px": None,
                "sha256": case["sha256"],
                "thumbnail_url": None,
            },
            candidates=[candidate],
            selected=candidate,
            result=result,
            artifacts=artifacts,
            held_out=False,
            hard_negative=True,
        )

    def _match_ranked(
        self,
        case_name: str,
        query: dict[str, Any],
        query_descriptor: np.ndarray,
        *,
        held_out: bool,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        ranking_started = time.perf_counter()
        ranking = rank_references(query_descriptor, self.reference_descriptors)
        retrieval_seconds = time.perf_counter() - ranking_started
        candidates: list[dict[str, Any]] = []
        selected: dict[str, Any] | None = None
        selected_result: dict[str, Any] | None = None
        selected_images: tuple[np.ndarray, np.ndarray] | None = None
        for retrieved in ranking[: int(self.config["matching"]["top_k"])]:
            reference = self.records[retrieved.product_id]
            candidate = {
                **self._record_summary(reference),
                "rank": retrieved.rank,
                "retrieval_score": retrieved.score,
                "status": "CANDIDATE",
            }
            crop = _candidate_center(
                query,
                reference,
                self.resolution,
                float(self.config["matching"]["minimum_common_valid_coverage"]),
            )
            if crop is None:
                candidate.update(
                    {
                        "status": "GEOMETRY FAILED",
                        "reason": "no_nodata_aware_common_window",
                    }
                )
                candidates.append(candidate)
                continue
            location, center = crop
            _, source_origin = _crop_metadata(query["metadata"], center, self.resolution)
            _, target_origin = _crop_metadata(reference["metadata"], center, self.resolution)
            source, _ = _read_window(query["raster"], source_origin, self.resolution)
            target, _ = _read_window(reference["raster"], target_origin, self.resolution)
            result = self._match_pair(source, target)
            candidate.update(
                {
                    "crop_location": location,
                    "query_origin": list(source_origin),
                    "reference_origin": list(target_origin),
                    "geometric_match_count": result["geometric_match_count"],
                    "verification": result["verification"],
                    "status": "VERIFIED"
                    if result["verification"]["accepted"]
                    else "GEOMETRY FAILED",
                }
            )
            candidates.append(candidate)
            if result["verification"]["accepted"]:
                candidate["status"] = "SELECTED"
                selected, selected_result, selected_images = candidate, result, (source, target)
                break
        if selected is None:
            selected = candidates[0]
            selected_result = {
                "verification": selected.get("verification", {"accepted": False}),
                "display_matches": [],
                "geometric_match_count": int(selected.get("geometric_match_count", 0)),
                "warp": None,
                "timing": {},
            }
            selected_images = None
        selected_result["timing"]["retrieval_seconds"] = round(retrieval_seconds, 4)
        selected_result["timing"]["total_seconds"] = round(time.perf_counter() - started, 4)
        artifacts = (
            self._write_artifacts(
                case_name, selected_images[0], selected_images[1], selected_result["warp"]
            )
            if selected_images is not None and selected_result["warp"] is not None
            else {}
        )
        return self._result_payload(
            case_name,
            self._record_summary(query),
            candidates,
            selected,
            selected_result,
            artifacts,
            held_out=held_out,
            hard_negative=False,
        )

    def _match_pair(self, source: np.ndarray, target: np.ndarray) -> dict[str, Any]:
        assert self.model is not None
        started = time.perf_counter()
        with torch.inference_mode():
            prediction = self.model.match(
                as_roma_tensor(preprocess_science(source, "raw"), self.runtime_device.torch_device),
                as_roma_tensor(preprocess_science(target, "raw"), self.runtime_device.torch_device),
            )
        inference_seconds = time.perf_counter() - started
        geometry_started = time.perf_counter()
        verification, source_xy, target_xy, confidence, warp = _geometry(
            prediction,
            self.resolution,
            float(self.config["matching"]["confidence_threshold"]),
            int(self.config["matching"]["maximum_geometric_matches"]),
        )
        display = spatially_distributed_matches(
            source_xy,
            target_xy,
            confidence,
            verification,
            self.resolution,
            int(self.config["visualization"]["max_all_visible_matches"]),
        )
        return {
            "verification": verification,
            "display_matches": display,
            "geometric_match_count": int(len(source_xy)),
            "warp": warp,
            "timing": {
                "roma_seconds": round(inference_seconds, 4),
                "geometry_seconds": round(time.perf_counter() - geometry_started, 4),
            },
        }

    def _write_artifacts(
        self, case_name: str, source: np.ndarray, target: np.ndarray, warp: np.ndarray
    ) -> dict[str, str]:
        import cv2

        artifact_root = self.paths.artifacts / f"{case_name}-{uuid.uuid4().hex[:8]}"
        artifact_root.mkdir(parents=True, exist_ok=True)
        source_u8 = np.rint(preprocess_science(source, "raw") * 255).astype(np.uint8)
        target_u8 = np.rint(preprocess_science(target, "raw") * 255).astype(np.uint8)
        registered = cv2.remap(
            target_u8, warp[..., 0], warp[..., 1], cv2.INTER_LINEAR, borderValue=0
        )
        assets = {
            "query": source_u8,
            "reference": target_u8,
            "registered": registered,
            "overlay": cv2.addWeighted(source_u8, 0.5, registered, 0.5, 0),
            "difference": cv2.absdiff(source_u8, registered),
        }
        for name, image in assets.items():
            destination = artifact_root / f"{name}.png"
            if not cv2.imwrite(str(destination), image):
                raise RuntimeError(f"failed to write presentation artifact: {destination}")
        return {name: f"/artifacts/{artifact_root.name}/{name}.png" for name in assets}

    def _result_payload(
        self,
        case_name: str,
        query: dict[str, Any],
        candidates: list[dict[str, Any]],
        selected: dict[str, Any],
        result: dict[str, Any],
        artifacts: dict[str, str],
        *,
        held_out: bool,
        hard_negative: bool,
    ) -> dict[str, Any]:
        accepted = bool(result["verification"].get("accepted"))
        return {
            "case": case_name,
            "mode": "LIVE_LOCAL_INFERENCE",
            "network_required": False,
            "model": "Official RoMa v2",
            "region_id": "RP-001",
            "query": query,
            "query_training_status": "USED FOR TRAINING: NO" if held_out else "HARD NEGATIVE",
            "reference_bank": "RP-001 TRAIN ONLY",
            "candidates": candidates,
            "selected_reference": selected.get("product_id"),
            "selected_rank": selected.get("rank"),
            "verdict": "VERIFIED" if accepted else "REJECTED",
            "reason": (
                None
                if accepted
                else "Visual similarity was insufficient to satisfy geometric verification."
            ),
            "verification": result["verification"],
            "display_matches": result["display_matches"],
            "geometric_match_count": result["geometric_match_count"],
            "artifacts": artifacts,
            "timing": result["timing"],
            "hard_negative": hard_negative,
        }
