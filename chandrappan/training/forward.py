"""Gradient-enabled RoMa v2 training forward, separate from official inference."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def forward_train(model, image_a: torch.Tensor, image_b: torch.Tensor) -> dict[str, object]:
    """Run the public RoMa modules without calling its inference-only forward."""
    if image_a.ndim != 4 or image_b.ndim != 4 or image_a.shape[1] != 3 or image_b.shape[1] != 3:
        raise ValueError("training images must be BCHW tensors with three channels")
    if image_a.shape[0] != image_b.shape[0]:
        raise ValueError("training image batches must have equal size")

    image_a_lr = F.interpolate(
        image_a, size=(model.H_lr, model.W_lr), mode="bicubic", align_corners=False, antialias=True
    )
    image_b_lr = F.interpolate(
        image_b, size=(model.H_lr, model.W_lr), mode="bicubic", align_corners=False, antialias=True
    )
    features_a = model.f(image_a_lr)
    features_b = model.f(image_b_lr)
    coarse = model.matcher(
        features_a,
        features_b,
        img_A=image_a_lr,
        img_B=image_b_lr,
        bidirectional=model.bidirectional,
    )
    warp_ab, confidence_ab = coarse["warp_AB"], coarse["confidence_AB"]
    warp_ba, confidence_ba = coarse["warp_BA"], coarse["confidence_BA"]
    refiners = []

    image_a_hr = (
        None
        if model.H_hr is None
        else F.interpolate(
            image_a, (model.H_hr, model.W_hr), mode="bicubic", align_corners=False, antialias=True
        )
    )
    image_b_hr = (
        None
        if model.H_hr is None
        else F.interpolate(
            image_b, (model.H_hr, model.W_hr), mode="bicubic", align_corners=False, antialias=True
        )
    )
    for stage, (image_a_stage, image_b_stage) in enumerate(
        zip([image_a_lr, image_a_hr], [image_b_lr, image_b_hr])
    ):
        if image_a_stage is None or image_b_stage is None:
            continue
        from romav2.romav2 import _interpolate_warp_and_confidence

        _, _, height, width = image_a_stage.shape
        scale_factor = image_a_stage.new_tensor(
            (width / model.anchor_width, height / model.anchor_height)
        )
        fine_a = model.refiner_features(image_a_stage)
        fine_b = model.refiner_features(image_b_stage)
        for patch_size_text, refiner in model.refiners.items():
            patch_size = int(patch_size_text)
            warp_ab, confidence_ab = _interpolate_warp_and_confidence(
                warp=warp_ab,
                confidence=confidence_ab,
                H=height,
                W=width,
                patch_size=patch_size,
                zero_out_precision=model.H_hr is not None and patch_size == 4 and stage == 1,
            )
            if model.bidirectional:
                warp_ba, confidence_ba = _interpolate_warp_and_confidence(
                    warp=warp_ba,
                    confidence=confidence_ba,
                    H=height,
                    W=width,
                    patch_size=patch_size,
                    zero_out_precision=model.H_hr is not None and patch_size == 4 and stage == 1,
                )
            output_ab = refiner(
                f_A=fine_a[patch_size],
                f_B=fine_b[patch_size],
                prev_warp=warp_ab,
                prev_confidence=confidence_ab,
                scale_factor=scale_factor,
            )
            output_ba = None
            if model.bidirectional:
                output_ba = refiner(
                    f_A=fine_b[patch_size],
                    f_B=fine_a[patch_size],
                    prev_warp=warp_ba,
                    prev_confidence=confidence_ba,
                    scale_factor=scale_factor,
                )
            refiners.append({"stride": patch_size, "ab": output_ab, "ba": output_ba})
            warp_ab, confidence_ab = output_ab["warp"], output_ab["confidence"]
            if output_ba is not None:
                warp_ba, confidence_ba = output_ba["warp"], output_ba["confidence"]

    return {
        "coarse": coarse,
        "refiners": refiners,
        "final": {
            "warp_ab": warp_ab,
            "warp_ba": warp_ba,
            "confidence_ab": confidence_ab,
            "confidence_ba": confidence_ba,
        },
    }
