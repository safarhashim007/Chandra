"""Gradient-enabled RoMa v2 training forward, separate from official inference."""

from __future__ import annotations

import torch

from chandrappan.training.coordinates import refiner_head_delta_to_pixel


def forward_train(model, image_a: torch.Tensor, image_b: torch.Tensor) -> dict[str, object]:
    """Run the public RoMa modules without calling its inference-only forward."""
    if image_a.ndim != 4 or image_b.ndim != 4 or image_a.shape[1] != 3 or image_b.shape[1] != 3:
        raise ValueError("training images must be BCHW tensors with three channels")
    if image_a.shape[0] != image_b.shape[0]:
        raise ValueError("training image batches must have equal size")
    expected_lr = (model.H_lr, model.W_lr)
    if image_a.shape[-2:] != expected_lr or image_b.shape[-2:] != expected_lr:
        raise ValueError(
            "forward_train requires source-faithful inputs already prepared at "
            f"the selected RoMa resolution {expected_lr}; it will not anisotropically resize them"
        )

    image_a_lr = image_a
    image_b_lr = image_b
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

    if model.H_hr is not None:
        raise ValueError("forward_train currently supports single-resolution RoMa settings only")
    image_a_hr = image_b_hr = None
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
            raw_delta_ab = (output_ab["warp"] - warp_ab) * output_ab["warp"].new_tensor(
                (refiner.cfg.refine_init * width, refiner.cfg.refine_init * height)
            )
            refiners.append(
                {
                    "stride": patch_size,
                    "size": (height, width),
                    "ab": output_ab,
                    "ba": output_ba,
                    "previous_warp_ab": warp_ab,
                    "raw_delta_ab": raw_delta_ab,
                    "pixel_delta_ab": refiner_head_delta_to_pixel(
                        raw_delta_ab, refine_init=refiner.cfg.refine_init
                    ),
                }
            )
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
