# Training objective and coordinate contract

The experimental trainer is separate from official RoMa v2 inference. It accepts only already prepared 640×640 base inputs, so it cannot silently anisotropically resize a lunar image.

RoMa target coordinates are x/y normalized values with `pixel = (normalized + 1) * (width, height) / 2` (`align_corners=False`). Master ground truth is in target pixel coordinates, sampled bilinearly at each source-stage cell centre, then converted once to normalized target coordinates.

The refiner objective is the source RoMa robust regression form:

`cs^alpha * ((EPE / cs)^2 + 1)^(alpha / 2)`, where `alpha=.5`, `c=1e-3`, and `cs=c*stride` for strides 4, 2, and 1. Smooth-L1 is retained only to reproduce the legacy ablation. Its beta is normalized-coordinate units: at 640, beta `1.0` is 320 pixels per axis and beta `.01` is 3.2 pixels.

`ConvRefiner` detaches its input warp. Therefore every refiner stage has its own loss; a final-stage-only loss cannot train strides 4 and 2. The official head update makes a raw head delta equal to `raw / 8` pixels for default `refine_init=4`, independent of stride.

EMA (`.999`) tracks only refiner parameters. Overlap BCE and precision NLL are disabled in the controlled run: current geometry-only validity lacks a nodata-aware overlap mask and a verified covisible precision target. They must not be enabled with fabricated labels.

The historical 320px objective experiment used T0 and is blocked from re-execution. The report in `results/training_objective_fix.{json,md}` uses a geographic TRAIN pair and a separate validation split. It remains a strict-gate failure, not evidence for full fine-tuning.
