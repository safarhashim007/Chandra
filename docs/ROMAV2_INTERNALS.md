# RoMa v2 internals (source inspection)

Inspected source: `vendor/romav2` commit `95c9968145c8906b7b59383258e9f73b02853d89`.

| Concern | Source-derived finding |
| --- | --- |
| Entry class | `src/romav2/romav2.py:RoMaV2` |
| Official inference | `RoMaV2.forward`, marked `@torch.inference_mode()` and asserts `not self.training` |
| Descriptor | `features.Descriptor`, stored as `model.f` |
| Coarse matcher | `matcher.Matcher`, stored as `model.matcher`; exposes `attn_AB_logits`, `attn_AB`, `warp_AB`, and `confidence_AB` (plus BA when bidirectional) |
| Fine features | `features.FineFeatures`, stored as `model.refiner_features` |
| Refiners | `refiner.Refiners`; configured stages are strides 4, 2, and 1 |
| Warp | BHWC x/y target coordinates normalized to `[-1, 1]`; grid sampling uses `align_corners=False` |
| Confidence | Matcher emits overlap logit; refiner confidence has overlap plus three precision parameters. `_map_confidence` maps overlap by sigmoid and precision via `prec_mat_from_prec_params`. |
| Precision update | `ConvRefiner` uses softplus-positive Cholesky diagonal terms and emits symmetric precision parameters `[p00, p10, p11]`. |
| Checkpoint | `RoMaV2.__init__` calls `torch.hub.load_state_dict_from_url` for release `v2.0.1`, then `load_state_dict`. |

This project does not alter the public inference method. Any training integration must be a separate, tested gradient-enabled path that uses these modules and preserves the original state-dict naming. `scripts/inspect_romav2.py` writes a repeatable inspection report; its `--run` option is intentionally explicit because it downloads and initializes the official checkpoint.
