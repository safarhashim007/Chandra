# Training

Training remains blocked until coordinate and GT tests, loss tests, gradient checks, a passing validation quality gate, and an immutable held-out T0 are complete. The current code has the separate gradient-enabled forward and passes the gradient smoke check. On the available NAC crop, normalized Huber beta=1.0 collapses PCK while reducing loss; beta=0.01 avoids collapse but still fails the non-regression gate. The smoke split has no validation region and is not a substitute for LunarMatch-NASA. See `docs/ROMAV2_INTERNALS.md` and `docs/DATASET_EVALUATION.md` for details.
