# Training

Training remains blocked until coordinate and GT tests, loss tests, gradient checks, a passing one-batch overfit, and an immutable held-out T0 are complete. The current code has the separate gradient-enabled forward and passes the gradient smoke check, but the available NAC crop fails the one-batch PCK/EPE gate and is not a substitute for LunarMatch-NASA. See `docs/ROMAV2_INTERNALS.md` for source-derived integration details.
