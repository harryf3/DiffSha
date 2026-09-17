# Differentiable relaxation variants

These are preserved research variants of the canonical tensor implementation in
`src/bbtsha/`. They are intentionally separate from the supported package: they
explore different continuous representations for SHA-256-style Boolean and
arithmetic operations, but are not validated as exact SHA-256 implementations.

| Module | Representation explored | Distinguishing idea |
| --- | --- | --- |
| `probability_coercion.py` | Probability-like bit values | Applies a sigmoid-like coercion after `AND` and `XOR`. |
| `probability_coercion_variant.py` | Probability-like bit values | Tests input coercion and an absolute-value `NOT` alternative. |
| `log_probability.py` | Log/probability-inspired values | Experiments with alternative arithmetic and an exponential carry path. |
| `logit_relaxation.py` | Logit-valued bits | Uses logit-space `NOT`, an approximate `AND`, and a sigmoid output. |

The code is retained to show the design search behind the project. Use the
package in `src/bbtsha/` for the exact binary single-block forward pass.
