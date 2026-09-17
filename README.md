# Differentiable SHA-256 in PyTorch

A tensor-level implementation of single-block SHA-256 in PyTorch, built
entirely from differentiable tensor operations so PyTorch autograd can trace
through the computation.

On binary inputs, it exactly reproduces standard SHA-256 for supported
single-block messages. On continuous inputs, the Boolean operations extend to
differentiable functions, allowing gradients to propagate through the resulting
computation graph. The project is a compact demonstration of low-level PyTorch
work: representing bits as tensors and rebuilding the hash's Boolean logic,
word arithmetic, message schedule, and compression rounds from those pieces.

## Why?

SHA-256 is normally described with discrete bit operations that are
incompatible with ordinary gradient-based optimization. Reconstructing those
operations at the tensor level provides a small experiment in what happens when
a fundamentally discrete computation is embedded in a differentiable framework.

## What is implemented

- Big-endian byte-to-bit and bit-to-byte tensor conversion
- SHA-256 single-block padding for UTF-8 messages up to 55 bytes
- Tensor versions of `AND`, `XOR`, `NOT`, shifts, rotations, `Ch`, and `Maj`
- 32-bit ripple-carry addition built from those tensor primitives
- The SHA-256 message schedule and 64-round compression path
- A continuous input path that PyTorch can differentiate through

## Quick start

Install PyTorch for your platform, then install this project:

```bash
python -m pip install -e .
python examples/verify_against_hashlib.py
python -m unittest discover -s tests
```

The verification script prints the digest for `"Hello"` and asserts that it
matches Python's standard-library SHA-256 implementation.

```python
from bbtsha import DifferentiableSHA256, bits_to_bytes, pad_single_block

block, _ = pad_single_block("Hello")
digest_bits = DifferentiableSHA256().hash_block(block)
print(bits_to_bytes(digest_bits).hex())
```

## Autograd in practice

The binary exactness check is only half of the experiment. A padded block can
also be made into a differentiable input tensor:

```python
import torch
from bbtsha import DifferentiableSHA256, pad_single_block

block, message_bits = pad_single_block("Hello")
block.requires_grad_()

objective = DifferentiableSHA256().hash_block(block).sum()
objective.backward()

print(block.grad[:8])
assert torch.isfinite(block.grad).all()
```

On the current implementation, that produces very large—but finite—raw
gradients, which is itself a useful illustration of the numerical behavior of
this deeply composed discrete relaxation:

```text
tensor([-1.8351e+176,  7.2111e+176, -1.7978e+177, -2.4679e+177,
         8.6164e+176, -3.6346e+177, -5.1333e+177, -7.2439e+177],
       dtype=torch.float64)
```

These values demonstrate that autograd reaches the input; they are not evidence
of a practical gradient-based attack or an optimized numerical method.

## Project layout

- `src/bbtsha/` — supported differentiable SHA-256 implementation
- `examples/` — small exactness verification script
- `tests/` — regression tests for exact binary behavior, padding, and autograd
- `notebooks/` — a cleaned walkthrough of the differentiability experiment
- `experiments/` — preserved alternate formulations and exploratory notebooks

## Scope and boundaries

This is educational and experimental code, not a production cryptographic
library. It currently handles exactly one padded 512-bit SHA-256 block; it is
not a multi-block implementation. The exactness claim applies to binary tensor
inputs. Fractional inputs intentionally use a differentiable relaxation and do
not have standard SHA-256 semantics.

This repository does not claim a collision attack, hash inversion method, or
security result. The interesting part is the implementation exercise itself:
expressing the mechanics of a familiar cryptographic primitive directly as a
PyTorch computation graph and studying what gradients through that graph look
like.

## License

[MIT](LICENSE)
