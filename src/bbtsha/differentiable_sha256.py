"""A differentiable, single-block SHA-256 computation in PyTorch.

Binary inputs reproduce the SHA-256 compression path exactly. Fractional inputs
are useful for autograd experiments, but are a relaxation rather than standard
SHA-256 semantics.
"""

from __future__ import annotations

from typing import Final

import torch
from torch import Tensor


BLOCK_BITS: Final = 512
MAX_MESSAGE_BYTES: Final = 55


def bytes_to_bits(byte_data: bytes) -> Tensor:
    """Convert bytes into a most-significant-bit-first float tensor."""
    return torch.tensor(
        [float(bit) for byte in byte_data for bit in f"{byte:08b}"],
        dtype=torch.float64,
    )


def bits_to_bytes(bits: Tensor) -> bytes:
    """Convert a one-dimensional binary tensor into bytes."""
    values = bits.to(torch.int64).tolist()
    if len(values) % 8:
        raise ValueError("Bit tensor length must be divisible by 8.")
    return bytes(
        int("".join(map(str, values[index : index + 8])), 2)
        for index in range(0, len(values), 8)
    )


def pad_single_block(message: str) -> tuple[Tensor, int]:
    """Return SHA-256 padding for a UTF-8 message that fits one block.

    The returned tensor contains 512 float64 bits. Enable ``requires_grad`` on
    it to use the continuous relaxation in an autograd experiment.
    """
    if not isinstance(message, str):
        raise TypeError("message must be a string")
    encoded = message.encode("utf-8")
    if len(encoded) > MAX_MESSAGE_BYTES:
        raise ValueError(
            f"Single-block SHA-256 supports at most {MAX_MESSAGE_BYTES} UTF-8 bytes."
        )

    message_bits = bytes_to_bits(encoded)
    original_length = message_bits.numel()
    length_bits = bytes_to_bits(original_length.to_bytes(8, "big"))
    padded = torch.cat((
        message_bits,
        torch.ones(1, dtype=torch.float64),
        torch.zeros(447 - original_length, dtype=torch.float64),
        length_bits,
    ))
    return padded, original_length


def _right_rotate(bits: Tensor, amount: int) -> Tensor:
    return torch.roll(bits, amount)


def _bit_not(bits: Tensor) -> Tensor:
    return 1.0 - bits


def _bit_and(left: Tensor, right: Tensor) -> Tensor:
    return left * right


def _right_shift(bits: Tensor, amount: int) -> Tensor:
    amount = min(amount, len(bits))
    if amount == 0:
        return bits
    return torch.cat((torch.zeros(amount, dtype=bits.dtype, device=bits.device), bits[:-amount]))


def _left_shift(bits: Tensor, amount: int) -> Tensor:
    amount = min(amount, len(bits))
    if amount == 0:
        return bits
    return torch.cat((bits[amount:], torch.zeros(amount, dtype=bits.dtype, device=bits.device)))


def _xor(left: Tensor, right: Tensor) -> Tensor:
    return left + right - 2 * _bit_and(left, right)


def _bitwise_add(left: Tensor, right: Tensor) -> Tensor:
    """Add two 32-bit words using a 32-step ripple-carry circuit."""
    for _ in range(32):
        carry = _left_shift(_bit_and(left, right), 1)
        left, right = _xor(left, right), carry
    return left


def _bitwise_adds(words: list[Tensor]) -> Tensor:
    total = words[0]
    for word in words[1:]:
        total = _bitwise_add(total, word)
    return total


def _sigma0(bits: Tensor) -> Tensor:
    return _xor(_xor(_right_rotate(bits, 7), _right_rotate(bits, 18)), _right_shift(bits, 3))


def _sigma1(bits: Tensor) -> Tensor:
    return _xor(_xor(_right_rotate(bits, 17), _right_rotate(bits, 19)), _right_shift(bits, 10))


def _capsigma0(bits: Tensor) -> Tensor:
    return _xor(_xor(_right_rotate(bits, 2), _right_rotate(bits, 13)), _right_rotate(bits, 22))


def _capsigma1(bits: Tensor) -> Tensor:
    return _xor(_xor(_right_rotate(bits, 6), _right_rotate(bits, 11)), _right_rotate(bits, 25))


def _choose(x: Tensor, y: Tensor, z: Tensor) -> Tensor:
    return _xor(_bit_and(x, y), _bit_and(_bit_not(x), z))


def _majority(x: Tensor, y: Tensor, z: Tensor) -> Tensor:
    return _xor(_xor(_bit_and(x, y), _bit_and(x, z)), _bit_and(y, z))


class DifferentiableSHA256:
    """Evaluate SHA-256's 64-round compression path on one 512-bit tensor."""

    _ROUND_CONSTANTS: Final = (
        0x428A2F98, 0x71374491, 0xB5C0FBCF, 0xE9B5DBA5, 0x3956C25B, 0x59F111F1, 0x923F82A4, 0xAB1C5ED5,
        0xD807AA98, 0x12835B01, 0x243185BE, 0x550C7DC3, 0x72BE5D74, 0x80DEB1FE, 0x9BDC06A7, 0xC19BF174,
        0xE49B69C1, 0xEFBE4786, 0x0FC19DC6, 0x240CA1CC, 0x2DE92C6F, 0x4A7484AA, 0x5CB0A9DC, 0x76F988DA,
        0x983E5152, 0xA831C66D, 0xB00327C8, 0xBF597FC7, 0xC6E00BF3, 0xD5A79147, 0x06CA6351, 0x14292967,
        0x27B70A85, 0x2E1B2138, 0x4D2C6DFC, 0x53380D13, 0x650A7354, 0x766A0ABB, 0x81C2C92E, 0x92722C85,
        0xA2BFE8A1, 0xA81A664B, 0xC24B8B70, 0xC76C51A3, 0xD192E819, 0xD6990624, 0xF40E3585, 0x106AA070,
        0x19A4C116, 0x1E376C08, 0x2748774C, 0x34B0BCB5, 0x391C0CB3, 0x4ED8AA4A, 0x5B9CCA4F, 0x682E6FF3,
        0x748F82EE, 0x78A5636F, 0x84C87814, 0x8CC70208, 0x90BEFFFA, 0xA4506CEB, 0xBEF9A3F7, 0xC67178F2,
    )
    _INITIAL_HASH: Final = (
        0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
        0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
    )

    def __init__(self) -> None:
        self._constants = [bytes_to_bits(value.to_bytes(4, "big")) for value in self._ROUND_CONSTANTS]

    def hash_block(self, block: Tensor, rounds: int = 64) -> Tensor:
        """Hash a padded 512-bit block and return its 256-bit output tensor."""
        if block.ndim != 1 or block.numel() != BLOCK_BITS:
            raise ValueError("block must be a one-dimensional tensor with 512 bits")
        if not 0 <= rounds <= 64:
            raise ValueError("rounds must be between 0 and 64")

        constants = [constant.to(dtype=block.dtype, device=block.device) for constant in self._constants]
        state = [
            bytes_to_bits(value.to_bytes(4, "big")).to(dtype=block.dtype, device=block.device)
            for value in self._INITIAL_HASH
        ]
        schedule = [block[index * 32 : (index + 1) * 32] for index in range(16)]
        for index in range(16, 64):
            schedule.append(_bitwise_adds([
                _sigma1(schedule[index - 2]), schedule[index - 7],
                _sigma0(schedule[index - 15]), schedule[index - 16],
            ]))

        a, b, c, d, e, f, g, h = state
        for index in range(rounds):
            first = _bitwise_adds([h, _capsigma1(e), _choose(e, f, g), constants[index], schedule[index]])
            second = _bitwise_adds([_capsigma0(a), _majority(a, b, c)])
            h, g, f, e, d, c, b, a = g, f, e, _bitwise_add(d, first), c, b, a, _bitwise_add(first, second)

        return torch.cat([_bitwise_add(original, updated) for original, updated in zip(state, (a, b, c, d, e, f, g, h))])

    def __call__(self, block: Tensor, rounds: int = 64) -> Tensor:
        return self.hash_block(block, rounds)
