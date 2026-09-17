"""Regression tests for the single-block tensor implementation."""

from hashlib import sha256
import unittest

import torch

from bbtsha import DifferentiableSHA256, bits_to_bytes, pad_single_block


class DifferentiableSHA256Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.hasher = DifferentiableSHA256()

    def test_matches_hashlib_for_single_block_messages(self) -> None:
        for message in ("", "Hello", "a" * 55):
            with self.subTest(message=message):
                block, _ = pad_single_block(message)
                actual = bits_to_bytes(self.hasher.hash_block(block))
                self.assertEqual(actual, sha256(message.encode("utf-8")).digest())

    def test_padding_boundary(self) -> None:
        block, length = pad_single_block("a" * 55)
        self.assertEqual(block.shape, (512,))
        self.assertEqual(length, 440)
        with self.assertRaises(ValueError):
            pad_single_block("a" * 56)

    def test_relaxed_input_retains_a_gradient(self) -> None:
        block, _ = pad_single_block("Hello")
        block.requires_grad_()
        loss = self.hasher.hash_block(block).sum()
        loss.backward()
        self.assertIsNotNone(block.grad)
        self.assertTrue(torch.isfinite(block.grad).all())


if __name__ == "__main__":
    unittest.main()
