"""Verify the exact binary forward pass against Python's hashlib."""

from hashlib import sha256

from bbtsha import DifferentiableSHA256, bits_to_bytes, pad_single_block


def main() -> None:
    message = "Hello"
    block, _ = pad_single_block(message)
    actual = bits_to_bytes(DifferentiableSHA256().hash_block(block))
    expected = sha256(message.encode("utf-8")).digest()
    print(f"message: {message!r}")
    print(f"digest:  {actual.hex()}")
    assert actual == expected, "Differentiable forward pass did not match hashlib"
    print("matches hashlib.sha256")


if __name__ == "__main__":
    main()
