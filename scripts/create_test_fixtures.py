#!/usr/bin/env python3
import struct
import json
import os


def create_empty_gguf(path):
    """Create a minimal GGUF file with zero tensors for testing."""
    with open(path, "wb") as f:
        f.write(b"GGUF")
        f.write(struct.pack("<I", 3))
        f.write(struct.pack("<Q", 0))
        f.write(struct.pack("<Q", 1))

        f.write(struct.pack("<Q", 20))
        f.write(b"general.architecture")
        f.write(struct.pack("<I", 8))
        f.write(struct.pack("<Q", 5))
        f.write(b"llama")


def create_minimal_gguf(path):
    """Create a minimal valid GGUF file with 1 tensor for testing."""
    with open(path, "wb") as f:
        f.write(b"GGUF")
        f.write(struct.pack("<I", 3))
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<Q", 1))

        f.write(struct.pack("<Q", 20))
        f.write(b"general.architecture")
        f.write(struct.pack("<I", 8))
        f.write(struct.pack("<Q", 5))
        f.write(b"llama")

        name = b"test.weight.0"
        f.write(struct.pack("<Q", len(name)))
        f.write(name)
        f.write(struct.pack("<I", 2))
        f.write(struct.pack("<Q", 64))
        f.write(struct.pack("<Q", 128))
        f.write(struct.pack("<I", 0))
        f.write(struct.pack("<Q", 0))


def create_minimal_safetensors(path):
    tensors = {
        "test.weight.0": {
            "dtype": "F32",
            "shape": [64, 128],
            "data_offsets": [0, 32768],
        }
    }

    header = tensors
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    header_len = len(header_bytes)
    padded_len = (header_len + 7) // 8 * 8

    with open(path, "wb") as f:
        f.write(struct.pack("<Q", header_len))
        f.write(header_bytes)
        padding = padded_len - header_len
        f.write(b"\x00" * padding)
        f.write(b"\x00" * 32768)


os.makedirs("tests/fixtures", exist_ok=True)
create_empty_gguf("tests/fixtures/empty.gguf")
create_minimal_gguf("tests/fixtures/tiny.gguf")
create_minimal_safetensors("tests/fixtures/tiny.safetensors")
print("Created test fixtures")

for name in ["empty.gguf", "tiny.gguf", "tiny.safetensors"]:
    path = f"tests/fixtures/{name}"
    size = os.path.getsize(path)
    print(f"{name}: {size} bytes")
