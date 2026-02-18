# modeldiff

Deterministic structural diff engine for GGUF and safetensors ML model files.

## Features

- **Parse** GGUF and safetensors files
- **Hash** - Compute structural identity hash (SHA256 of canonical JSON)
- **Diff** - Compare two model files structurally

## Installation

```bash
cargo install modeldiff
```

Or build from source:

```bash
cargo build --release
```

## Usage

```bash
# Inspect a model file
modeldiff inspect model.gguf

# Get structural identity
modeldiff id model.gguf

# Diff two files
modeldiff diff a.gguf b.gguf

# JSON output (for tooling/integration)
modeldiff diff a.gguf b.gguf --json

# Quiet mode (exit 0 if identical, 1 if different)
modeldiff diff a.gguf b.gguf --quiet
```

## What it does

- Parses GGUF and safetensors headers
- Extracts metadata (hyperparameters, tokenizer config, etc.)
- Lists tensor names, dtypes, and shapes
- Computes structural hash (deterministic JSON → SHA256)
- Compares two files showing:
  - Added/removed/changed metadata keys
  - Added/removed/changed tensors
  - Tensor dtype/shape/byte_length changes

## What it does NOT do

- **Never loads weight data** - Only reads headers and tensor descriptors
- Execute models
- Predict runtime compatibility
- Security scanning

### Why no weight data?

We intentionally only read headers/metadata because:

- **Fast**: Works on huge models without loading gigabytes
- **Memory safe**: No memory spikes on large files
- **Cross-platform**: Deterministic across any machine
- **Structural focus**: Answer "are these the same model?" not "are these identical files?"

If you need content equality (byte-for-byte), that's a separate mode that would require loading all weights.

## Design Principles

1. **Deterministic**: Same input → same output across machines
2. **Canonical**: Floats serialized as bit patterns, maps sorted
3. **No heuristics**: Pure structural comparison
4. **Fail fast**: Invalid files produce clear errors

See [SPEC.md](SPEC.md) for full specification.

## Use cases

- "Why are these two model files different?"
- Verify model file integrity
- Compare quantization variants
- Detect structural changes in converted models

## License

MIT
