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

# JSON output
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

- Execute models
- Predict runtime compatibility
- Load tensor data (only reads headers/metadata)
- Security scanning

## Use cases

- "Why are these two model files different?"
- Verify model file integrity
- Compare quantization variants
- Detect structural changes in converted models

## License

MIT
