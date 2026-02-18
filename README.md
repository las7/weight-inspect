# weight-inspect

Deterministic structural identity for GGUF and safetensors ML model files.

## The Problem

Model files are opaque blobs. When you download or convert a model, you can't easily answer:

- "What model family/architecture is this?"
- "What quantization is used?"
- "What are the tensor shapes?"
- "Is this the same model as before?"

## The Solution

weight-inspect provides **structural identity** - a canonical, deterministic representation of what a model file actually contains.

## Quick Start

```bash
# Get the structural identity (the core operation)
weight-inspect id model.gguf

# Inspect full details
weight-inspect inspect model.gguf

# Compare two models
weight-inspect diff a.gguf b.gguf

# One-line summary (for GitHub issues)
weight-inspect summary model.gguf
```

## Installation

```bash
cargo install weight-inspect
```

Or build from source:

```bash
cargo build --release
```

## Usage

### Identity (core feature)

```bash
weight-inspect id model.gguf

# Output:
# format: gguf
# structural_hash: abc123...
# tensor_count: 242
# metadata_count: 22
```

### One-line summary

```bash
weight-inspect summary model.gguf

# Output: gguf,3,242,22,abc123...
```

### Inspect (full details)

```bash
weight-inspect inspect model.gguf
weight-inspect inspect model.gguf --json
```

### Compare

```bash
weight-inspect diff a.gguf b.gguf
weight-inspect diff a.gguf b.gguf --json
weight-inspect diff a.gguf b.gguf --quiet
```

## What it does

- Parses GGUF and safetensors headers
- Extracts metadata (hyperparameters, tokenizer config, etc.)
- Lists tensor names, dtypes, and shapes
- Computes structural hash (deterministic JSON → SHA256)
- Compares two files showing structural differences

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

- "What model is this?" - Get instant structural identity
- "Is this the same model?" - Compare structural hashes
- "Why are these two models different?" - See exact differences
- Verify model file integrity
- Compare quantization variants
- Detect structural changes in converted models

## License

MIT
