# safetensors Format Guide

safetensors is a fast, safe, memory-mapped format for storing PyTorch tensors. This guide explains the structure for understanding model files.

## File Structure

```
┌─────────────────────────────────────┐
│ Header (JSON + padding)             │
├─────────────────────────────────────┤
│ Tensor Data (binary)               │
└─────────────────────────────────────┘
```

## Header

The header is a JSON object that describes all tensors in the file. It is prefixed with an 8-byte little-endian integer indicating the header size.

### Header Structure

| Field | Type | Description |
|-------|------|-------------|
| header_size | u64 | Size of JSON header in bytes |
| header | bytes | JSON object describing tensors and metadata |
| tensor_data | bytes | Binary tensor data |

### Tensor Descriptor

Each tensor is represented by a key in the JSON header:

```json
{
  "tensor_name": {
    "dtype": "F32",
    "shape": [1024, 4096],
    "data_offsets": [0, 16777216]
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| dtype | string | Data type (e.g., "F32", "F16", "I64") |
| shape | array | Tensor dimensions |
| data_offsets | array | [start, end] byte offsets in tensor data section |

### Metadata

Optional metadata is stored in `__metadata__` key:

```json
{
  "__metadata__": {
    "format_version": "1",
    "model_name": "Llama-2-7b"
  }
}
```

## Data Types

| Type ID | Name | Description |
|---------|------|-------------|
| F32 | Float32 | 32-bit float |
| F16 | Float16 | 16-bit float |
| BF16 | BFloat16 | Brain float16 |
| I8 | Int8 | 8-bit signed integer |
| I16 | Int16 | 16-bit signed integer |
| I32 | Int32 | 32-bit signed integer |
| I64 | Int64 | 64-bit signed integer |
| U8 | UInt8 | 8-bit unsigned integer |
| U16 | UInt16 | 16-bit unsigned integer |
| U32 | UInt32 | 32-bit unsigned integer |
| U64 | UInt64 | 64-bit unsigned integer |
| F64 | Float64 | 64-bit float |
| BOOL | Bool | Boolean |

### Quantized Types

| Type ID | Name | Description |
|---------|------|-------------|
| I4 | Int4 | 4-bit signed integer (packed) |
| I2 | Int2 | 2-bit signed integer (packed) |

## What weight-inspect Reads

weight-inspect only reads:

1. **Header** - JSON tensor descriptors
2. **Metadata** - From `__metadata__` key
3. **Tensor Info** - Names, shapes, dtypes, byte lengths

It does NOT read:
- Tensor data/weights
- Data offsets (beyond computing byte_length)

This is intentional for fast, safe structural analysis.

## Example: Inspecting a safetensors File

```bash
weight-inspect inspect model.safetensors
```

Output:
```
safetensors
───
Version:  N/A
Tensors:  291
Metadata: 5

Structural ID
────────────
Hash: abc123...
```

## Structural Hash Contents

For safetensors files, the structural hash includes:

| Field | Included | Notes |
|-------|----------|-------|
| format | ✓ | "safetensors" |
| tensor names | ✓ | Sorted lexicographically |
| tensor dtype | ✓ | Normalized to lowercase |
| tensor shape | ✓ | Dimension order preserved |
| tensor byte_length | ✓ | Computed from data_offsets |

## Diff Behavior

For safetensors files, `weight-inspect diff` compares:

| Comparison | Description |
|------------|-------------|
| Tensors | Names, shapes, dtypes, byte lengths |
| Metadata | All metadata key-value pairs |

## References

- [safetensors Specification](https://huggingface.co/docs/safetensors)
- [safetensors GitHub](https://github.com/huggingface/safetensors)
