use crate::types::{Artifact, CanonicalValue, Format, Tensor};
use std::collections::BTreeMap;
use std::io::{Read, Seek};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum SafetensorsParserError {
    #[error("invalid safetensors JSON header")]
    InvalidHeader,
    #[error("header size exceeds maximum allowed ({max} bytes): {size} bytes")]
    HeaderTooLarge { size: usize, max: usize },
    #[error("invalid tensor length for '{name}': offset {offset} > {end}")]
    InvalidByteLength { name: String, offset: u64, end: u64 },
    #[error("missing required field '{field}' for tensor '{name}'")]
    MissingField { name: String, field: String },
    #[error("invalid shape dimension at index {index} for tensor '{name}': not a valid u64")]
    InvalidShape { name: String, index: usize },
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),
}

const MAX_HEADER_SIZE: usize = 100 * 1024 * 1024; // 100MB

/// Parse a safetensors model file.
///
/// # Example
///
/// ```
/// use weight_inspect::safetensors;
///
/// let data = std::fs::read("tests/fixtures/tiny.safetensors").unwrap();
/// let mut cursor = std::io::Cursor::new(data);
/// let artifact = safetensors::parse_safetensors(&mut cursor).unwrap();
/// assert_eq!(artifact.format, weight_inspect::types::Format::Safetensors);
/// ```
pub fn parse_safetensors<R: Read + Seek>(
    reader: &mut R,
) -> Result<Artifact, SafetensorsParserError> {
    let header_size = read_header_size(reader)?;
    let mut header_buf = vec![0u8; header_size];
    reader.read_exact(&mut header_buf)?;
    let header_str =
        String::from_utf8(header_buf).map_err(|_| SafetensorsParserError::InvalidHeader)?;

    let json: serde_json::Value =
        serde_json::from_str(&header_str).map_err(|_| SafetensorsParserError::InvalidHeader)?;

    let obj = json
        .as_object()
        .ok_or(SafetensorsParserError::InvalidHeader)?;

    let mut metadata = BTreeMap::new();
    let mut tensors = BTreeMap::new();

    for (key, value) in obj {
        if key == "__metadata__" {
            if let Some(meta_obj) = value.as_object() {
                for (mk, mv) in meta_obj {
                    let cv = match mv {
                        serde_json::Value::String(s) => CanonicalValue::String(s.clone()),
                        serde_json::Value::Number(n) => {
                            if let Some(i) = n.as_i64() {
                                CanonicalValue::Int(i)
                            } else if let Some(f) = n.as_f64() {
                                CanonicalValue::Float(f)
                            } else {
                                CanonicalValue::String(mv.to_string())
                            }
                        }
                        serde_json::Value::Bool(b) => CanonicalValue::Bool(*b),
                        serde_json::Value::Null => CanonicalValue::Null,
                        _ => CanonicalValue::String(mv.to_string()),
                    };
                    metadata.insert(mk.clone(), cv);
                }
            }
        } else if let Some(tensor_obj) = value.as_object() {
            let dtype = match tensor_obj.get("dtype").and_then(|v| v.as_str()) {
                Some(s) => s.to_lowercase(),
                None => {
                    return Err(SafetensorsParserError::MissingField {
                        name: key.clone(),
                        field: "dtype".to_string(),
                    })
                }
            };
            let shape: Vec<u64> = match tensor_obj.get("shape").and_then(|v| v.as_array()) {
                Some(arr) => {
                    let mut dims = Vec::with_capacity(arr.len());
                    for (i, v) in arr.iter().enumerate() {
                        let dim =
                            v.as_u64()
                                .ok_or_else(|| SafetensorsParserError::InvalidShape {
                                    name: key.clone(),
                                    index: i,
                                })?;
                        dims.push(dim);
                    }
                    dims
                }
                None => {
                    return Err(SafetensorsParserError::MissingField {
                        name: key.clone(),
                        field: "shape".to_string(),
                    })
                }
            };
            let data_offsets =
                match tensor_obj.get("data_offsets").and_then(|v| v.as_array()) {
                    Some(arr) if arr.len() >= 2 => {
                        let a = arr[0].as_u64().ok_or_else(|| {
                            SafetensorsParserError::MissingField {
                                name: key.clone(),
                                field: "data_offsets[0]".to_string(),
                            }
                        })?;
                        let b = arr[1].as_u64().ok_or_else(|| {
                            SafetensorsParserError::MissingField {
                                name: key.clone(),
                                field: "data_offsets[1]".to_string(),
                            }
                        })?;
                        [a, b]
                    }
                    _ => {
                        return Err(SafetensorsParserError::MissingField {
                            name: key.clone(),
                            field: "data_offsets".to_string(),
                        })
                    }
                };
            let offset = data_offsets[0];
            let end = data_offsets[1];
            if end <= offset {
                return Err(SafetensorsParserError::InvalidByteLength {
                    name: key.clone(),
                    offset,
                    end,
                });
            }
            let byte_length = end - offset;

            tensors.insert(
                key.clone(),
                Tensor {
                    name: key.clone(),
                    dtype,
                    shape,
                    byte_length,
                },
            );
        }
    }

    Ok(Artifact {
        format: Format::Safetensors,
        gguf_version: None,
        metadata,
        tensors,
    })
}

fn read_header_size<R: Read + Seek>(reader: &mut R) -> Result<usize, SafetensorsParserError> {
    let mut buf = [0u8; 8];
    reader.read_exact(&mut buf)?;
    let size = u64::from_le_bytes(buf) as usize;
    if size > MAX_HEADER_SIZE {
        return Err(SafetensorsParserError::HeaderTooLarge {
            size,
            max: MAX_HEADER_SIZE,
        });
    }
    Ok(size)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn make_safetensors(header_json: &str) -> Vec<u8> {
        let header_bytes = header_json.as_bytes();
        let header_len = header_bytes.len() as u64;
        let padded_len = ((header_len as usize) + 7) / 8 * 8;

        let mut data = Vec::new();
        data.extend_from_slice(&header_len.to_le_bytes());
        data.extend_from_slice(header_bytes);
        data.extend(vec![0u8; padded_len - header_bytes.len()]);
        data.extend(vec![0u8; 100]); // minimal tensor data
        data
    }

    #[test]
    fn test_parse_valid_safetensors() {
        let header = r#"{"test.weight":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]}}"#;
        let data = make_safetensors(header);
        let mut cursor = Cursor::new(data);

        let artifact = parse_safetensors(&mut cursor).unwrap();

        assert_eq!(artifact.format, Format::Safetensors);
        assert!(artifact.tensors.contains_key("test.weight"));
        let tensor = &artifact.tensors["test.weight"];
        assert_eq!(tensor.dtype, "f32");
        assert_eq!(tensor.shape, vec![2, 3]);
        assert_eq!(tensor.byte_length, 24);
    }

    #[test]
    fn test_parse_with_metadata() {
        let header = r#"{"__metadata__":{"format_version":"1"},"test.weight":{"dtype":"F16","shape":[4],"data_offsets":[0,8]}}"#;
        let data = make_safetensors(header);
        let mut cursor = Cursor::new(data);

        let artifact = parse_safetensors(&mut cursor).unwrap();

        assert!(artifact.metadata.contains_key("format_version"));
        assert_eq!(
            artifact.metadata["format_version"],
            CanonicalValue::String("1".to_string())
        );
    }

    #[test]
    fn test_missing_dtype_field() {
        let header = r#"{"test.weight":{"shape":[2,3],"data_offsets":[0,24]}}"#;
        let data = make_safetensors(header);
        let mut cursor = Cursor::new(data);

        let result = parse_safetensors(&mut cursor);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, SafetensorsParserError::MissingField { field, .. } if field == "dtype")
        );
    }

    #[test]
    fn test_missing_shape_field() {
        let header = r#"{"test.weight":{"dtype":"F32","data_offsets":[0,24]}}"#;
        let data = make_safetensors(header);
        let mut cursor = Cursor::new(data);

        let result = parse_safetensors(&mut cursor);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, SafetensorsParserError::MissingField { field, .. } if field == "shape")
        );
    }

    #[test]
    fn test_missing_data_offsets_field() {
        let header = r#"{"test.weight":{"dtype":"F32","shape":[2,3]}}"#;
        let data = make_safetensors(header);
        let mut cursor = Cursor::new(data);

        let result = parse_safetensors(&mut cursor);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, SafetensorsParserError::MissingField { field, .. } if field == "data_offsets")
        );
    }

    #[test]
    fn test_invalid_shape_dimension() {
        let header =
            r#"{"test.weight":{"dtype":"F32","shape":["not","a","number"],"data_offsets":[0,24]}}"#;
        let data = make_safetensors(header);
        let mut cursor = Cursor::new(data);

        let result = parse_safetensors(&mut cursor);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            SafetensorsParserError::InvalidShape { .. }
        ));
    }

    #[test]
    fn test_invalid_byte_length() {
        let header = r#"{"test.weight":{"dtype":"F32","shape":[2,3],"data_offsets":[10,5]}}"#;
        let data = make_safetensors(header);
        let mut cursor = Cursor::new(data);

        let result = parse_safetensors(&mut cursor);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            SafetensorsParserError::InvalidByteLength { .. }
        ));
    }

    #[test]
    fn test_invalid_json_header() {
        let header_json = b"not valid json";
        let header_len = header_json.len() as u64;

        let mut data = Vec::new();
        data.extend_from_slice(&header_len.to_le_bytes());
        data.extend_from_slice(header_json);
        // No padding needed for invalid JSON test

        let mut cursor = Cursor::new(data);
        let result = parse_safetensors(&mut cursor);
        assert!(result.is_err());
    }

    #[test]
    fn test_header_too_large() {
        let mut data = vec![0u8; 8];
        let too_big = (MAX_HEADER_SIZE + 1) as u64;
        data[0..8].copy_from_slice(&too_big.to_le_bytes());
        let mut cursor = Cursor::new(data);

        let result = parse_safetensors(&mut cursor);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            SafetensorsParserError::HeaderTooLarge { .. }
        ));
    }

    #[test]
    fn test_multiple_tensors() {
        let header = r#"{
            "tensor1": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},
            "tensor2": {"dtype": "I64", "shape": [3], "data_offsets": [8, 32]}
        }"#;
        let data = make_safetensors(header);
        let mut cursor = Cursor::new(data);

        let artifact = parse_safetensors(&mut cursor).unwrap();

        assert_eq!(artifact.tensors.len(), 2);
        assert!(artifact.tensors.contains_key("tensor1"));
        assert!(artifact.tensors.contains_key("tensor2"));
    }

    #[test]
    fn test_dtype_normalization() {
        let header = r#"{"test.weight":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        let data = make_safetensors(header);
        let mut cursor = Cursor::new(data);

        let artifact = parse_safetensors(&mut cursor).unwrap();

        assert_eq!(artifact.tensors["test.weight"].dtype, "f32");
    }
}
