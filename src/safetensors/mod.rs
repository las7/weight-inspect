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
                Some(s) => s.to_string(),
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
    #[test]
    fn test_dtype_str() {
        assert_eq!(safetensors_dtype_str("F32"), "f32");
        assert_eq!(safetensors_dtype_str("F16"), "f16");
        assert_eq!(safetensors_dtype_str("I64"), "i64");
    }

    fn safetensors_dtype_str(dtype: &str) -> String {
        match dtype.to_uppercase().as_str() {
            "F32" => "f32".to_string(),
            "F16" => "f16".to_string(),
            "BF16" => "bf16".to_string(),
            "U8" => "u8".to_string(),
            "I8" => "i8".to_string(),
            "I64" => "i64".to_string(),
            "I32" => "i32".to_string(),
            "I16" => "i16".to_string(),
            "BOOL" => "bool".to_string(),
            "F8E5M2" => "f8e5m2".to_string(),
            "F8E4M3" => "f8e4m3".to_string(),
            _ => format!("unknown_{}", dtype),
        }
    }
}
