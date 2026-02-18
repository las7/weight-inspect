use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Artifact {
    pub format: Format,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gguf_version: Option<i64>,
    pub metadata: BTreeMap<String, CanonicalValue>,
    pub tensors: BTreeMap<String, Tensor>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Format {
    GGUF,
    Safetensors,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum CanonicalValue {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    String(String),
    Array(Vec<CanonicalValue>),
    Uint8(i64),
    Int8(i64),
    Uint16(i64),
    Int16(i64),
    Uint32(i64),
    Int32(i64),
    Uint64(i64),
    Int64(i64),
    Float32(f64),
}

impl fmt::Display for CanonicalValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CanonicalValue::Null => write!(f, "null"),
            CanonicalValue::Bool(b) => write!(f, "{}", b),
            CanonicalValue::Int(i) => write!(f, "{}", i),
            CanonicalValue::Float(fl) => write!(f, "{}", fl),
            CanonicalValue::String(s) => write!(f, "{}", s),
            CanonicalValue::Array(arr) => write!(f, "{:?}", arr),
            CanonicalValue::Uint8(i) => write!(f, "{}", i),
            CanonicalValue::Int8(i) => write!(f, "{}", i),
            CanonicalValue::Uint16(i) => write!(f, "{}", i),
            CanonicalValue::Int16(i) => write!(f, "{}", i),
            CanonicalValue::Uint32(i) => write!(f, "{}", i),
            CanonicalValue::Int32(i) => write!(f, "{}", i),
            CanonicalValue::Uint64(i) => write!(f, "{}", i),
            CanonicalValue::Int64(i) => write!(f, "{}", i),
            CanonicalValue::Float32(fl) => write!(f, "{}", fl),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Tensor {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<u64>,
    pub byte_length: u64,
}

impl Artifact {
    pub fn canonical_json(&self) -> String {
        let buf = Vec::new();
        let formatter = serde_json::ser::PrettyFormatter::with_indent(b"");
        let mut ser = serde_json::Serializer::with_formatter(buf, formatter);
        self.serialize(&mut ser).unwrap();
        String::from_utf8(ser.into_inner()).unwrap()
    }
}
