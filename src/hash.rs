use crate::types::Artifact;
use sha2::{Digest, Sha256};

/// Compute a deterministic structural hash for an artifact.
///
/// The hash is based on the canonical JSON representation of the artifact,
/// making it independent of file layout and ordering.
///
/// # Example
///
/// ```
/// use weight_inspect::{gguf, hash};
///
/// let data = std::fs::read("tests/fixtures/tiny.gguf").unwrap();
/// let mut cursor = std::io::Cursor::new(data);
/// let artifact = gguf::parse_gguf(&mut cursor).unwrap();
/// let hash = hash::compute_structural_hash(&artifact).unwrap();
/// println!("Hash: {}", hash);
/// assert!(hash.len() == 64); // SHA256 hex = 64 chars
/// ```
pub fn compute_structural_hash(artifact: &Artifact) -> Result<String, serde_json::Error> {
    let canonical = serde_json::to_string(artifact)?;
    let mut hasher = Sha256::new();
    hasher.update(canonical.as_bytes());
    let result = hasher.finalize();
    Ok(hex::encode(result))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Format, Tensor};
    use std::collections::BTreeMap;

    #[test]
    fn test_hash_determinism() {
        let mut artifact1 = Artifact {
            format: Format::GGUF,
            gguf_version: Some(3),
            metadata: BTreeMap::new(),
            tensors: BTreeMap::new(),
        };
        artifact1.metadata.insert(
            "test".to_string(),
            crate::types::CanonicalValue::String("value".to_string()),
        );

        let hash1 = compute_structural_hash(&artifact1).unwrap();

        let mut artifact2 = Artifact {
            format: Format::GGUF,
            gguf_version: Some(3),
            metadata: BTreeMap::new(),
            tensors: BTreeMap::new(),
        };
        artifact2.metadata.insert(
            "test".to_string(),
            crate::types::CanonicalValue::String("value".to_string()),
        );

        let hash2 = compute_structural_hash(&artifact2).unwrap();

        assert_eq!(hash1, hash2, "same artifact should produce same hash");
    }

    #[test]
    fn test_hash_different_artifacts_different_hashes() {
        let mut artifact1 = Artifact {
            format: Format::GGUF,
            gguf_version: Some(3),
            metadata: BTreeMap::new(),
            tensors: BTreeMap::new(),
        };
        artifact1.metadata.insert(
            "test".to_string(),
            crate::types::CanonicalValue::String("value1".to_string()),
        );

        let mut artifact2 = Artifact {
            format: Format::GGUF,
            gguf_version: Some(3),
            metadata: BTreeMap::new(),
            tensors: BTreeMap::new(),
        };
        artifact2.metadata.insert(
            "test".to_string(),
            crate::types::CanonicalValue::String("value2".to_string()),
        );

        let hash1 = compute_structural_hash(&artifact1).unwrap();
        let hash2 = compute_structural_hash(&artifact2).unwrap();

        assert_ne!(
            hash1, hash2,
            "different artifacts should produce different hashes"
        );
    }

    #[test]
    fn test_hash_format_affects_hash() {
        let artifact1 = Artifact {
            format: Format::GGUF,
            gguf_version: Some(3),
            metadata: BTreeMap::new(),
            tensors: BTreeMap::new(),
        };

        let artifact2 = Artifact {
            format: Format::Safetensors,
            gguf_version: None,
            metadata: BTreeMap::new(),
            tensors: BTreeMap::new(),
        };

        let hash1 = compute_structural_hash(&artifact1).unwrap();
        let hash2 = compute_structural_hash(&artifact2).unwrap();

        assert_ne!(
            hash1, hash2,
            "different formats should produce different hashes"
        );
    }

    #[test]
    fn test_hash_tensor_count_affects_hash() {
        let mut artifact1 = Artifact {
            format: Format::GGUF,
            gguf_version: Some(3),
            metadata: BTreeMap::new(),
            tensors: BTreeMap::new(),
        };
        artifact1.tensors.insert(
            "tensor1".to_string(),
            Tensor {
                name: "tensor1".to_string(),
                dtype: "f32".to_string(),
                shape: vec![10],
                byte_length: 40,
            },
        );

        let mut artifact2 = Artifact {
            format: Format::GGUF,
            gguf_version: Some(3),
            metadata: BTreeMap::new(),
            tensors: BTreeMap::new(),
        };
        artifact2.tensors.insert(
            "tensor1".to_string(),
            Tensor {
                name: "tensor1".to_string(),
                dtype: "f32".to_string(),
                shape: vec![10],
                byte_length: 40,
            },
        );
        artifact2.tensors.insert(
            "tensor2".to_string(),
            Tensor {
                name: "tensor2".to_string(),
                dtype: "f32".to_string(),
                shape: vec![10],
                byte_length: 40,
            },
        );

        let hash1 = compute_structural_hash(&artifact1).unwrap();
        let hash2 = compute_structural_hash(&artifact2).unwrap();

        assert_ne!(
            hash1, hash2,
            "different tensor counts should produce different hashes"
        );
    }
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use crate::types::{CanonicalValue, Format, Tensor};
    use proptest::prelude::*;
    use std::collections::BTreeMap;

    proptest! {
        #[test]
        fn test_permutation_invariance_metadata(keys in prop::collection::vec("[a-z]{1,5}", 1..10)) {
            // Skip if there are duplicate keys
            if keys.len() != keys.iter().collect::<std::collections::HashSet<_>>().len() {
                return Ok(());
            }

            // Create two artifacts with same metadata but different insertion order
            let mut artifact1 = Artifact {
                format: Format::GGUF,
                gguf_version: Some(3),
                metadata: BTreeMap::new(),
                tensors: BTreeMap::new(),
            };
            let mut artifact2 = Artifact {
                format: Format::GGUF,
                gguf_version: Some(3),
                metadata: BTreeMap::new(),
                tensors: BTreeMap::new(),
            };

            for (i, key) in keys.iter().enumerate() {
                artifact1.metadata.insert(
                    key.clone(),
                    CanonicalValue::String(format!("value{}", i)),
                );
            }

            // Insert in reverse order
            for (i, key) in keys.iter().enumerate().rev() {
                artifact2.metadata.insert(
                    key.clone(),
                    CanonicalValue::String(format!("value{}", i)),
                );
            }

            let hash1 = compute_structural_hash(&artifact1).unwrap();
            let hash2 = compute_structural_hash(&artifact2).unwrap();

            prop_assert_eq!(hash1, hash2, "metadata order should not affect hash");
        }

        #[test]
        fn test_permutation_invariance_tensors(names in prop::collection::vec("[a-z]{1,5}", 1..5)) {
            // Skip if there are duplicate names
            if names.len() != names.iter().collect::<std::collections::HashSet<_>>().len() {
                return Ok(());
            }

            let mut artifact1 = Artifact {
                format: Format::GGUF,
                gguf_version: Some(3),
                metadata: BTreeMap::new(),
                tensors: BTreeMap::new(),
            };
            let mut artifact2 = Artifact {
                format: Format::GGUF,
                gguf_version: Some(3),
                metadata: BTreeMap::new(),
                tensors: BTreeMap::new(),
            };

            for name in names.iter() {
                artifact1.tensors.insert(
                    name.clone(),
                    Tensor {
                        name: name.clone(),
                        dtype: "f32".to_string(),
                        shape: vec![10, 10],
                        byte_length: 400,
                    },
                );
            }

            // Insert in reverse order
            for name in names.iter().rev() {
                artifact2.tensors.insert(
                    name.clone(),
                    Tensor {
                        name: name.clone(),
                        dtype: "f32".to_string(),
                        shape: vec![10, 10],
                        byte_length: 400,
                    },
                );
            }

            let hash1 = compute_structural_hash(&artifact1).unwrap();
            let hash2 = compute_structural_hash(&artifact2).unwrap();

            prop_assert_eq!(hash1, hash2, "tensor order should not affect hash");
        }
    }
}
