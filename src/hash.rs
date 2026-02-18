use crate::types::Artifact;
use sha2::{Digest, Sha256};

pub fn compute_structural_hash(artifact: &Artifact) -> String {
    let canonical = serde_json::to_string(artifact).unwrap();
    let mut hasher = Sha256::new();
    hasher.update(canonical.as_bytes());
    let result = hasher.finalize();
    hex::encode(result)
}
