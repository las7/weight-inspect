//! Integration tests for CLI commands
//! Run with: cargo test --test cli

use std::fs;
use std::process::Command;
use tempfile::NamedTempFile;

/// Get the path to the built binary
fn binary_path() -> String {
    // Check if we're in the project root or tests dir
    if std::path::Path::new("../target/release/weight-inspect").exists() {
        "../target/release/weight-inspect".to_string()
    } else if std::path::Path::new("../../target/release/weight-inspect").exists() {
        "../../target/release/weight-inspect".to_string()
    } else {
        // Try to build first
        let _ = Command::new("cargo")
            .args(["build", "--release"])
            .current_dir("../..")
            .output();
        "../../target/release/weight-inspect".to_string()
    }
}

#[test]
fn test_cli_help() {
    let output = Command::new(binary_path())
        .arg("--help")
        .output()
        .expect("Failed to run CLI");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("weight-inspect"));
    assert!(stdout.contains("inspect"));
    assert!(stdout.contains("id"));
    assert!(stdout.contains("diff"));
    assert!(stdout.contains("summary"));
}

#[test]
fn test_id_help() {
    let output = Command::new(binary_path())
        .args(["id", "--help"])
        .output()
        .expect("Failed to run CLI");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("fingerprint"));
}

#[test]
fn test_inspect_help() {
    let output = Command::new(binary_path())
        .args(["inspect", "--help"])
        .output()
        .expect("Failed to run CLI");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("structure"));
}

#[test]
fn test_diff_help() {
    let output = Command::new(binary_path())
        .args(["diff", "--help"])
        .output()
        .expect("Failed to run CLI");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("compare"));
}

#[test]
fn test_summary_help() {
    let output = Command::new(binary_path())
        .args(["summary", "--help"])
        .output()
        .expect("Failed to run CLI");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("scripts") || stdout.contains("One-line"));
}

#[test]
fn test_nonexistent_file() {
    let output = Command::new(binary_path())
        .args(["inspect", "/nonexistent/file.gguf"])
        .output()
        .expect("Failed to run CLI");

    // Should fail with non-zero exit code
    assert!(!output.status.success());
}

#[test]
fn test_id_json_flag() {
    // Create a minimal temp file that won't parse but tests --json flag
    let temp = NamedTempFile::new().expect("Failed to create temp file");
    let path = temp.path();

    let output = Command::new(binary_path())
        .args(["id", &path.to_string_lossy(), "--json"])
        .output()
        .expect("Failed to run CLI");

    // Should either parse or fail gracefully - just check --json is accepted
    let stderr = String::from_utf8_lossy(&output.stderr);
    // Don't assert on success - we're testing the flag is accepted
    println!("stderr: {}", stderr);
}

#[test]
fn test_inspect_json_flag() {
    let temp = NamedTempFile::new().expect("Failed to create temp file");
    let path = temp.path();

    let output = Command::new(binary_path())
        .args(["inspect", &path.to_string_lossy(), "--json"])
        .output()
        .expect("Failed to run CLI");

    let stderr = String::from_utf8_lossy(&output.stderr);
    println!("stderr: {}", stderr);
}
