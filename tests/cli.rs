//! Integration tests for CLI commands
//! Run with: cargo test --test cli

use std::process::{Command, Stdio};
use tempfile::NamedTempFile;

fn run_cli(args: &[&str]) -> std::process::Output {
    // Tests run from target/debug/deps/, so we need to go up to project root
    let project_dir = std::path::Path::new("..");

    // Use cargo run from project root
    let mut cmd = Command::new("cargo");
    cmd.args(["run", "--manifest-path", "Cargo.toml"])
        .arg("--")
        .args(args)
        .current_dir(project_dir)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    match cmd.output() {
        Ok(output) => output,
        Err(_) => {
            // Fallback: try /home/runner path for CI
            let project_dir =
                std::path::Path::new("/home/runner/work/weight-inspect/weight-inspect");
            Command::new("cargo")
                .args(["run", "--manifest-path", "Cargo.toml"])
                .arg("--")
                .args(args)
                .current_dir(project_dir)
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .output()
                .expect("Failed to run cargo")
        }
    }
}

#[test]
fn test_nonexistent_file() {
    let output = run_cli(&["inspect", "/nonexistent/file.gguf"]);

    // Should fail with non-zero exit code
    assert!(!output.status.success());
}

#[test]
fn test_id_json_flag() {
    let temp = NamedTempFile::new().expect("Failed to create temp file");
    let path = temp.path();

    let output = run_cli(&["id", &path.to_string_lossy(), "--json"]);

    // Just check the flag is accepted (should parse or fail gracefully)
    let stderr = String::from_utf8_lossy(&output.stderr);
    println!("stderr: {}", stderr);
}

#[test]
fn test_inspect_json_flag() {
    let temp = NamedTempFile::new().expect("Failed to create temp file");
    let path = temp.path();

    let output = run_cli(&["inspect", &path.to_string_lossy(), "--json"]);

    let stderr = String::from_utf8_lossy(&output.stderr);
    println!("stderr: {}", stderr);
}
