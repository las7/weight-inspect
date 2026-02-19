//! Integration tests for CLI commands
//! Run with: cargo test --test cli

use std::process::{Command, Stdio};
use tempfile::NamedTempFile;

fn run_cli(args: &[&str]) -> std::process::Output {
    // Find the project root by walking up from the test binary location
    // The test runs from target/debug/deps/cli-XXX, project root is 4 levels up
    let exe_path = std::env::current_exe().expect("Failed to get current exe");
    let parent1 = exe_path.parent().expect("Failed to get parent");
    let parent2 = parent1.parent().expect("Failed to get grandparent");
    let parent3 = parent2.parent().expect("Failed to get great-grandparent");
    let parent4 = parent3
        .parent()
        .expect("Failed to get great-great-grandparent");
    let project_dir = parent4.to_path_buf();

    let manifest_path = project_dir.join("Cargo.toml");

    // Use cargo run from project root
    let mut cmd = Command::new("cargo");
    cmd.args(["run", "--manifest-path"])
        .arg(&manifest_path)
        .arg("--")
        .args(args)
        .current_dir(&project_dir)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    match cmd.output() {
        Ok(output) => output,
        Err(_) => {
            // Fallback: try /home/runner path for CI
            let project_dir =
                std::path::PathBuf::from("/home/runner/work/weight-inspect/weight-inspect");
            let manifest_path = project_dir.join("Cargo.toml");
            Command::new("cargo")
                .args(["run", "--manifest-path"])
                .arg(&manifest_path)
                .arg("--")
                .args(args)
                .current_dir(&project_dir)
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
    assert!(!output.status.success());
}

#[test]
fn test_id_json_flag() {
    let temp = NamedTempFile::new().expect("Failed to create temp file");
    let path = temp.path();
    let output = run_cli(&["id", &path.to_string_lossy(), "--json"]);
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

#[test]
fn test_gguf_fixture_inspect() {
    let output = run_cli(&["inspect", "tests/fixtures/tiny.gguf"]);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    eprintln!("stdout: {}\nstderr: {}", stdout, stderr);
    assert!(output.status.success(), "Command failed: {}", stdout);
    assert!(stdout.contains("gguf"));
    assert!(stdout.contains("Version:"));
    assert!(stdout.contains("Tensors:"));
    assert!(stdout.contains("Hash:"));
}

#[test]
fn test_diff_different() {
    let output = run_cli(&[
        "diff",
        "tests/fixtures/empty.gguf",
        "tests/fixtures/tiny.gguf",
    ]);
    let stdout = String::from_utf8_lossy(&output.stdout);
    // diff returns success even when files are different (use --fail-on-diff for CI)
    assert!(output.status.success());
    assert!(stdout.contains("DIFFERENT") || stdout.contains("Added") || stdout.contains("Removed"));
}

#[test]
fn test_safetensors_fixture_inspect() {
    let output = run_cli(&["inspect", "tests/fixtures/tiny.safetensors"]);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    eprintln!("stdout: {}\nstderr: {}", stdout, stderr);
    assert!(output.status.success(), "Command failed: {}", stdout);
    assert!(stdout.contains("safetensors"));
    assert!(stdout.contains("Tensors:"));
}

#[test]
fn test_diff_identical() {
    let output = run_cli(&[
        "diff",
        "tests/fixtures/tiny.gguf",
        "tests/fixtures/tiny.gguf",
    ]);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    eprintln!("stdout: {}\nstderr: {}", stdout, stderr);
    assert!(output.status.success(), "Command failed: {}", stdout);
    assert!(stdout.contains("IDENTICAL") || stdout.contains("No structural differences"));
}

#[test]
fn test_id_json_output() {
    let output = run_cli(&["id", "tests/fixtures/tiny.gguf", "--json"]);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success());
    assert!(stdout.contains("\"format\":"));
    assert!(stdout.contains("\"structural_hash\":"));
}

#[test]
fn test_inspect_verbose() {
    let output = run_cli(&["inspect", "tests/fixtures/tiny.gguf", "--verbose"]);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success());
    assert!(stdout.contains("gguf"));
}

#[test]
fn test_diff_format_md() {
    let output = run_cli(&[
        "diff",
        "tests/fixtures/tiny.gguf",
        "tests/fixtures/tiny.gguf",
        "--format",
        "md",
    ]);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success());
    // Markdown output should contain markdown table syntax
    assert!(stdout.contains("|") || stdout.contains("Status"));
}

#[test]
fn test_diff_fail_on_diff() {
    let output = run_cli(&[
        "diff",
        "--fail-on-diff",
        "tests/fixtures/empty.gguf",
        "tests/fixtures/tiny.gguf",
    ]);
    // Should fail (non-zero exit) because files are different
    assert!(!output.status.success());
}

#[test]
fn test_diff_only_changes() {
    let output = run_cli(&[
        "diff",
        "--only-changes",
        "tests/fixtures/empty.gguf",
        "tests/fixtures/tiny.gguf",
    ]);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success());
    // Should show only changes (Added tensors)
    assert!(stdout.contains("Added") || stdout.contains("Removed") || stdout.contains("Modified"));
}

#[test]
fn test_summary_command() {
    let output = run_cli(&["summary", "tests/fixtures/tiny.gguf"]);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success());
    // Summary should be one line, comma-separated
    assert!(stdout.contains(","));
}

#[test]
fn test_invalid_format_error() {
    let output = run_cli(&[
        "diff",
        "--format",
        "invalid",
        "tests/fixtures/tiny.gguf",
        "tests/fixtures/tiny.gguf",
    ]);
    // Should fail with invalid format error
    assert!(!output.status.success());
}
