//! Build script for acton-ai.
//!
//! Compiles Hyperlight guest binaries and makes them available via include_bytes!.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=guests/shell_guest/src/lib.rs");
    println!("cargo:rerun-if-changed=guests/http_guest/src/lib.rs");
    println!("cargo:rerun-if-changed=guests/shell_guest/Cargo.toml");
    println!("cargo:rerun-if-changed=guests/http_guest/Cargo.toml");
    println!("cargo:rerun-if-changed=build.rs");

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));
    let guests_dir = out_dir.join("guests");
    fs::create_dir_all(&guests_dir).expect("Failed to create guests output directory");

    // Check if we can compile guests (requires nightly and x86_64-unknown-none target)
    if can_compile_guests() {
        compile_guests(&guests_dir);
    } else {
        // Create stub binaries with clear error message embedded
        create_stub_guests(&guests_dir);
    }
}

/// Check if we can compile guest binaries.
///
/// Requirements:
/// - Running on x86_64 architecture
/// - The x86_64-unknown-none target is available
/// - The guests source directory exists (not in crates.io package)
fn can_compile_guests() -> bool {
    // Check architecture
    if env::consts::ARCH != "x86_64" {
        println!(
            "cargo:warning=Guest compilation skipped: requires x86_64, got {}",
            env::consts::ARCH
        );
        return false;
    }

    // Check if guests source directory exists (won't exist in crates.io package)
    let manifest_dir =
        PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set"));
    let guests_dir = manifest_dir.join("guests").join("shell_guest").join("src");
    if !guests_dir.exists() {
        println!("cargo:warning=Guest compilation skipped: guests source not found (crates.io build)");
        return false;
    }

    // Check if x86_64-unknown-none target is available
    let output = Command::new("rustup")
        .args(["target", "list", "--installed"])
        .output();

    match output {
        Ok(output) => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            if stdout.contains("x86_64-unknown-none") {
                true
            } else {
                println!("cargo:warning=Guest compilation skipped: x86_64-unknown-none target not installed");
                println!("cargo:warning=Run: rustup target add x86_64-unknown-none");
                false
            }
        }
        Err(_) => {
            println!("cargo:warning=Guest compilation skipped: rustup not available");
            false
        }
    }
}

/// Compile guest binaries to x86_64-unknown-none.
fn compile_guests(output_dir: &Path) {
    let manifest_dir =
        PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set"));
    let guests_workspace = manifest_dir.join("guests");

    // Build shell_guest
    compile_guest(&guests_workspace, "shell_guest", output_dir);

    // Build http_guest
    compile_guest(&guests_workspace, "http_guest", output_dir);
}

/// Compile a single guest binary.
fn compile_guest(workspace: &Path, guest_name: &str, output_dir: &Path) {
    println!("cargo:warning=Compiling {} guest binary...", guest_name);

    let status = Command::new("cargo")
        .current_dir(workspace.join(guest_name))
        .args(["build", "--release", "--target", "x86_64-unknown-none"])
        .env("CARGO_TARGET_DIR", workspace.join("target"))
        .status();

    match status {
        Ok(status) if status.success() => {
            // Copy the compiled binary to output_dir
            let binary_path = workspace
                .join("target")
                .join("x86_64-unknown-none")
                .join("release")
                .join(format!("lib{}.a", guest_name));

            let dest_path = output_dir.join(guest_name);

            if binary_path.exists() {
                fs::copy(&binary_path, &dest_path)
                    .unwrap_or_else(|e| panic!("Failed to copy {} binary: {}", guest_name, e));
                println!("cargo:warning={} guest compiled successfully", guest_name);
            } else {
                // Try alternative path (might be named differently)
                let alt_path = workspace
                    .join("target")
                    .join("x86_64-unknown-none")
                    .join("release")
                    .join(guest_name);

                if alt_path.exists() {
                    fs::copy(&alt_path, &dest_path)
                        .unwrap_or_else(|e| panic!("Failed to copy {} binary: {}", guest_name, e));
                    println!("cargo:warning={} guest compiled successfully", guest_name);
                } else {
                    panic!(
                        "Compiled {} binary not found at {:?} or {:?}",
                        guest_name, binary_path, alt_path
                    );
                }
            }
        }
        Ok(status) => {
            panic!(
                "Failed to compile {} guest: exit code {:?}",
                guest_name,
                status.code()
            );
        }
        Err(e) => {
            panic!("Failed to run cargo for {} guest: {}", guest_name, e);
        }
    }
}

/// Create stub guest binaries when compilation isn't possible.
///
/// These will cause a clear runtime error when loaded into Hyperlight.
fn create_stub_guests(output_dir: &Path) {
    let stub_content = b"STUB_GUEST_BINARY:This is not a valid Hyperlight guest. \
        To compile real guests, install the x86_64-unknown-none target: \
        rustup target add x86_64-unknown-none";

    fs::write(output_dir.join("shell_guest"), stub_content)
        .expect("Failed to write stub shell_guest");
    fs::write(output_dir.join("http_guest"), stub_content)
        .expect("Failed to write stub http_guest");

    println!("cargo:warning=Created stub guest binaries (not functional)");
}
