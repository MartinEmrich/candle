//! Compiles every `src/*.glsl` compute shader into SPIR-V using glslc.
//!
//! The glslc binary can be overridden with the `CANDLE_VULKAN_GLSLC` environment
//! variable. Compiled modules are emitted into OUT_DIR and referenced from
//! `src/lib.rs` via `include_bytes!`.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

fn main() {
    println!("cargo:rerun-if-env-changed=CANDLE_VULKAN_GLSLC");
    println!("cargo:rerun-if-changed=src");

    let glslc = std::env::var("CANDLE_VULKAN_GLSLC").unwrap_or_else(|_| "glslc".to_string());
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());

    let mut shaders = vec![];
    for entry in std::fs::read_dir("src").unwrap() {
        let path = entry.unwrap().path();
        if path.extension().and_then(|e| e.to_str()) != Some("glsl") {
            continue;
        }
        shaders.push(path);
    }
    shaders.sort();

    for shader in &shaders {
        println!("cargo:rerun-if-changed={}", shader.display());
        let stem = shader.file_stem().unwrap().to_str().unwrap().to_string();
        let spv = out_dir.join(format!("{stem}.spv"));
        compile(&glslc, shader, &spv);
    }
}

fn compile(glslc: &str, shader: &Path, spv: &Path) {
    let status = Command::new(glslc)
        .arg("-fshader-stage=compute")
        .arg("--target-env=vulkan1.2")
        .arg("-O")
        .arg("-o")
        .arg(spv)
        .arg(shader)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .status()
        .unwrap_or_else(|err| {
            panic!(
                "failed to run glslc ({glslc}), install the Vulkan SDK or set CANDLE_VULKAN_GLSLC: {err}"
            )
        });
    if !status.success() {
        panic!("glslc failed to compile {}", shader.display());
    }
}
