//! Compiles every `src/*.glsl` compute shader into SPIR-V using glslc.
//!
//! The glslc binary can be overridden with the `CANDLE_VULKAN_GLSLC` environment
//! variable. Compiled modules are emitted into OUT_DIR and referenced from
//! `src/lib.rs` via `include_bytes!`.
//!
//! Most kernels target Vulkan 1.2. The cooperative-matrix matmul needs Vulkan 1.3
//! and a glslc that supports `VK_KHR_cooperative_matrix`; when the build-time
//! glslc lacks it (e.g. a stock Debian shaderc), that kernel is skipped and an
//! empty stub `.spv` is emitted so `include_bytes!` still resolves, and the host
//! disables the coopmat path when the stub is empty.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// A minimal valid `VK_KHR_cooperative_matrix` compute shader, compiled at
/// Vulkan 1.3 to decide whether this glslc understands the extension at all.
const COOPMAT_PROBE: &str = r#"#version 450
#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
layout(local_size_x = 32) in;
layout(set = 0, binding = 0) readonly buffer A { float16_t a[]; } bA;
void main() {
    coopmat<float16_t, gl_ScopeSubgroup, 16, 16, 0> a;
    coopMatLoad(a, bA.a, 0u, 16u, 0);
}
"#;

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

    // Only pay for the coopmat probe if a coopmat kernel is actually present.
    let need_coopmat = shaders
        .iter()
        .any(|s| s.file_stem().and_then(|s| s.to_str()).is_some_and(|s| s.contains("coopmat")));
    let coopmat_supported = if need_coopmat {
        probe_coopmat(&glslc, &out_dir)
    } else {
        false
    };

    for shader in &shaders {
        println!("cargo:rerun-if-changed={}", shader.display());
        let stem = shader.file_stem().unwrap().to_str().unwrap().to_string();
        let spv = out_dir.join(format!("{stem}.spv"));
        if stem.contains("coopmat") {
            if coopmat_supported {
                compile(&glslc, shader, &spv, "vulkan1.3");
            } else {
                // glslc lacks VK_KHR_cooperative_matrix: emit an empty stub so
                // include_bytes! resolves; the host skips the path when empty.
                std::fs::write(&spv, []).unwrap();
                println!(
                    "cargo:warning=glslc does not support VK_KHR_cooperative_matrix; \
                     compiling {stem} as an empty stub (coopmat matmul disabled at runtime)"
                );
            }
        } else {
            compile(&glslc, shader, &spv, "vulkan1.2");
        }
    }
}

/// Compile the coopmat probe; true only if this glslc accepts it at Vulkan 1.3.
/// Any failure (missing binary, unknown target-env, missing extension) yields
/// `false`; the regular `compile` below still surfaces a missing glslc with a
/// helpful message when it compiles the non-coopmat kernels.
fn probe_coopmat(glslc: &str, out_dir: &Path) -> bool {
    let src = out_dir.join("coopmat_probe.glsl");
    let spv = out_dir.join("coopmat_probe.spv");
    if std::fs::write(&src, COOPMAT_PROBE).is_err() {
        return false;
    }
    Command::new(glslc)
        .arg("-fshader-stage=compute")
        .arg("--target-env=vulkan1.3")
        .arg("-O")
        .arg("-o")
        .arg(&spv)
        .arg(&src)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn compile(glslc: &str, shader: &Path, spv: &Path, target_env: &str) {
    let status = Command::new(glslc)
        .arg("-fshader-stage=compute")
        .arg(format!("--target-env={target_env}"))
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
