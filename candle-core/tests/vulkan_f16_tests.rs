//! Device tests for the 16-bit storage (f16 / bf16) / f32-compute Vulkan kernels.
//!
//! Each op is computed on the Vulkan device in narrow storage and compared
//! against a CPU f32 reference computed from the *same* narrow-rounded inputs (so
//! both sides round inputs identically). Skips silently when no Vulkan device is
//! present.

#![cfg(feature = "vulkan")]

use candle_core::{DType, Device, Tensor};

fn vulkan_device() -> Option<Device> {
    if !candle_core::utils::vulkan_is_available() {
        return None;
    }
    match Device::new_vulkan(0) {
        Ok(d) => Some(d),
        Err(e) => {
            println!("skipping: no usable Vulkan device ({e})");
            None
        }
    }
}

fn to_cpu_f32(t: &Tensor) -> Vec<f32> {
    t.to_dtype(DType::F32)
        .unwrap()
        .to_device(&Device::Cpu)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
}

/// Compare a device result against a CPU f32 reference using a scale-normalized
/// error (abs error for small values, rel error for large ones).
fn check(got: &Tensor, want: &Tensor, tol: f32, name: &str) {
    let g = to_cpu_f32(got);
    let w = to_cpu_f32(want);
    assert_eq!(g.len(), w.len(), "{name}: length mismatch");
    let mut max = 0f32;
    for (a, b) in g.iter().zip(w.iter()) {
        if a.is_infinite() && b.is_infinite() && a.signum() == b.signum() {
            continue;
        }
        let err = (a - b).abs() / (1.0 + b.abs());
        max = max.max(err);
    }
    println!("{name}: normalized max err = {max:.3e}");
    assert!(max < tol, "{name}: max err {max:.3e} >= tol {tol:.3e}");
}

fn narrow_on(device: &Device, data: Vec<f32>, dims: (usize, usize), dt: DType) -> Tensor {
    Tensor::from_vec(data, dims, &Device::Cpu)
        .unwrap()
        .to_dtype(dt)
        .unwrap()
        .to_device(device)
        .unwrap()
}

fn f32_cpu(data: Vec<f32>, dims: (usize, usize)) -> Tensor {
    Tensor::from_vec(data, dims, &Device::Cpu).unwrap()
}

fn apply_unary(t: &Tensor, name: &str) -> Tensor {
    match name {
        "gelu" => t.gelu().unwrap(),
        "gelu_erf" => t.gelu_erf().unwrap(),
        "silu" => t.silu().unwrap(),
        "tanh" => t.tanh().unwrap(),
        "neg" => t.neg().unwrap(),
        "exp" => t.exp().unwrap(),
        "abs" => t.abs().unwrap(),
        other => panic!("unknown unary op {other}"),
    }
}

fn run_unary(dev: &Device, dt: DType, tol: f32) {
    let n = 1000usize;
    // Range kept within f16 exp headroom (max ~8) so exp doesn't saturate to inf.
    let data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.011 - 3.0).collect();
    for name in ["gelu", "gelu_erf", "silu", "tanh", "neg", "exp", "abs"] {
        let vk = apply_unary(&narrow_on(dev, data.clone(), (1, n), dt), name);
        let ref_f32 = apply_unary(&f32_cpu(data.clone(), (1, n)), name);
        check(&vk, &ref_f32, tol, &format!("unary/{name}"));
    }
}

fn run_binary(dev: &Device, dt: DType, tol: f32) {
    let (r, c) = (37usize, 64usize);
    let a: Vec<f32> = (0..r * c).map(|i| (i as f32) * 0.01 - 1.0).collect();
    let b: Vec<f32> = (0..r * c).map(|i| ((i as f32) * 0.03).sin()).collect();
    let av = narrow_on(dev, a.clone(), (r, c), dt);
    let bv = narrow_on(dev, b.clone(), (r, c), dt);
    let af = f32_cpu(a.clone(), (r, c));
    let bf = f32_cpu(b.clone(), (r, c));

    check(&av.broadcast_add(&bv).unwrap(), &af.broadcast_add(&bf).unwrap(), tol, "binary/add");
    check(&av.broadcast_sub(&bv).unwrap(), &af.broadcast_sub(&bf).unwrap(), tol, "binary/sub");
    check(&av.broadcast_mul(&bv).unwrap(), &af.broadcast_mul(&bf).unwrap(), tol, "binary/mul");
    check(&av.broadcast_div(&bv).unwrap(), &af.broadcast_div(&bf).unwrap(), tol, "binary/div");

    // Broadcast (bias add): [r, c] + [1, c] exercises stride-0 on the row dim.
    let bias: Vec<f32> = (0..c).map(|i| (i as f32) * 0.1).collect();
    let biasv = narrow_on(dev, bias.clone(), (1, c), dt);
    let biasf = f32_cpu(bias.clone(), (1, c));
    check(
        &av.broadcast_add(&biasv).unwrap(),
        &af.broadcast_add(&biasf).unwrap(),
        tol,
        "binary/add_broadcast",
    );
}

fn run_affine(dev: &Device, dt: DType, tol: f32) {
    let (r, c) = (16usize, 100usize);
    let data: Vec<f32> = (0..r * c).map(|i| (i as f32) * 0.02 - 2.0).collect();
    let vk = narrow_on(dev, data.clone(), (r, c), dt).affine(1.7, -0.4).unwrap();
    let ref_f32 = f32_cpu(data, (r, c)).affine(1.7, -0.4).unwrap();
    check(&vk, &ref_f32, tol, "affine");
}

fn run_index_select(dev: &Device, dt: DType, tol: f32) {
    let (rows, cols) = (64usize, 128usize);
    let data: Vec<f32> = (0..rows * cols).map(|i| ((i as f32) * 0.013).cos()).collect();
    let ids_v: Vec<u32> = vec![5, 0, 3, 3, 63, 1, 42, 7];
    let src = narrow_on(dev, data.clone(), (rows, cols), dt);
    let ids = Tensor::from_vec(ids_v.clone(), (ids_v.len(),), dev).unwrap();
    let got = src.index_select(&ids, 0).unwrap();
    let src_f = f32_cpu(data, (rows, cols));
    let ids_f = Tensor::from_vec(ids_v.clone(), (ids_v.len(),), &Device::Cpu).unwrap();
    let want = src_f.index_select(&ids_f, 0).unwrap();
    check(&got, &want, tol, "index_select/rows");
}

fn run_matmul(dev: &Device, dt: DType, tol: f32) {
    let (m, k, n) = (48usize, 96usize, 80usize);
    let a: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.017).sin()).collect();
    let b: Vec<f32> = (0..k * n).map(|i| ((i as f32) * 0.011).cos()).collect();
    let av = narrow_on(dev, a.clone(), (m, k), dt);
    let bv = narrow_on(dev, b.clone(), (k, n), dt);
    let got = av.matmul(&bv).unwrap();
    let want = f32_cpu(a, (m, k)).matmul(&f32_cpu(b, (k, n))).unwrap();
    check(&got, &want, tol, "matmul");
}

fn run_matmul_splitk(dev: &Device, dt: DType, tol: f32) {
    // Small M/N, large K (>=256) so the dispatch takes the split-K path.
    let (m, k, n) = (32usize, 1024usize, 64usize);
    let a: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.007).sin()).collect();
    let b: Vec<f32> = (0..k * n).map(|i| ((i as f32) * 0.013).cos()).collect();
    let av = narrow_on(dev, a.clone(), (m, k), dt);
    let bv = narrow_on(dev, b.clone(), (k, n), dt);
    let got = av.matmul(&bv).unwrap();
    let want = f32_cpu(a, (m, k)).matmul(&f32_cpu(b, (k, n))).unwrap();
    check(&got, &want, tol, "matmul_splitk");
}

fn run_matmul_batched(dev: &Device, dt: DType, tol: f32) {
    let (b, m, k, n) = (4usize, 32usize, 64usize, 40usize);
    let a: Vec<f32> = (0..b * m * k).map(|i| ((i as f32) * 0.021).cos()).collect();
    let w: Vec<f32> = (0..b * k * n).map(|i| ((i as f32) * 0.013).sin()).collect();
    let av = Tensor::from_vec(a.clone(), (b, m, k), &Device::Cpu)
        .unwrap()
        .to_dtype(dt)
        .unwrap()
        .to_device(dev)
        .unwrap();
    let wv = Tensor::from_vec(w.clone(), (b, k, n), &Device::Cpu)
        .unwrap()
        .to_dtype(dt)
        .unwrap()
        .to_device(dev)
        .unwrap();
    let got = av.matmul(&wv).unwrap();
    let want = f32_cpu(a, (b * m, k))
        .reshape((b, m, k))
        .unwrap()
        .matmul(&f32_cpu(w, (b * k, n)).reshape((b, k, n)).unwrap())
        .unwrap();
    check(&got, &want, tol, "matmul_batched");
}

const F16: DType = DType::F16;
const BF16: DType = DType::BF16;
// bf16 has ~8 mantissa bits (eps ~ 0.0078) so tolerances are looser than f16.
const T16: f32 = 0.02;
const T16_MM: f32 = 0.05;
const TB16: f32 = 0.06;
const TB16_MM: f32 = 0.12;

// f16 tests
#[test]
fn f16_unary() { if let Some(d) = vulkan_device() { run_unary(&d, F16, T16) } }
#[test]
fn f16_binary() { if let Some(d) = vulkan_device() { run_binary(&d, F16, T16) } }
#[test]
fn f16_affine() { if let Some(d) = vulkan_device() { run_affine(&d, F16, T16) } }
#[test]
fn f16_index_select() { if let Some(d) = vulkan_device() { run_index_select(&d, F16, T16) } }
#[test]
fn f16_matmul() { if let Some(d) = vulkan_device() { run_matmul(&d, F16, T16_MM) } }
#[test]
fn f16_matmul_splitk() { if let Some(d) = vulkan_device() { run_matmul_splitk(&d, F16, T16_MM) } }
#[test]
fn f16_matmul_batched() { if let Some(d) = vulkan_device() { run_matmul_batched(&d, F16, T16_MM) } }

// bf16 tests
#[test]
fn bf16_unary() { if let Some(d) = vulkan_device() { run_unary(&d, BF16, TB16) } }
#[test]
fn bf16_binary() { if let Some(d) = vulkan_device() { run_binary(&d, BF16, TB16) } }
#[test]
fn bf16_affine() { if let Some(d) = vulkan_device() { run_affine(&d, BF16, TB16) } }
#[test]
fn bf16_index_select() { if let Some(d) = vulkan_device() { run_index_select(&d, BF16, TB16) } }
#[test]
fn bf16_matmul() { if let Some(d) = vulkan_device() { run_matmul(&d, BF16, TB16_MM) } }
#[test]
fn bf16_matmul_splitk() { if let Some(d) = vulkan_device() { run_matmul_splitk(&d, BF16, TB16_MM) } }
#[test]
fn bf16_matmul_batched() { if let Some(d) = vulkan_device() { run_matmul_batched(&d, BF16, TB16_MM) } }
