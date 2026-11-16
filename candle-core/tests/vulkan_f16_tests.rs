//! Device tests for the f16-storage / f32-compute Vulkan kernels.
//!
//! Each op is computed on the Vulkan device in f16 storage and compared against
//! a CPU f32 reference computed from the *same* f16-rounded inputs (so both sides
//! round inputs identically). Skips silently when no Vulkan device is present.

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
        // f16 saturates to inf; if both sides overflow identically, that matches.
        if a.is_infinite() && b.is_infinite() && a.signum() == b.signum() {
            continue;
        }
        let err = (a - b).abs() / (1.0 + b.abs());
        max = max.max(err);
    }
    println!("{name}: normalized max err = {max:.3e}");
    assert!(max < tol, "{name}: max err {max:.3e} >= tol {tol:.3e}");
}

fn f16_on(device: &Device, data: Vec<f32>, dims: (usize, usize)) -> Tensor {
    Tensor::from_vec(data, dims, &Device::Cpu)
        .unwrap()
        .to_dtype(DType::F16)
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

#[test]
fn f16_unary() {
    let Some(dev) = vulkan_device() else { return };
    let n = 1000usize;
    // Range kept within f16 exp headroom (max ~8) so exp doesn't saturate to inf.
    let data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.011 - 3.0).collect();
    for name in ["gelu", "gelu_erf", "silu", "tanh", "neg", "exp", "abs"] {
        let vk = apply_unary(&f16_on(&dev, data.clone(), (1, n)), name);
        let ref_f32 = apply_unary(&f32_cpu(data.clone(), (1, n)), name);
        check(&vk, &ref_f32, 0.02, &format!("unary/{name}"));
    }
}

#[test]
fn f16_binary() {
    let Some(dev) = vulkan_device() else { return };
    let (r, c) = (37usize, 64usize);
    let a: Vec<f32> = (0..r * c).map(|i| (i as f32) * 0.01 - 1.0).collect();
    let b: Vec<f32> = (0..r * c).map(|i| ((i as f32) * 0.03).sin()).collect();
    let av = f16_on(&dev, a.clone(), (r, c));
    let bv = f16_on(&dev, b.clone(), (r, c));
    let af = f32_cpu(a.clone(), (r, c));
    let bf = f32_cpu(b.clone(), (r, c));

    check(&av.broadcast_add(&bv).unwrap(), &af.broadcast_add(&bf).unwrap(), 0.02, "binary/add");
    check(&av.broadcast_sub(&bv).unwrap(), &af.broadcast_sub(&bf).unwrap(), 0.02, "binary/sub");
    check(&av.broadcast_mul(&bv).unwrap(), &af.broadcast_mul(&bf).unwrap(), 0.02, "binary/mul");
    check(&av.broadcast_div(&bv).unwrap(), &af.broadcast_div(&bf).unwrap(), 0.02, "binary/div");

    // Broadcast (bias add): [r, c] + [1, c] exercises stride-0 on the row dim.
    let bias: Vec<f32> = (0..c).map(|i| (i as f32) * 0.1).collect();
    let biasv = f16_on(&dev, bias.clone(), (1, c));
    let biasf = f32_cpu(bias.clone(), (1, c));
    check(
        &av.broadcast_add(&biasv).unwrap(),
        &af.broadcast_add(&biasf).unwrap(),
        0.02,
        "binary/add_broadcast",
    );
}

#[test]
fn f16_affine() {
    let Some(dev) = vulkan_device() else { return };
    let (r, c) = (16usize, 100usize);
    let data: Vec<f32> = (0..r * c).map(|i| (i as f32) * 0.02 - 2.0).collect();
    let vk = f16_on(&dev, data.clone(), (r, c)).affine(1.7, -0.4).unwrap();
    let ref_f32 = f32_cpu(data, (r, c)).affine(1.7, -0.4).unwrap();
    check(&vk, &ref_f32, 0.02, "affine");
}

#[test]
fn f16_index_select() {
    let Some(dev) = vulkan_device() else { return };
    let (rows, cols) = (64usize, 128usize);
    let data: Vec<f32> = (0..rows * cols).map(|i| ((i as f32) * 0.013).cos()).collect();
    let ids_v: Vec<u32> = vec![5, 0, 3, 3, 63, 1, 42, 7];
    let src = f16_on(&dev, data.clone(), (rows, cols));
    let ids = Tensor::from_vec(ids_v.clone(), (ids_v.len(),), &dev).unwrap();
    let got = src.index_select(&ids, 0).unwrap();
    let src_f = f32_cpu(data, (rows, cols));
    let ids_f = Tensor::from_vec(ids_v.clone(), (ids_v.len(),), &Device::Cpu).unwrap();
    let want = src_f.index_select(&ids_f, 0).unwrap();
    check(&got, &want, 0.02, "index_select/rows");
}

#[test]
fn f16_matmul() {
    let Some(dev) = vulkan_device() else { return };
    // m x k @ k x n, n even (packed store). Values O(1), sums up to ~k.
    let (m, k, n) = (48usize, 96usize, 80usize);
    let a: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.017).sin()).collect();
    let b: Vec<f32> = (0..k * n).map(|i| ((i as f32) * 0.011).cos()).collect();
    let av = f16_on(&dev, a.clone(), (m, k));
    let bv = f16_on(&dev, b.clone(), (k, n));
    let got = av.matmul(&bv).unwrap();
    let want = f32_cpu(a, (m, k)).matmul(&f32_cpu(b, (k, n))).unwrap();
    check(&got, &want, 0.05, "matmul");
}

#[test]
fn f16_matmul_splitk() {
    let Some(dev) = vulkan_device() else { return };
    // Small M/N, large K (>=256) so the dispatch takes the split-K path.
    let (m, k, n) = (32usize, 1024usize, 64usize);
    let a: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.007).sin()).collect();
    let b: Vec<f32> = (0..k * n).map(|i| ((i as f32) * 0.013).cos()).collect();
    let av = f16_on(&dev, a.clone(), (m, k));
    let bv = f16_on(&dev, b.clone(), (k, n));
    let got = av.matmul(&bv).unwrap();
    let want = f32_cpu(a, (m, k)).matmul(&f32_cpu(b, (k, n))).unwrap();
    check(&got, &want, 0.05, "matmul_splitk");
}

#[test]
fn f16_matmul_batched() {
    let Some(dev) = vulkan_device() else { return };
    let (b, m, k, n) = (4usize, 32usize, 64usize, 40usize);
    let a: Vec<f32> = (0..b * m * k).map(|i| ((i as f32) * 0.021).cos()).collect();
    let w: Vec<f32> = (0..b * k * n).map(|i| ((i as f32) * 0.013).sin()).collect();
    let av = Tensor::from_vec(a.clone(), (b, m, k), &Device::Cpu)
        .unwrap()
        .to_dtype(DType::F16)
        .unwrap()
        .to_device(&dev)
        .unwrap();
    let wv = Tensor::from_vec(w.clone(), (b, k, n), &Device::Cpu)
        .unwrap()
        .to_dtype(DType::F16)
        .unwrap()
        .to_device(&dev)
        .unwrap();
    let got = av.matmul(&wv).unwrap();
    let want = f32_cpu(a, (b * m, k))
        .reshape((b, m, k))
        .unwrap()
        .matmul(&f32_cpu(w, (b * k, n)).reshape((b, k, n)).unwrap())
        .unwrap();
    check(&got, &want, 0.05, "matmul_batched");
}
