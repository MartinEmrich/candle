//! Regression test for the deferred-batch memory bound.
//!
//! A model forward is only a few hundred ops, so with a naive op-count-only flush
//! threshold the whole forward is batched and every intermediate is kept alive
//! until the single submit at the end — peak VRAM becomes the *sum* of all
//! intermediates, which OOMs at production batch sizes (this is what the TEI
//! warmup at `max_batch_tokens=16384` hit). The batcher now also flushes on a
//! byte budget so live memory stays bounded regardless of op count.
//!
//! This test chains many large-buffer ops whose logical total (~12 GB) far
//! exceeds any consumer GPU's VRAM. With the byte-budget flush it completes; if
//! the bound regressed it would exhaust VRAM and error / lose the device.

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

#[test]
fn long_op_chain_bounds_peak_memory() {
    let Some(dev) = vulkan_device() else { return };

    // ~25 MB per buffer (f16). 480 chained ops => ~12 GB of intermediates if
    // none are released until the end.
    let (rows, cols) = (16384usize, 768usize);
    let base = Tensor::arange(0f32, (rows * cols) as f32, &Device::Cpu)
        .unwrap()
        .affine(1e-4, 0.0)
        .unwrap()
        .to_dtype(DType::F16)
        .unwrap()
        .to_device(&dev)
        .unwrap();

    let mut x = base.clone();
    for _ in 0..480 {
        // Identity affine: a fresh output buffer per op, previous dropped by the
        // caller (only the batcher's keepalive can hold it).
        x = x.affine(1.0, 0.0).unwrap();
    }

    // The chain is an identity, so the result must equal the input exactly.
    let got = x
        .to_dtype(DType::F32)
        .unwrap()
        .to_device(&Device::Cpu)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let want = base
        .to_dtype(DType::F32)
        .unwrap()
        .to_device(&Device::Cpu)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    assert_eq!(got, want, "identity affine chain must preserve values");
}
