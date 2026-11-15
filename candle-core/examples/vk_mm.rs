use candle_core::{Device, DType, Tensor};

fn bench(name: &str, dev: &Device, mut f: impl FnMut() -> candle_core::Result<Tensor>, n: usize) {
    for _ in 0..5 {
        let _ = f().unwrap();
    }
    dev.synchronize().unwrap();
    let t0 = std::time::Instant::now();
    for _ in 0..n {
        let _ = f().unwrap();
    }
    dev.synchronize().unwrap();
    let us = t0.elapsed().as_secs_f64() * 1e6 / n as f64;
    println!("{name:38} {us:>9.1} us/iter");
}

fn main() -> anyhow::Result<()> {
    let dev = Device::new_vulkan(0)?;
    // Qwen3-Embedding-0.6B projection shapes for a 3-sentence batch (~45 tokens packed).
    for m in [45usize, 128, 512] {
        let x = Tensor::ones((m, 1024), DType::F32, &dev)?;
        let xk = Tensor::ones((m, 3072), DType::F32, &dev)?;
        let wq = Tensor::ones((1024, 2048), DType::F32, &dev)?;
        let wkv = Tensor::ones((1024, 1024), DType::F32, &dev)?;
        let wo = Tensor::ones((2048, 1024), DType::F32, &dev)?;
        let wg = Tensor::ones((1024, 3072), DType::F32, &dev)?;
        let wd = Tensor::ones((3072, 1024), DType::F32, &dev)?;
        println!("--- M={m} ---");
        bench(&format!("q_proj  ({m}x1024x2048)"), &dev, || x.matmul(&wq), 200);
        bench(&format!("kv_proj ({m}x1024x1024)"), &dev, || x.matmul(&wkv), 200);
        bench(&format!("o_proj  ({m}x2048x1024)"), &dev, || {
            let a = Tensor::ones((m, 2048), DType::F32, &dev)?;
            a.matmul(&wo)
        }, 200);
        bench(&format!("gate/up ({m}x1024x3072)"), &dev, || x.matmul(&wg), 200);
        bench(&format!("down    ({m}x3072x1024)"), &dev, || xk.matmul(&wd), 200);
    }
    Ok(())
}
