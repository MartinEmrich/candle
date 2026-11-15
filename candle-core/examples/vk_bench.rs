use candle_core::{Device, Tensor, DType};

fn bench(name: &str, dev: &Device, mut f: impl FnMut() -> candle_core::Result<Tensor>, n: usize) {
    for _ in 0..5 { let _ = f().unwrap(); }
    dev.synchronize().unwrap();
    let t0 = std::time::Instant::now();
    for _ in 0..n { let _ = f().unwrap(); }
    dev.synchronize().unwrap();
    println!("{name:34} {:>9.1?} /iter", t0.elapsed() / n as u32);
}

fn layer_norm(x: &Tensor, eps: f32) -> candle_core::Result<Tensor> {
    let mean = x.mean_keepdim(1)?;
    let d = x.broadcast_sub(&mean)?;
    let var = d.sqr()?.mean_keepdim(1)?;
    Ok(d.broadcast_div(&var.affine(1.0, eps as f64)?.sqrt()?)?)
}

fn softmax(x: &Tensor, dim: usize) -> candle_core::Result<Tensor> {
    let m = x.max_keepdim(dim)?;
    let e = x.broadcast_sub(&m)?.exp()?;
    Ok(e.broadcast_div(&e.sum_keepdim(dim)?)?)
}

fn main() -> anyhow::Result<()> {
    let dev = Device::new_vulkan(0)?;
    let (b, s, h, hs, ff) = (3usize, 13usize, 12usize, 768usize, 3072usize);
    let d = hs / h;
    let x = Tensor::ones((b, s, hs), DType::F32, &dev)?;
    let wqkv = Tensor::ones((hs, 3*hs), DType::F32, &dev)?;
    let wo = Tensor::ones((hs, hs), DType::F32, &dev)?;
    let wf1 = Tensor::ones((hs, ff), DType::F32, &dev)?;
    let wf2 = Tensor::ones((ff, hs), DType::F32, &dev)?;

    bench("1x QKV mm (39x768x2304)", &dev, || x.reshape((b*s, hs))?.matmul(&wqkv), 100);
    bench("1x out mm (39x768x768)", &dev, || x.reshape((b*s, hs))?.matmul(&wo), 100);
    bench("1x ffn1 mm (39x768x3072)", &dev, || x.reshape((b*s, hs))?.matmul(&wf1), 100);
    bench("1x ffn2 mm (39x3072x768)", &dev, || {
        let f = Tensor::ones((b*s, ff), DType::F32, &dev)?; f.matmul(&wf2)
    }, 100);

    // full layer
    let mut layer = || -> candle_core::Result<Tensor> {
        let xn = layer_norm(&x, 1e-5)?;
        let x2d = xn.reshape((b*s, hs))?;
        let qkv = x2d.matmul(&wqkv)?;
        let q = qkv.narrow(1, 0, hs)?.reshape((b, s, h, d))?.transpose(1, 2)?.contiguous()?;
        let k = qkv.narrow(1, hs, hs)?.reshape((b, s, h, d))?.transpose(1, 2)?.contiguous()?;
        let v = qkv.narrow(1, 2*hs, hs)?.reshape((b, s, h, d))?.transpose(1, 2)?.contiguous()?;
        let att = q.matmul(&k.transpose(2, 3)?)?.affine(1.0/(d as f32).sqrt() as f64, 0.0)?;
        let att = softmax(&att, 3)?;
        let ctx = att.matmul(&v)?.transpose(1, 2)?.contiguous()?.reshape((b, s, hs))?;
        let out = ctx.reshape((b*s, hs))?.matmul(&wo)?;
        let h1 = x.reshape((b*s, hs))?.add(&out)?;
        let hn = layer_norm(&h1, 1e-5)?;
        let ff1 = hn.matmul(&wf1)?.gelu()?;
        let ff2 = ff1.matmul(&wf2)?;
        h1.add(&ff2)?.reshape((b, s, hs))
    };
    bench("FULL transformer layer", &dev, layer, 30);
    println!("=> x12 layers = {:?}", { let t=std::time::Instant::now(); for _ in 0..12 { layer().unwrap(); } dev.synchronize().unwrap(); t.elapsed() });
    Ok(())
}
