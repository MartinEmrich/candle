//! M0 smoke test for the candle Vulkan backend.
//!
//! Run on a machine with a Vulkan-capable GPU (e.g. RX 580 / RADV):
//! `cargo run --release --features vulkan --example vulkan_basics`
//!
//! Exercises: device creation, H2D/D2H round-trip, zeros (vkCmdFillBuffer),
//! ones (fill compute shader dispatch), f16 storage round-trip.
//! Only touches ops implemented in M0 (no reduce/matmul).

use candle_core::{DType, Device, Tensor};

fn main() -> anyhow::Result<()> {
    let device = Device::new_vulkan(0)?;
    println!("Using device: {:?}", device);

    // H2D / D2H round-trip
    let data: Vec<f32> = (0..1024 * 1024).map(|i| i as f32 * 0.5).collect();
    let t = Tensor::from_vec(data.clone(), (1024 * 1024,), &device)?;
    println!("uploaded {} f32 elements", data.len());
    let back = t.to_vec1::<f32>()?;
    anyhow::ensure!(back == data, "round-trip mismatch");
    println!("round-trip ok");

    // zeros via vkCmdFillBuffer
    let n = 4096usize;
    let z = Tensor::zeros((n,), DType::F32, &device)?.to_vec1::<f32>()?;
    anyhow::ensure!(z.iter().all(|v| *v == 0.0), "zeros broken");
    println!("zeros ok ({n} f32)");

    // ones via fill compute shader dispatch
    let o = Tensor::ones((n,), DType::F32, &device)?.to_vec1::<f32>()?;
    anyhow::ensure!(o.iter().all(|v| *v == 1.0), "ones broken");
    println!("ones ok ({n} f32, fill shader dispatched)");

    // f16 storage round-trip
    let m = 65536usize;
    let h = Tensor::from_vec(vec![half::f16::from_f32(1.5); m], (m,), &device)?;
    let hback = h.to_vec1::<half::f16>()?;
    anyhow::ensure!(
        hback.iter().all(|v| v.to_f32() == 1.5),
        "f16 round-trip broken"
    );
    println!("f16 round-trip ok ({m} f16)");

    // integer dtypes through fill/staging paths
    let u = Tensor::ones((n / 3,), DType::U32, &device)?.to_vec1::<u32>()?;
    anyhow::ensure!(u.iter().all(|v| *v == 1), "u32 ones broken");
    let i = Tensor::zeros((n / 8,), DType::I64, &device)?.to_vec1::<i64>()?;
    anyhow::ensure!(i.iter().all(|v| *v == 0), "i64 zeros broken");
    println!("u32/i64 ok");

    // affine: y = x * 2 + 3, contiguous and strided (transposed) source
    let ref_cpu = Device::Cpu;
    let x = Tensor::from_vec((0..4096).map(|i| i as f32 * 0.25).collect::<Vec<f32>>(), (64, 64), &device)?;
    let y = x.affine(2.0, 3.0)?.to_vec2::<f32>()?;
    let y_ref = Tensor::from_vec((0..4096).map(|i| i as f32 * 0.25).collect::<Vec<f32>>(), (64, 64), &ref_cpu)?.affine(2.0, 3.0)?.to_vec2::<f32>()?;
    anyhow::ensure!(y == y_ref, "affine mismatch (contiguous)");
    let xt = x.t()?; // strided view
    let yt = xt.affine(0.5, -1.0)?.to_vec2::<f32>()?;
    let yt_ref = Tensor::from_vec((0..4096).map(|i| i as f32 * 0.25).collect::<Vec<f32>>(), (64, 64), &ref_cpu)?.t()?.affine(0.5, -1.0)?.to_vec2::<f32>()?;
    anyhow::ensure!(yt == yt_ref, "affine mismatch (strided)");
    println!("affine ok (contiguous + strided)");

    // binary ops: add, mul, div, broadcast row vector, transposed lhs
    let mk = |dev: &Device| -> anyhow::Result<Tensor> {
        Ok(Tensor::from_vec((0..3072).map(|i| (i as f32 * 0.125).sin()).collect::<Vec<f32>>(), (48, 64), dev)?)
    };
    let (dv, dc) = (&device, &ref_cpu);
    let (a_v, b_v) = (mk(dv)?, mk(dv)?);
    let (a_c, b_c) = (mk(dc)?, mk(dc)?);
    let row_v = Tensor::from_vec((0..64).map(|i| 1.0 + i as f32).collect::<Vec<f32>>(), (1, 64), dv)?;
    let row_c = Tensor::from_vec((0..64).map(|i| 1.0 + i as f32).collect::<Vec<f32>>(), (1, 64), dc)?;
    let checks: Vec<(&str, Tensor, Tensor)> = vec![
        ("add", a_v.broadcast_add(&b_v)?, a_c.broadcast_add(&b_c)?),
        ("mul", a_v.mul(&b_v)?, a_c.mul(&b_c)?),
        ("div", a_v.div(&b_v.affine(1.0, 3.0)?)?, a_c.div(&b_c.affine(1.0, 3.0)?)?),
        ("broadcast_add", a_v.broadcast_add(&row_v)?, a_c.broadcast_add(&row_c)?),
        ("broadcast_mul", a_v.broadcast_mul(&row_v)?, a_c.broadcast_mul(&row_c)?),
        ("transposed_add", a_v.t()?.broadcast_add(&row_v.t()?)?, a_c.t()?.broadcast_add(&row_c.t()?)?),
        ("max", a_v.maximum(&b_v)?, a_c.maximum(&b_c)?),
        ("min", a_v.minimum(&b_v)?, a_c.minimum(&b_c)?),
    ];
    for (name, got, want) in checks {
        let g = got.to_vec2::<f32>()?;
        let w = want.to_vec2::<f32>()?;
        let mut max_diff = 0f32;
        for (gr, wr) in g.iter().zip(w.iter()) {
            for (gv, wv) in gr.iter().zip(wr.iter()) {
                let d = (gv - wv).abs();
                anyhow::ensure!(!(d.is_nan() || d.is_infinite()), "binary {name}: nan/inf at {gv} vs {wv}");
                if d > max_diff {
                    max_diff = d;
                }
            }
        }
        anyhow::ensure!(max_diff < 1e-6, "binary {name} mismatch, max_diff {max_diff}");
    }
    println!("binary ops ok (add/mul/div/broadcast/strided/min/max)");

    let pos_v = a_v.affine(1.0, 2.0)?;
    let pos_c = a_c.affine(1.0, 2.0)?;
    let unary_checks: Vec<(&str, Tensor, Tensor)> = vec![
        ("exp", a_v.exp()?, a_c.exp()?),
        ("log", pos_v.log()?, pos_c.log()?),
        ("sin", a_v.sin()?, a_c.sin()?),
        ("cos", a_v.cos()?, a_c.cos()?),
        ("tanh", a_v.tanh()?, a_c.tanh()?),
        ("neg", a_v.neg()?, a_c.neg()?),
        ("sqr", a_v.sqr()?, a_c.sqr()?),
        ("sqrt", pos_v.sqrt()?, pos_c.sqrt()?),
        ("abs", a_v.abs()?, a_c.abs()?),
        ("ceil", a_v.ceil()?, a_c.ceil()?),
        ("floor", a_v.floor()?, a_c.floor()?),
        ("round", a_v.round()?, a_c.round()?),
        ("relu", a_v.relu()?, a_c.relu()?),
        ("sign", a_v.sign()?, a_c.sign()?),
        ("erf", a_v.erf()?, a_c.erf()?),
        ("gelu", a_v.gelu()?, a_c.gelu()?),
        ("gelu_erf", a_v.gelu_erf()?, a_c.gelu_erf()?),
        ("silu", a_v.silu()?, a_c.silu()?),
        ("recip", pos_v.recip()?, pos_c.recip()?),
    ];
    for (name, got, want) in unary_checks {
        let g = got.to_vec2::<f32>()?;
        let w = want.to_vec2::<f32>()?;
        let mut max_diff = 0f32;
        for (gr, wr) in g.iter().zip(w.iter()) {
            for (gv, wv) in gr.iter().zip(wr.iter()) {
                let d = (gv - wv).abs();
                anyhow::ensure!(!(d.is_nan() || d.is_infinite()), "unary {name}: nan/inf at {gv} vs {wv}");
                if d > max_diff {
                    max_diff = d;
                }
            }
        }
        let tol = if matches!(name, "neg" | "abs" | "ceil" | "floor" | "round" | "relu" | "sign" | "sqr") { 1e-7 } else { 1e-6 };
        anyhow::ensure!(max_diff < tol, "unary {name} mismatch, max_diff {max_diff}");
    }
    println!("unary ops ok (19 ops incl. erf/gelu/gelu_erf/silu)");

    // reductions
    let r_checks: Vec<(&str, Tensor, Tensor)> = vec![
        ("sum_all", a_v.sum_all()?, a_c.sum_all()?),
        ("sum_d1", a_v.sum(1)?, a_c.sum(1)?),
        ("sum_d0", a_v.sum(0)?, a_c.sum(0)?),
        ("max_d1", a_v.max(1)?, a_c.max(1)?),
        ("min_d0", a_v.min(0)?, a_c.min(0)?),
        ("mean_d1", a_v.mean(1)?, a_c.mean(1)?),
        ("sum_keepdim_d1", a_v.sum_keepdim(1)?, a_c.sum_keepdim(1)?),
    ];
    for (name, got, want) in r_checks {
        let g = got.flatten_all()?.to_vec1::<f32>()?;
        let w = want.flatten_all()?.to_vec1::<f32>()?;
        let md = g.iter().zip(w.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        anyhow::ensure!(md < 1e-4, "reduce {name} mismatch, max_diff {md}");
    }
    println!("reductions ok (sum/max/min/mean/keepdim)");

    // softmax (composed) + layer_norm-style (composed)
    let sm_v = {
        let mx = a_v.max_keepdim(1)?;
        let d = a_v.broadcast_sub(&mx)?.exp()?;
        d.broadcast_div(&d.sum_keepdim(1)?)?
    };
    let sm_c = {
        let mx = a_c.max_keepdim(1)?;
        let d = a_c.broadcast_sub(&mx)?.exp()?;
        d.broadcast_div(&d.sum_keepdim(1)?)?
    };
    let md = sm_v
        .flatten_all()?
        .to_vec1::<f32>()?
        .iter()
        .zip(sm_c.flatten_all()?.to_vec1::<f32>()?)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    anyhow::ensure!(md < 1e-6, "softmax mismatch {md}");
    let mean_v = a_v.mean_keepdim(1)?;
    let var_v = a_v.broadcast_sub(&mean_v)?.sqr()?.mean_keepdim(1)?;
    let ln_v = a_v.broadcast_sub(&mean_v)?.broadcast_div(&var_v.affine(1.0, 1e-5)?.sqrt()?)?;
    let mean_c = a_c.mean_keepdim(1)?;
    let var_c = a_c.broadcast_sub(&mean_c)?.sqr()?.mean_keepdim(1)?;
    let ln_c = a_c.broadcast_sub(&mean_c)?.broadcast_div(&var_c.affine(1.0, 1e-5)?.sqrt()?)?;
    let md = ln_v
        .flatten_all()?
        .to_vec1::<f32>()?
        .iter()
        .zip(ln_c.flatten_all()?.to_vec1::<f32>()?)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    anyhow::ensure!(md < 1e-5, "layer_norm-style mismatch {md}");
    println!("softmax + layer_norm-style ok");

    // cmp / where_cond / bool casts
    let cond_v = a_v.gt(0.0)?;
    let cond_c = a_c.gt(0.0)?;
    let ones_v = Tensor::ones_like(&a_v)?;
    let ones_c = Tensor::ones_like(&a_c)?;
    let wc_v = cond_v.where_cond(&ones_v, &a_v.affine(-1.0, 0.0)?)?;
    let wc_c = cond_c.where_cond(&ones_c, &a_c.affine(-1.0, 0.0)?)?;
    let md = wc_v
        .flatten_all()?
        .to_vec1::<f32>()?
        .iter()
        .zip(wc_c.flatten_all()?.to_vec1::<f32>()?)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    anyhow::ensure!(md == 0.0, "where_cond mismatch {md}");
    let b2f_v = cond_v.to_dtype(DType::F32)?;
    let b2f_c = cond_c.to_dtype(DType::F32)?;
    anyhow::ensure!(
        b2f_v.flatten_all()?.to_vec1::<f32>()? == b2f_c.flatten_all()?.to_vec1::<f32>()?,
        "bool->f32 mismatch"
    );
    println!("cmp / where_cond / bool casts ok");

    // index_select (embedding)
    let emb_v = Tensor::from_vec(
        (0..1600).map(|i| (i as f32 * 0.01).cos()).collect::<Vec<f32>>(),
        (100, 16),
        &device,
    )?;
    let emb_c = Tensor::from_vec(
        (0..1600).map(|i| (i as f32 * 0.01).cos()).collect::<Vec<f32>>(),
        (100, 16),
        &ref_cpu,
    )?;
    let ids = Tensor::new(&[0i64, 5, 5, 99, 3], &device)?;
    let ids_c = Tensor::new(&[0i64, 5, 5, 99, 3], &ref_cpu)?;
    let sel_v = emb_v.index_select(&ids, 0)?;
    let sel_c = emb_c.index_select(&ids_c, 0)?;
    anyhow::ensure!(
        sel_v.to_vec2::<f32>()? == sel_c.to_vec2::<f32>()?,
        "index_select mismatch"
    );
    println!("index_select ok (i64 ids, dup entries)");

    // matmul: plain, batched, transposed lhs
    let w_v = Tensor::from_vec((0..2048).map(|i| (i as f32 * 0.003).sin()).collect::<Vec<f32>>(), (64, 32), &device)?;
    let w_c = Tensor::from_vec((0..2048).map(|i| (i as f32 * 0.003).sin()).collect::<Vec<f32>>(), (64, 32), &ref_cpu)?;
    let mm_v = a_v.matmul(&w_v)?;
    let mm_c = a_c.matmul(&w_c)?;
    let md = mm_v
        .flatten_all()?
        .to_vec1::<f32>()?
        .iter()
        .zip(mm_c.flatten_all()?.to_vec1::<f32>()?)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    anyhow::ensure!(md < 1e-3, "matmul mismatch {md}");
    let at_v = a_v.t()?;
    let at_c = a_c.t()?;
    let w2_v = Tensor::from_vec((0..1536).map(|i| (i as f32 * 0.002).cos()).collect::<Vec<f32>>(), (48, 32), &device)?;
    let w2_c = Tensor::from_vec((0..1536).map(|i| (i as f32 * 0.002).cos()).collect::<Vec<f32>>(), (48, 32), &ref_cpu)?;
    let mmt_v = at_v.matmul(&w2_v)?;
    let mmt_c = at_c.matmul(&w2_c)?;
    let md = mmt_v
        .flatten_all()?
        .to_vec1::<f32>()?
        .iter()
        .zip(mmt_c.flatten_all()?.to_vec1::<f32>()?)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    anyhow::ensure!(md < 1e-3, "matmul(transposed) mismatch {md}");
    let b3_v = Tensor::from_vec((0..3 * 8 * 16).map(|i| (i as f32 * 0.011).sin()).collect::<Vec<f32>>(), (3, 8, 16), &device)?;
    let b3_c = Tensor::from_vec((0..3 * 8 * 16).map(|i| (i as f32 * 0.011).sin()).collect::<Vec<f32>>(), (3, 8, 16), &ref_cpu)?;
    let w3_v = Tensor::from_vec((0..3 * 16 * 4).map(|i| (i as f32 * 0.021).cos()).collect::<Vec<f32>>(), (3, 16, 4), &device)?;
    let w3_c = Tensor::from_vec((0..3 * 16 * 4).map(|i| (i as f32 * 0.021).cos()).collect::<Vec<f32>>(), (3, 16, 4), &ref_cpu)?;
    let mmb_v = b3_v.matmul(&w3_v)?;
    let mmb_c = b3_c.matmul(&w3_c)?;
    let md = mmb_v
        .flatten_all()?
        .to_vec1::<f32>()?
        .iter()
        .zip(mmb_c.flatten_all()?.to_vec1::<f32>()?)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    anyhow::ensure!(md < 1e-4, "matmul(batched) mismatch {md}");
    println!("matmul ok (plain/transposed/batched)");

    // casts + contiguous
    let h_v = a_v.to_dtype(DType::F16)?;
    let back_v = h_v.to_dtype(DType::F32)?;
    let md = back_v
        .flatten_all()?
        .to_vec1::<f32>()?
        .iter()
        .zip(a_v.flatten_all()?.to_vec1::<f32>()?)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    anyhow::ensure!(md < 1e-3, "f16 roundtrip mismatch {md}");
    let tc_v = a_v.t()?.contiguous()?;
    let tc_c = a_c.t()?.contiguous()?;
    anyhow::ensure!(
        tc_v.to_vec2::<f32>()? == tc_c.to_vec2::<f32>()?,
        "contiguous mismatch"
    );
    println!("casts + contiguous ok");

    // attention-bias style chain: (b,1,1,s) bias with -inf, broadcast to
    // (b,h,s,s), contiguous, add to scores, softmax on last dim.
    {
        use candle_core::D;
        let b = 2usize;
        let h = 12usize;
        let s = 13usize;
        let bias_data: Vec<f32> = (0..b)
            .flat_map(|bi| {
                let len = if bi == 0 { 6 } else { 10 };
                (0..s).map(move |j| if j < len { 0.0 } else { f32::NEG_INFINITY })
            })
            .collect();
        let scores_data: Vec<f32> = (0..b * h * s * s)
            .map(|i| ((i as f64 * 0.37).sin()) as f32)
            .collect();
        let run = |dev: &Device| -> Result<Tensor, candle_core::Error> {
            let bias = Tensor::from_vec(bias_data.clone(), (b, 1, 1, s), dev)?
                .broadcast_as((b, h, s, s))?
                .contiguous()?;
            let scores = Tensor::from_vec(scores_data.clone(), (b, h, s, s), dev)?;
            let scores = scores.broadcast_add(&bias)?;
            let mx = scores.max_keepdim(D::Minus1)?;
            let e = scores.broadcast_sub(&mx)?.exp()?;
            e.broadcast_div(&e.sum_keepdim(D::Minus1)?)
        };
        let gv = run(&device)?.flatten_all()?.to_vec1::<f32>()?;
        let gc = run(&ref_cpu)?.flatten_all()?.to_vec1::<f32>()?;
        let md = gv
            .iter()
            .zip(gc.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        anyhow::ensure!(md < 1e-6, "attn-bias chain mismatch {md}");
        println!("attn-bias chain ok (broadcast -inf bias + softmax)");
    }

    println!("M0 smoke test passed");
    Ok(())
}
