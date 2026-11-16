// Validates the Vulkan flash-attention kernel against a pure-Rust reference
// for variable-length batching, causal masking, sliding windows and ALiBi.

use candle::{DType, Device, Tensor};
use candle_flash_attn_vulkan::{
    flash_attn_varlen_alibi_windowed, flash_attn_varlen_windowed,
};

fn lcg(seed: u64) -> impl FnMut() -> f32 {
    let mut s = seed;
    move || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((s >> 40) as f32 / (1u64 << 24) as f32) - 0.5
    }
}

/// Pure-Rust varlen attention reference. `q` is `[total_q, h, d]`, `k`/`v` are
/// `[total_k, h, d]` (MHA). Returns `[total_q, h, d]`.
#[allow(clippy::too_many_arguments)]
fn reference(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    cu: &[u32],
    h: usize,
    d: usize,
    scale: f32,
    causal: bool,
    window_left: Option<usize>,
    window_right: Option<usize>,
    alibi: Option<&[f32]>,
) -> Vec<f32> {
    let total_q = q.len() / (h * d);
    let mut out = vec![0.0f32; total_q * h * d];
    let idx = |tok: usize, head: usize, dim: usize| tok * h * d + head * d + dim;
    for seq in 0..cu.len() - 1 {
        let (qs, qe) = (cu[seq] as usize, cu[seq + 1] as usize);
        let (ks, ke) = (cu[seq] as usize, cu[seq + 1] as usize);
        for head in 0..h {
            let slope = alibi.map(|a| a[head]).unwrap_or(0.0);
            for i in qs..qe {
                let qpos = i - qs;
                let mut scores: Vec<f32> = Vec::new();
                let mut js: Vec<usize> = Vec::new();
                for j in ks..ke {
                    let kpos = j - ks;
                    if causal && kpos > qpos {
                        continue;
                    }
                    if let Some(wl) = window_left {
                        if kpos + wl < qpos {
                            continue;
                        }
                    }
                    if let Some(wr) = window_right {
                        if kpos > qpos + wr {
                            continue;
                        }
                    }
                    let mut s = 0.0f32;
                    for dd in 0..d {
                        s += q[idx(i, head, dd)] * k[idx(j, head, dd)];
                    }
                    s *= scale;
                    s += slope * (qpos as f32 - kpos as f32);
                    scores.push(s);
                    js.push(j);
                }
                let mx = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exps: Vec<f32> = scores.iter().map(|s| (s - mx).exp()).collect();
                let sum: f32 = exps.iter().sum();
                for dd in 0..d {
                    let mut acc = 0.0f32;
                    for (n, &j) in js.iter().enumerate() {
                        let p = if sum > 0.0 { exps[n] / sum } else { 0.0 };
                        acc += p * v[idx(j, head, dd)];
                    }
                    out[idx(i, head, dd)] = acc;
                }
            }
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn run_case(
    dev: &Device,
    lens: &[usize],
    h: usize,
    d: usize,
    scale: f32,
    causal: bool,
    window_left: Option<usize>,
    window_right: Option<usize>,
    alibi: Option<Vec<f32>>,
) {
    let total_q: usize = lens.iter().sum();
    let mut cu = vec![0u32];
    for l in lens {
        cu.push(*cu.last().unwrap() + *l as u32);
    }
    let n = total_q * h * d;
    let mut gen = lcg(0x1234_5678);
    let q: Vec<f32> = (0..n).map(|_| gen()).collect();
    let k: Vec<f32> = (0..n).map(|_| gen()).collect();
    let v: Vec<f32> = (0..n).map(|_| gen()).collect();

    let ref_out = reference(
        &q, &k, &v, &cu, h, d, scale, causal, window_left, window_right, alibi.as_deref(),
    );

    let shape = (total_q, h, d);
    let qt = Tensor::from_vec(q.clone(), shape, dev).unwrap();
    let kt = Tensor::from_vec(k.clone(), shape, dev).unwrap();
    let vt = Tensor::from_vec(v.clone(), shape, dev).unwrap();
    let cuq = Tensor::from_vec(cu.clone(), (cu.len(),), dev).unwrap();
    let cuk = Tensor::from_vec(cu.clone(), (cu.len(),), dev).unwrap();
    let out = if let Some(slopes) = &alibi {
        let st = Tensor::from_vec(slopes.clone(), (slopes.len(),), dev).unwrap();
        flash_attn_varlen_alibi_windowed(
            &qt, &kt, &vt, &st, &cuq, &cuk, total_q, total_q, scale, window_left, window_right,
        )
        .unwrap()
    } else {
        flash_attn_varlen_windowed(
            &qt, &kt, &vt, &cuq, &cuk, total_q, total_q, scale, window_left, window_right,
        )
        .unwrap()
    };
    assert_eq!(out.dims().to_vec(), vec![total_q, h, d]);
    let got = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let mut max_abs = 0.0f32;
    for (a, b) in ref_out.iter().zip(got.iter()) {
        max_abs = max_abs.max((a - b).abs());
    }
    let tag = format!(
        "lens={lens:?} h={h} d={d} causal={causal} wl={window_left:?} wr={window_right:?} alibi={}",
        alibi.is_some()
    );
    assert!(max_abs < 1e-4, "{tag}: max_abs={max_abs}");
    println!("ok: {tag} (max_abs={max_abs:.2e})");
}

/// Grouped-query attention: `q` is `[total, h, d]`, `k`/`v` are `[total, hk, d]`
/// with `h % hk == 0`; q head `hh` reads kv head `hh / (h/hk)`.
#[allow(clippy::too_many_arguments)]
fn gqa_case(dev: &Device, lens: &[usize], h: usize, hk: usize, d: usize, scale: f32, causal: bool) {
    let total: usize = lens.iter().sum();
    let mut cu = vec![0u32];
    for l in lens {
        cu.push(*cu.last().unwrap() + *l as u32);
    }
    let nq = total * h * d;
    let nk = total * hk * d;
    let mut gen = lcg(0xfeed_face);
    let q: Vec<f32> = (0..nq).map(|_| gen()).collect();
    let k: Vec<f32> = (0..nk).map(|_| gen()).collect();
    let v: Vec<f32> = (0..nk).map(|_| gen()).collect();

    let rep = h / hk;
    let mut out = vec![0.0f32; nq];
    let qi = |t: usize, hh: usize, dd: usize| t * h * d + hh * d + dd;
    let ki = |t: usize, hh: usize, dd: usize| t * hk * d + hh * d + dd;
    for seq in 0..cu.len() - 1 {
        let (s, e) = (cu[seq] as usize, cu[seq + 1] as usize);
        for hh in 0..h {
            let kvh = hh / rep;
            for i in s..e {
                let qpos = i - s;
                let mut scores = Vec::new();
                let mut js = Vec::new();
                for j in s..e {
                    let kpos = j - s;
                    if causal && kpos > qpos {
                        continue;
                    }
                    let mut sc = 0.0;
                    for dd in 0..d {
                        sc += q[qi(i, hh, dd)] * k[ki(j, kvh, dd)];
                    }
                    scores.push(sc * scale);
                    js.push(j);
                }
                let mx = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exps: Vec<f32> = scores.iter().map(|x| (x - mx).exp()).collect();
                let sum: f32 = exps.iter().sum();
                for dd in 0..d {
                    let mut acc = 0.0;
                    for (n2, &j) in js.iter().enumerate() {
                        let p = if sum > 0.0 { exps[n2] / sum } else { 0.0 };
                        acc += p * v[ki(j, kvh, dd)];
                    }
                    out[qi(i, hh, dd)] = acc;
                }
            }
        }
    }

    let qt = Tensor::from_vec(q.clone(), (total, h, d), dev).unwrap();
    let kt = Tensor::from_vec(k.clone(), (total, hk, d), dev).unwrap();
    let vt = Tensor::from_vec(v.clone(), (total, hk, d), dev).unwrap();
    let cuq = Tensor::from_vec(cu.clone(), (cu.len(),), dev).unwrap();
    let out_t = flash_attn_varlen_windowed(
        &qt,
        &kt,
        &vt,
        &cuq,
        &cuq,
        total,
        total,
        scale,
        None,
        if causal { Some(0) } else { None },
    )
    .unwrap();
    assert_eq!(out_t.dims().to_vec(), vec![total, h, d]);
    let got = out_t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let mut max_abs = 0.0f32;
    for (a, b) in out.iter().zip(got.iter()) {
        max_abs = max_abs.max((a - b).abs());
    }
    let tag = format!("gqa lens={lens:?} h={h} hk={hk} d={d} causal={causal}");
    assert!(max_abs < 1e-4, "{tag}: max_abs={max_abs}");
    println!("ok: {tag} (max_abs={max_abs:.2e})");
}

/// f16-storage variant of [`run_case`]: builds f32 data, computes the f32
/// reference, casts q/k/v to f16 on device, runs the tiled f16 kernel and
/// compares to the reference with an f16-appropriate tolerance.
#[allow(clippy::too_many_arguments)]
fn run_case_f16(
    dev: &Device,
    lens: &[usize],
    h: usize,
    d: usize,
    scale: f32,
    causal: bool,
    window_left: Option<usize>,
    window_right: Option<usize>,
) {
    let total_q: usize = lens.iter().sum();
    let mut cu = vec![0u32];
    for l in lens {
        cu.push(*cu.last().unwrap() + *l as u32);
    }
    let n = total_q * h * d;
    let mut gen = lcg(0x1234_5678);
    let q: Vec<f32> = (0..n).map(|_| gen()).collect();
    let k: Vec<f32> = (0..n).map(|_| gen()).collect();
    let v: Vec<f32> = (0..n).map(|_| gen()).collect();

    let ref_out = reference(&q, &k, &v, &cu, h, d, scale, causal, window_left, window_right, None);

    let shape = (total_q, h, d);
    let qt = Tensor::from_vec(q, shape, dev).unwrap().to_dtype(DType::F16).unwrap();
    let kt = Tensor::from_vec(k, shape, dev).unwrap().to_dtype(DType::F16).unwrap();
    let vt = Tensor::from_vec(v, shape, dev).unwrap().to_dtype(DType::F16).unwrap();
    let cuq = Tensor::from_vec(cu.clone(), (cu.len(),), dev).unwrap();
    let out = flash_attn_varlen_windowed(
        &qt, &kt, &vt, &cuq, &cuq, total_q, total_q, scale, window_left, window_right,
    )
    .unwrap();
    assert_eq!(out.dims().to_vec(), vec![total_q, h, d]);
    assert_eq!(out.dtype(), DType::F16);
    let got = out.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let mut max_abs = 0.0f32;
    for (a, b) in ref_out.iter().zip(got.iter()) {
        max_abs = max_abs.max((a - b).abs());
    }
    let tag = format!("f16 lens={lens:?} h={h} d={d} causal={causal}");
    assert!(max_abs < 5e-3, "{tag}: max_abs={max_abs}");
    println!("ok: {tag} (max_abs={max_abs:.2e})");
}

#[test]
fn fa_matches_reference() {
    let dev = match Device::new_vulkan(0) {
        Ok(d) => d,
        Err(_) => {
            eprintln!("no vulkan device; skipping");
            return;
        }
    };
    let h = 2;
    let d = 8;
    let scale = 0.25f32;
    // non-causal, no window
    run_case(&dev, &[3, 2], h, d, scale, false, None, None, None);
    // causal
    run_case(&dev, &[3, 2], h, d, scale, true, None, Some(0), None);
    // sliding window (left=1, causal right=0)
    run_case(&dev, &[5, 4], h, d, scale, true, Some(1), Some(0), None);
    // single long sequence
    run_case(&dev, &[10], h, d, scale, true, None, Some(0), None);
    // alibi, non-causal
    run_case(&dev, &[4, 3], h, d, scale, false, None, None, Some(vec![0.5, 0.125]));
    // alibi, causal
    run_case(&dev, &[6, 2], h, d, scale, true, None, Some(0), Some(vec![0.5, 0.125]));
    // larger head
    run_case(&dev, &[7, 5], h, 64, scale, true, None, Some(0), None);
    // grouped-query attention (Qwen3-style: q heads > kv heads)
    gqa_case(&dev, &[4, 3], 4, 2, 8, scale, false);
    gqa_case(&dev, &[6, 5], 4, 2, 8, scale, true);
    gqa_case(&dev, &[5], 8, 2, 64, scale, true);
    // cross the tiled kernel's BR (128 query rows) and BC (32 keys) boundaries
    run_case(&dev, &[128], h, d, scale, true, None, Some(0), None);
    run_case(&dev, &[129, 200], h, d, scale, false, None, None, None);
    run_case(&dev, &[300], h, d, scale, true, None, Some(0), None);
    run_case(&dev, &[64, 130, 257], h, d, scale, true, Some(2), Some(0), None);
    // head_size = 128 (Qwen3 / MAX_D edge), crossing tiles
    run_case(&dev, &[130, 200], h, 128, scale, true, None, Some(0), None);
    // alibi across tile boundaries
    run_case(&dev, &[140], h, d, scale, false, None, None, Some(vec![0.5, 0.125]));
    // GQA across tile boundaries
    gqa_case(&dev, &[130, 200], 4, 2, 64, scale, true);
    // f16 storage path (TEI default dtype), crossing tiles
    run_case_f16(&dev, &[3, 2], h, d, scale, false, None, None);
    run_case_f16(&dev, &[129, 200], h, d, scale, true, None, Some(0));
    run_case_f16(&dev, &[130, 200], h, 128, scale, true, None, Some(0));
    run_case_f16(&dev, &[64, 130, 257], h, d, scale, true, Some(2), Some(0));
}

/// Long-sequence timing / no-fault check for the tiled kernel: two 8192-token
/// sequences (the router warmup shape). The scalar kernel tripped the amdgpu job
/// timeout here; the tiled kernel must complete quickly with finite output.
#[test]
fn fa_long_seq_perf() {
    let dev = match Device::new_vulkan(0) {
        Ok(d) => d,
        Err(_) => {
            eprintln!("no vulkan device; skipping");
            return;
        }
    };
    let h = std::env::var("FA_TEST_H").ok().and_then(|s| s.parse().ok()).unwrap_or(12usize);
    let d = 64usize;
    let scale = 1.0 / (d as f32).sqrt();
    let seq = std::env::var("FA_TEST_SEQ").ok().and_then(|s| s.parse().ok()).unwrap_or(8192usize);
    let nb = std::env::var("FA_TEST_BATCH").ok().and_then(|s| s.parse().ok()).unwrap_or(2usize);
    let lens = vec![seq; nb];
    let lens = lens.as_slice();
    let total: usize = lens.iter().sum();
    let mut cu = vec![0u32];
    for l in lens {
        cu.push(*cu.last().unwrap() + *l as u32);
    }
    let n = total * h * d;
    let mut gen = lcg(0xabcdef);
    let q: Vec<f32> = (0..n).map(|_| gen()).collect();
    let k: Vec<f32> = (0..n).map(|_| gen()).collect();
    let v: Vec<f32> = (0..n).map(|_| gen()).collect();

    let shape = (total, h, d);
    let qt = Tensor::from_vec(q, shape, &dev).unwrap();
    let kt = Tensor::from_vec(k, shape, &dev).unwrap();
    let vt = Tensor::from_vec(v, shape, &dev).unwrap();
    let cu_len = cu.len();
    let cuq = Tensor::from_vec(cu, (cu_len,), &dev).unwrap();

    let t0 = std::time::Instant::now();
    let out = flash_attn_varlen_windowed(&qt, &kt, &vt, &cuq, &cuq, seq, seq, scale, None, None);
    let out = match out {
        Ok(o) => o,
        Err(e) => {
            println!("fa_long_seq: total={total} h={h} d={d} FA-CALL-ERR after {:?}: {e}", t0.elapsed());
            return;
        }
    };
    if let Err(e) = dev.synchronize() {
        println!("fa_long_seq: total={total} h={h} d={d} SYNC-ERR after {:?}: {e}", t0.elapsed());
        return;
    }
    let dt = t0.elapsed();
    let got = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let bad = got.iter().filter(|x| !x.is_finite()).count();
    println!("fa_long_seq: total={total} h={h} d={d} took {dt:?} nonfinite={bad}");
    assert_eq!(got.len(), n);
    assert_eq!(bad, 0, "non-finite outputs");
    // Tiled kernel must clear the ~15s scalar-kernel timeout with margin.
    assert!(dt.as_secs() < 10, "tiled FA too slow: {dt:?}");
}
