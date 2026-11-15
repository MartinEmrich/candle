//! Variable-length flash attention for the candle Vulkan backend.
//!
//! Mirrors the `candle-flash-attn` variable-length API so that TEI's
//! `flash_attn.rs` can dispatch to it. The compute runs a SIMT GLSL kernel
//! (fp32) supporting causal / sliding-window masking, ALiBi slopes and
//! logit soft-capping.

use candle::{
    CpuStorage, CustomOp3, Layout, Result, Shape, Storage, Tensor, VulkanStorage,
};

struct FlashAttnVarLen {
    softmax_scale: f32,
    seqlens_q: Tensor,
    seqlens_k: Tensor,
    alibi_slopes: Option<Tensor>,
    window_size_left: Option<usize>,
    window_size_right: Option<usize>,
    softcap: Option<f32>,
}

fn vulkan_storage<'a>(s: &'a Storage, what: &str) -> Result<&'a VulkanStorage> {
    match s {
        Storage::Vulkan(s) => Ok(s),
        _ => candle::bail!("flash-attn-vulkan: {what} must be a vulkan tensor"),
    }
}

impl CustomOp3 for FlashAttnVarLen {
    fn name(&self) -> &'static str {
        "flash-attn-varlen-vulkan"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle::bail!("flash-attn-vulkan: cpu forward is not implemented")
    }

    fn vulkan_fwd(
        &self,
        q: &VulkanStorage,
        q_l: &Layout,
        k: &VulkanStorage,
        k_l: &Layout,
        v: &VulkanStorage,
        v_l: &Layout,
    ) -> Result<(VulkanStorage, Shape)> {
        let sq_pair = self.seqlens_q.storage_and_layout();
        let sk_pair = self.seqlens_k.storage_and_layout();
        let al_pair = self.alibi_slopes.as_ref().map(|t| t.storage_and_layout());
        let sq = vulkan_storage(&sq_pair.0, "seqlens_q")?;
        let sk = vulkan_storage(&sk_pair.0, "seqlens_k")?;
        let alibi = match &al_pair {
            Some((arc, layout)) => Some((vulkan_storage(arc, "alibi_slopes")?, *layout)),
            None => None,
        };
        let out = candle::vulkan_backend::flash_attn_varlen(
            q,
            q_l,
            k,
            k_l,
            v,
            v_l,
            sq,
            sq_pair.1,
            sk,
            sk_pair.1,
            alibi,
            self.softmax_scale,
            self.window_size_left,
            self.window_size_right,
            self.softcap,
        )?;
        Ok((out, q_l.shape().clone()))
    }
}

/// Variable-length flash attention with sliding-window / causal masking.
///
/// * `q` - `[total_q, num_heads_q, head_size]`
/// * `k`, `v` - `[total_kv, num_heads_kv, head_size]`
/// * `seqlens_q`, `seqlens_k` - cumulative lengths, `[batch + 1]` `u32`
///
/// `window_size_right = Some(0)` with `window_size_left = None` applies a
/// causal mask.
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_varlen_windowed(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    seqlens_q: &Tensor,
    seqlens_k: &Tensor,
    _max_seqlen_q: usize,
    _max_seqlen_k: usize,
    softmax_scale: f32,
    window_size_left: Option<usize>,
    window_size_right: Option<usize>,
) -> Result<Tensor> {
    let op = FlashAttnVarLen {
        softmax_scale,
        seqlens_q: seqlens_q.clone(),
        seqlens_k: seqlens_k.clone(),
        alibi_slopes: None,
        window_size_left,
        window_size_right,
        softcap: None,
    };
    let q = q.contiguous()?;
    let k = k.contiguous()?;
    let v = v.contiguous()?;
    q.apply_op3(&k, &v, op)
}

/// As [`flash_attn_varlen_windowed`] plus ALiBi slopes (`[num_heads_q]` f32)
/// and optional logit soft-capping.
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_varlen_alibi_windowed_softcap(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    alibi_slopes: &Tensor,
    seqlens_q: &Tensor,
    seqlens_k: &Tensor,
    _max_seqlen_q: usize,
    _max_seqlen_k: usize,
    softmax_scale: f32,
    window_size_left: Option<usize>,
    window_size_right: Option<usize>,
    softcap: Option<f32>,
) -> Result<Tensor> {
    let op = FlashAttnVarLen {
        softmax_scale,
        seqlens_q: seqlens_q.clone(),
        seqlens_k: seqlens_k.clone(),
        alibi_slopes: Some(alibi_slopes.clone()),
        window_size_left,
        window_size_right,
        softcap,
    };
    let q = q.contiguous()?;
    let k = k.contiguous()?;
    let v = v.contiguous()?;
    q.apply_op3(&k, &v, op)
}

/// As [`flash_attn_varlen_windowed`] plus ALiBi slopes (`[num_heads_q]` f32).
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_varlen_alibi_windowed(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    alibi_slopes: &Tensor,
    seqlens_q: &Tensor,
    seqlens_k: &Tensor,
    max_seqlen_q: usize,
    max_seqlen_k: usize,
    softmax_scale: f32,
    window_size_left: Option<usize>,
    window_size_right: Option<usize>,
) -> Result<Tensor> {
    flash_attn_varlen_alibi_windowed_softcap(
        q,
        k,
        v,
        alibi_slopes,
        seqlens_q,
        seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        window_size_left,
        window_size_right,
        None,
    )
}

/// Variable-length flash attention with an optional causal mask.
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_varlen(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    seqlens_q: &Tensor,
    seqlens_k: &Tensor,
    max_seqlen_q: usize,
    max_seqlen_k: usize,
    softmax_scale: f32,
    causal: bool,
) -> Result<Tensor> {
    let window_size_right = if causal { Some(0) } else { None };
    flash_attn_varlen_windowed(
        q,
        k,
        v,
        seqlens_q,
        seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        None,
        window_size_right,
    )
}
