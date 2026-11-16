//! SPIR-V compute kernels for the candle Vulkan backend.
//!
//! Shaders live as GLSL next to this file and are compiled to SPIR-V by
//! `build.rs` (glslc). Kernels ported from ggml-vulkan (MIT) keep their
//! attribution in the file header; see NOTICE.

// Kernels ported from ggml-vulkan (https://github.com/ggml-org/llama.cpp), MIT,
// Copyright (c) The ggml authors, will keep their attribution in the file header
// and be listed in NOTICE.

pub const FILL: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/fill.spv"));
pub const AFFINE_F32: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/affine_f32.spv"));
pub const AFFINE_F16: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/affine_f16.spv"));
pub const AFFINE_BF16: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/affine_bf16.spv"));
pub const ADD_F32: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/add_f32.spv"));
pub const ADD_F16: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/add_f16.spv"));
pub const SUB_F32: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/sub_f32.spv"));
pub const SUB_F16: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/sub_f16.spv"));
pub const ADD_U32: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/add_u32.spv"));
pub const SUB_U32: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/sub_u32.spv"));
pub const MUL_F32: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/mul_f32.spv"));
pub const MUL_F16: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/mul_f16.spv"));
pub const DIV_F32: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/div_f32.spv"));
pub const DIV_F16: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/div_f16.spv"));
pub const MINIMUM_F32: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/minimum_f32.spv"));
pub const MINIMUM_F16: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/minimum_f16.spv"));
pub const MAXIMUM_F32: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/maximum_f32.spv"));
pub const MAXIMUM_F16: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/maximum_f16.spv"));
pub const ADD_BF16: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/add_bf16.spv"));
pub const SUB_BF16: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/sub_bf16.spv"));
pub const MUL_BF16: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/mul_bf16.spv"));
pub const DIV_BF16: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/div_bf16.spv"));
pub const MINIMUM_BF16: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/minimum_bf16.spv"));
pub const MAXIMUM_BF16: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/maximum_bf16.spv"));

macro_rules! unary_kernel {
    ($name:ident, $file:literal) => {
        pub const $name: &[u8] =
            include_bytes!(concat!(env!("OUT_DIR"), "/", $file, ".spv"));
    };
}
unary_kernel!(EXP_F32, "exp_f32");
unary_kernel!(LOG_F32, "log_f32");
unary_kernel!(SIN_F32, "sin_f32");
unary_kernel!(COS_F32, "cos_f32");
unary_kernel!(TANH_F32, "tanh_f32");
unary_kernel!(NEG_F32, "neg_f32");
unary_kernel!(RECIP_F32, "recip_f32");
unary_kernel!(SQR_F32, "sqr_f32");
unary_kernel!(SQRT_F32, "sqrt_f32");
unary_kernel!(ABS_F32, "abs_f32");
unary_kernel!(CEIL_F32, "ceil_f32");
unary_kernel!(FLOOR_F32, "floor_f32");
unary_kernel!(ROUND_F32, "round_f32");
unary_kernel!(RELU_F32, "relu_f32");
unary_kernel!(SIGN_F32, "sign_f32");
unary_kernel!(ERF_F32, "erf_f32");
unary_kernel!(GELU_F32, "gelu_f32");
unary_kernel!(GELU_ERF_F32, "gelu_erf_f32");
unary_kernel!(SILU_F32, "silu_f32");

unary_kernel!(EXP_F16, "exp_f16");
unary_kernel!(LOG_F16, "log_f16");
unary_kernel!(SIN_F16, "sin_f16");
unary_kernel!(COS_F16, "cos_f16");
unary_kernel!(TANH_F16, "tanh_f16");
unary_kernel!(NEG_F16, "neg_f16");
unary_kernel!(RECIP_F16, "recip_f16");
unary_kernel!(SQR_F16, "sqr_f16");
unary_kernel!(SQRT_F16, "sqrt_f16");
unary_kernel!(ABS_F16, "abs_f16");
unary_kernel!(CEIL_F16, "ceil_f16");
unary_kernel!(FLOOR_F16, "floor_f16");
unary_kernel!(ROUND_F16, "round_f16");
unary_kernel!(RELU_F16, "relu_f16");
unary_kernel!(SIGN_F16, "sign_f16");
unary_kernel!(ERF_F16, "erf_f16");
unary_kernel!(GELU_F16, "gelu_f16");
unary_kernel!(GELU_ERF_F16, "gelu_erf_f16");
unary_kernel!(SILU_F16, "silu_f16");

unary_kernel!(EXP_BF16, "exp_bf16");
unary_kernel!(LOG_BF16, "log_bf16");
unary_kernel!(SIN_BF16, "sin_bf16");
unary_kernel!(COS_BF16, "cos_bf16");
unary_kernel!(TANH_BF16, "tanh_bf16");
unary_kernel!(NEG_BF16, "neg_bf16");
unary_kernel!(RECIP_BF16, "recip_bf16");
unary_kernel!(SQR_BF16, "sqr_bf16");
unary_kernel!(SQRT_BF16, "sqrt_bf16");
unary_kernel!(ABS_BF16, "abs_bf16");
unary_kernel!(CEIL_BF16, "ceil_bf16");
unary_kernel!(FLOOR_BF16, "floor_bf16");
unary_kernel!(ROUND_BF16, "round_bf16");
unary_kernel!(RELU_BF16, "relu_bf16");
unary_kernel!(SIGN_BF16, "sign_bf16");
unary_kernel!(ERF_BF16, "erf_bf16");
unary_kernel!(GELU_BF16, "gelu_bf16");
unary_kernel!(GELU_ERF_BF16, "gelu_erf_bf16");
unary_kernel!(SILU_BF16, "silu_bf16");

pub const REDUCE_F32: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/reduce_f32.spv"));
pub const CMP_EQ_F32: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/cmp_eq_f32.spv"));
pub const CMP_NE_F32: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/cmp_ne_f32.spv"));
pub const CMP_LT_F32: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/cmp_lt_f32.spv"));
pub const CMP_LE_F32: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/cmp_le_f32.spv"));
pub const CMP_GT_F32: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/cmp_gt_f32.spv"));
pub const CMP_GE_F32: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/cmp_ge_f32.spv"));
pub const F32_TO_BOOL_F32: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/f32_to_bool_f32.spv"));
pub const WHERE_COND_F32: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/where_cond_f32.spv"));
pub const F16_TO_F32: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/f16_to_f32.spv"));
pub const F32_TO_F16: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/f32_to_f16.spv"));
pub const BF16_TO_F32: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/bf16_to_f32.spv"));
pub const F32_TO_BF16: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/f32_to_bf16.spv"));
pub const BOOL_TO_F32: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/bool_to_f32.spv"));
pub const COPY_F16: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/copy_f16.spv"));
pub const COPY_BF16: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/copy_bf16.spv"));
pub const COPY2D_F32: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/copy2d_f32.spv"));
pub const COPY2D_F16: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/copy2d_f16.spv"));
pub const COPY2D_BF16: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/copy2d_bf16.spv"));
pub const INDEX_SELECT_F32: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/index_select_f32.spv"));
pub const INDEX_SELECT_F16: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/index_select_f16.spv"));
pub const INDEX_SELECT_BF16: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/index_select_bf16.spv"));
pub const MATMUL_F32: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/matmul_f32.spv"));
pub const MATMUL_F16: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/matmul_f16.spv"));
pub const MATMUL_BF16: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/matmul_bf16.spv"));
pub const MATMUL_F32_SPLITK: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/matmul_f32_splitk.spv"));
pub const MATMUL_F32_REDUCE: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/matmul_f32_reduce.spv"));
pub const MATMUL_F16_SPLITK: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/matmul_f16_splitk.spv"));
pub const MATMUL_F16_REDUCE: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/matmul_f16_reduce.spv"));
pub const MATMUL_BF16_SPLITK: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/matmul_bf16_splitk.spv"));
pub const MATMUL_BF16_REDUCE: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/matmul_bf16_reduce.spv"));
pub const U32_TO_F32: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/u32_to_f32.spv"));
pub const F32_TO_U32: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/f32_to_u32.spv"));
pub const I64_TO_F32: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/i64_to_f32.spv"));
pub const SOFTMAX_F32: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/softmax_f32.spv"));
pub const FA_VARLEN_F32: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/fa_varlen_f32.spv"));
pub const FA_VARLEN_TILED_F32_D64: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/fa_varlen_tiled_f32_d64.spv"));
pub const FA_VARLEN_TILED_F32_D128: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/fa_varlen_tiled_f32_d128.spv"));
pub const FA_VARLEN_TILED_F16_D64: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/fa_varlen_tiled_f16_d64.spv"));
pub const FA_VARLEN_TILED_F16_D128: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/fa_varlen_tiled_f16_d128.spv"));
pub const FA_VARLEN_F16: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/fa_varlen_f16.spv"));
pub const FA_VARLEN_BF16: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/fa_varlen_bf16.spv"));
