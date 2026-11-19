#version 450
// Split-K batched GEMM, f16 storage / NATIVE f16 compute. Each workgroup computes a
// partial 64x64 output tile over a K-slice, writing f32 partials to partial[ks]
// (plain store, no atomics). A separate reduce kernel (matmul_f16_reduce) sums the
// partials in f32 and packs to f16. Same tiling as matmul_f16c (16x16 threads, 4x4
// output per thread, K-tile 16, f16vec4 compute vectorised along the 4 owned
// columns, f16 sub-accumulation widened to an f32 master each tile).
//
// The partials are f32 (not f16) so the cross-split sum stays accurate; only the
// within-tile dot uses the fp16 ALUs. Selected host-side only when a runtime
// benchmark shows f16 compute beats f32 on the device (see matmul dispatch).
//
// pc layout (appends to the matmul_f16 push block):
//   [0..15] identical to matmul_f16
//   [16] k_chunk   elements per K-slice (multiple of 16)
//   [17] k_splits  number of K-slices
//   [18] split_stride = b * m * n (elements between consecutive ks planes)

#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require

#include "f16_common.glslh"

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer Lhs {
    uint data[];
} b_l;

layout(set = 0, binding = 1) readonly buffer Rhs {
    uint data[];
} b_r;

layout(set = 0, binding = 2) buffer Dst {
    float data[];
} b_o;

layout(push_constant) uniform PushConstants {
    uint l_offset;
    uint r_offset;
    uint o_offset;
    uint m;
    uint n;
    uint k;
    uint lb0;
    uint lb1;
    uint lm;
    uint lk;
    uint rb0;
    uint rb1;
    uint rk;
    uint rn;
    uint ob;
    uint b0_count;
    uint k_chunk;
    uint k_splits;
    uint split_stride;
} pc;

shared float16_t As[64][16];
shared float16_t Bs[16][64];

void main() {
    uint bz = gl_WorkGroupID.z;
    uint b = bz / pc.k_splits;
    uint ks = bz % pc.k_splits;
    uint b0 = b / pc.b0_count;
    uint b1 = b % pc.b0_count;
    uint m0 = gl_WorkGroupID.x * 64u;
    uint n0 = gl_WorkGroupID.y * 64u;
    uint tx = gl_LocalInvocationID.x;
    uint ty = gl_LocalInvocationID.y;
    uint lbase = pc.l_offset + b0 * pc.lb0 + b1 * pc.lb1;
    uint rbase = pc.r_offset + b0 * pc.rb0 + b1 * pc.rb1;

    uint k_start = ks * pc.k_chunk;
    uint k_end = min(pc.k, k_start + pc.k_chunk);

    float acc[4][4];
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            acc[i][j] = 0.0;
        }
    }

    uint ccol = tx * 4u;
    for (uint kt = k_start; kt < k_end; kt += 16u) {
        for (int i = 0; i < 4; ++i) {
            uint r_ = m0 + ty * 4u + uint(i);
            uint c_ = kt + tx;
            As[ty * 4 + i][tx] = (r_ < pc.m && c_ < k_end)
                ? float16_t(F16_LOAD(b_l.data, lbase + r_ * pc.lm + c_ * pc.lk))
                : float16_t(0.0);
        }
        for (int j = 0; j < 4; ++j) {
            uint r_ = kt + ty;
            uint c_ = n0 + ccol + uint(j);
            Bs[ty][ccol + j] = (r_ < k_end && c_ < pc.n)
                ? float16_t(F16_LOAD(b_r.data, rbase + r_ * pc.rk + c_ * pc.rn))
                : float16_t(0.0);
        }
        barrier();

        f16vec4 s0 = f16vec4(0.0);
        f16vec4 s1 = f16vec4(0.0);
        f16vec4 s2 = f16vec4(0.0);
        f16vec4 s3 = f16vec4(0.0);
        for (uint kk = 0u; kk < 16u; ++kk) {
            f16vec4 bv = f16vec4(Bs[kk][ccol], Bs[kk][ccol + 1u], Bs[kk][ccol + 2u], Bs[kk][ccol + 3u]);
            s0 += f16vec4(As[ty * 4 + 0][kk]) * bv;
            s1 += f16vec4(As[ty * 4 + 1][kk]) * bv;
            s2 += f16vec4(As[ty * 4 + 2][kk]) * bv;
            s3 += f16vec4(As[ty * 4 + 3][kk]) * bv;
        }
        acc[0][0] += float(s0.x); acc[0][1] += float(s0.y); acc[0][2] += float(s0.z); acc[0][3] += float(s0.w);
        acc[1][0] += float(s1.x); acc[1][1] += float(s1.y); acc[1][2] += float(s1.z); acc[1][3] += float(s1.w);
        acc[2][0] += float(s2.x); acc[2][1] += float(s2.y); acc[2][2] += float(s2.z); acc[2][3] += float(s2.w);
        acc[3][0] += float(s3.x); acc[3][1] += float(s3.y); acc[3][2] += float(s3.z); acc[3][3] += float(s3.w);
        barrier();
    }

    uint pbase = ks * pc.split_stride + b * pc.ob;
    for (int i = 0; i < 4; ++i) {
        uint r_ = m0 + ty * 4u + uint(i);
        if (r_ < pc.m) {
            for (int j = 0; j < 4; ++j) {
                uint c_ = n0 + ccol + uint(j);
                if (c_ < pc.n) {
                    b_o.data[pbase + r_ * pc.n + c_] = acc[i][j];
                }
            }
        }
    }
}
