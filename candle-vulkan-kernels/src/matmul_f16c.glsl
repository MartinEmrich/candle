#version 450
// Batched tiled GEMM with f16 storage and NATIVE f16 compute: out[b] = l[b] @ r[b].
//
// Same tiling / interface as matmul_f16 (64x64 tile per workgroup, 16x16 threads,
// 4x4 output per thread, K-tiled by 16, identical push constants and buffers), but
// the hot inner product runs on the hardware fp16 ALUs: shared K/V are stored as
// float16_t (half the LDS of the f32 body) and each thread accumulates its 4x4
// block with f16vec4 componentwise FMAs vectorised along the 4 owned columns
// (Bs[kk][tx*4+0..3] is contiguous, so it loads as one f16vec4 and As is broadcast).
//
// Each f16vec4 lane sums only the 16 products of one K-tile in fp16 before being
// widened into the f32 master accumulator, so the fp16 accumulation error stays at
// the same order as the tiled FA f16 path (cosine ~1e-5 vs f32). The master and the
// cross-tile sum are f32. On parts whose fp16 rate exceeds fp32 (Xe-LPG / Arc) this
// is a direct multiple; on fp16-at-fp32-rate parts (Xe-LP) it still wins a bit from
// 4-wide vectorisation and halved shared-memory traffic. The host selects this
// kernel over the f32-compute one only when a runtime benchmark shows it is faster
// on the actual device, so it never regresses a device where f32 compute wins.
//
// Requires the `shaderFloat16` device feature (enabled host-side when present) and
// an even n (all model projection/ffn widths are even; odd n falls back to f32).
// Inputs accessed through element strides so transposed views work.

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
    uint data[];
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
} pc;

shared float16_t As[64][16];
shared float16_t Bs[16][64];

void main() {
    uint b = gl_WorkGroupID.z;
    uint b0 = b / pc.b0_count;
    uint b1 = b % pc.b0_count;
    uint m0 = gl_WorkGroupID.x * 64u;
    uint n0 = gl_WorkGroupID.y * 64u;
    uint tx = gl_LocalInvocationID.x;
    uint ty = gl_LocalInvocationID.y;
    uint lbase = pc.l_offset + b0 * pc.lb0 + b1 * pc.lb1;
    uint rbase = pc.r_offset + b0 * pc.rb0 + b1 * pc.rb1;

    float acc[4][4];
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            acc[i][j] = 0.0;
        }
    }

    uint ccol = tx * 4u;
    for (uint kt = 0u; kt < pc.k; kt += 16u) {
        for (int i = 0; i < 4; ++i) {
            uint r_ = m0 + ty * 4u + uint(i);
            uint c_ = kt + tx;
            As[ty * 4 + i][tx] = (r_ < pc.m && c_ < pc.k)
                ? float16_t(F16_LOAD(b_l.data, lbase + r_ * pc.lm + c_ * pc.lk))
                : float16_t(0.0);
        }
        for (int j = 0; j < 4; ++j) {
            uint r_ = kt + ty;
            uint c_ = n0 + ccol + uint(j);
            Bs[ty][ccol + j] = (r_ < pc.k && c_ < pc.n)
                ? float16_t(F16_LOAD(b_r.data, rbase + r_ * pc.rk + c_ * pc.rn))
                : float16_t(0.0);
        }
        barrier();

        // f16 sub-accumulators for the 4x4 block, one f16vec4 per output row
        // (lanes = the 4 owned columns). Rows beyond kend stay zero (guarded load).
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
        // Widen the 16-term fp16 tile sums into the f32 master accumulator.
        acc[0][0] += float(s0.x); acc[0][1] += float(s0.y); acc[0][2] += float(s0.z); acc[0][3] += float(s0.w);
        acc[1][0] += float(s1.x); acc[1][1] += float(s1.y); acc[1][2] += float(s1.z); acc[1][3] += float(s1.w);
        acc[2][0] += float(s2.x); acc[2][1] += float(s2.y); acc[2][2] += float(s2.z); acc[2][3] += float(s2.w);
        acc[3][0] += float(s3.x); acc[3][1] += float(s3.y); acc[3][2] += float(s3.z); acc[3][3] += float(s3.w);
        barrier();
    }

    for (int i = 0; i < 4; ++i) {
        uint r_ = m0 + ty * 4u + uint(i);
        if (r_ < pc.m) {
            uint rowbase = pc.o_offset + b * pc.ob + r_ * pc.n;
            for (int j = 0; j < 4; j += 2) {
                uint c0 = n0 + ccol + uint(j);
                if (c0 + 1u < pc.n) {
                    uint word = (rowbase + c0) >> 1u;
                    uint lo = F16_LOW(acc[i][j]);
                    uint hi = F16_LOW(acc[i][j + 1]);
                    b_o.data[word] = (hi << 16u) | lo;
                } else if (c0 < pc.n) {
                    uint word = (rowbase + c0) >> 1u;
                    uint prev = b_o.data[word];
                    b_o.data[word] = (prev & 0xFFFF0000u) | F16_LOW(acc[i][j]);
                }
            }
        }
    }
}
