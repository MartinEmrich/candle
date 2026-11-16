#version 450
// Split-K batched GEMM, f16 storage / f32 compute. Each workgroup computes a
// partial 64x64 output tile over a K-slice, writing f32 partials to partial[ks]
// (plain store, no atomics). A separate reduce kernel sums the partials and packs
// to f16. Same tiling as matmul_f16 (16x16 threads, 4x4 accum, K-tile 16).
//
// pc layout (appends to the matmul_f16 push block):
//   [0..15] identical to matmul_f16
//   [16] k_chunk   elements per K-slice (multiple of 16)
//   [17] k_splits  number of K-slices
//   [18] split_stride = b * m * n (elements between consecutive ks planes)

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

shared float As[64][16];
shared float Bs[16][64];

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

    for (uint kt = k_start; kt < k_end; kt += 16u) {
        for (int i = 0; i < 4; ++i) {
            uint r_ = m0 + ty * 4u + uint(i);
            uint c_ = kt + tx;
            As[ty * 4 + i][tx] = (r_ < pc.m && c_ < k_end)
                ? F16_LOAD(b_l.data, lbase + r_ * pc.lm + c_ * pc.lk) : 0.0;
        }
        for (int j = 0; j < 4; ++j) {
            uint r_ = kt + ty;
            uint c_ = n0 + tx * 4u + uint(j);
            Bs[ty][tx * 4 + j] = (r_ < k_end && c_ < pc.n)
                ? F16_LOAD(b_r.data, rbase + r_ * pc.rk + c_ * pc.rn) : 0.0;
        }
        barrier();
        uint kend = min(16u, k_end - kt);
        for (uint kk = 0u; kk < kend; ++kk) {
            float a[4];
            float bb[4];
            for (int i = 0; i < 4; ++i) {
                a[i] = As[ty * 4 + i][kk];
            }
            for (int j = 0; j < 4; ++j) {
                bb[j] = Bs[kk][tx * 4 + j];
            }
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    acc[i][j] += a[i] * bb[j];
                }
            }
        }
        barrier();
    }

    uint pbase = ks * pc.split_stride + b * pc.ob;
    for (int i = 0; i < 4; ++i) {
        uint r_ = m0 + ty * 4u + uint(i);
        if (r_ < pc.m) {
            for (int j = 0; j < 4; ++j) {
                uint c_ = n0 + tx * 4u + uint(j);
                if (c_ < pc.n) {
                    b_o.data[pbase + r_ * pc.n + c_] = acc[i][j];
                }
            }
        }
    }
}
