#version 450
// Batched tiled SGEMM: out[b] = l[b] @ r[b], 64x64 tile per workgroup,
// 16x16 threads, 4x4 accumulator per thread, K-tiled by 16.
// Inputs accessed through element strides so transposed views work.

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer Lhs {
    float data[];
} b_l;

layout(set = 0, binding = 1) readonly buffer Rhs {
    float data[];
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
    uint lb0; // l batch stride, outer batch dim
    uint lb1; // l batch stride, inner batch dim
    uint lm; // l row stride
    uint lk; // l col stride
    uint rb0; // r batch stride, outer batch dim
    uint rb1; // r batch stride, inner batch dim
    uint rk; // r row stride
    uint rn; // r col stride
    uint ob; // out batch stride
    uint b0_count; // size of inner batch dim (1 unless rank 4)
} pc;

shared float As[64][16];
shared float Bs[16][64];

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

    for (uint kt = 0u; kt < pc.k; kt += 16u) {
        for (int i = 0; i < 4; ++i) {
            uint r_ = m0 + ty * 4u + uint(i);
            uint c_ = kt + tx;
            As[ty * 4 + i][tx] = (r_ < pc.m && c_ < pc.k)
                ? b_l.data[lbase + r_ * pc.lm + c_ * pc.lk] : 0.0;
        }
        for (int j = 0; j < 4; ++j) {
            uint r_ = kt + ty;
            uint c_ = n0 + tx * 4u + uint(j);
            Bs[ty][tx * 4 + j] = (r_ < pc.k && c_ < pc.n)
                ? b_r.data[rbase + r_ * pc.rk + c_ * pc.rn] : 0.0;
        }
        barrier();
        uint kend = min(16u, pc.k - kt);
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

    for (int i = 0; i < 4; ++i) {
        uint r_ = m0 + ty * 4u + uint(i);
        if (r_ < pc.m) {
            for (int j = 0; j < 4; ++j) {
                uint c_ = n0 + tx * 4u + uint(j);
                if (c_ < pc.n) {
                    b_o.data[pc.o_offset + b * pc.ob + r_ * pc.n + c_] = acc[i][j];
                }
            }
        }
    }
}
