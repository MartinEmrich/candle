#version 450
// 2D strided copy (contiguous inner dim), f16 storage / f32 compute.
// One thread per u32 output word (two adjacent elements). Requires word-aligned
// rows (d2, dst_offset, dst_stride1 even) so each output word maps to a single
// source row pair; model dims always satisfy this.

#include "f16_common.glslh"

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer Src {
    uint data[];
} b_src;

layout(set = 0, binding = 1) buffer Dst {
    uint data[];
} b_dst;

layout(push_constant) uniform PushConstants {
    uint src_offset;
    uint dst_offset;
    uint d1;
    uint d2;
    uint src_stride1;
    uint dst_stride1;
} pc;

uint global_index() {
    uint idx = gl_GlobalInvocationID.x;
    idx += gl_NumWorkGroups.x * gl_WorkGroupSize.x * gl_WorkGroupID.y;
    idx += gl_NumWorkGroups.x * gl_NumWorkGroups.y * gl_WorkGroupSize.x * gl_WorkGroupID.z;
    return idx;
}

void main() {
    uint w = global_index();
    uint i0 = w * 2u;
    uint n = pc.d1 * pc.d2;
    if (i0 >= n) {
        return;
    }
    uint r0 = i0 / pc.d2;
    uint c0 = i0 % pc.d2;
    uint s0 = pc.src_offset + r0 * pc.src_stride1 + c0;
    uint lo = F16_LOW(F16_LOAD(b_src.data, s0));
    uint d0 = pc.dst_offset + r0 * pc.dst_stride1 + c0;
    if (i0 + 1u < n) {
        uint i1 = i0 + 1u;
        uint r1 = i1 / pc.d2;
        uint c1 = i1 % pc.d2;
        uint s1 = pc.src_offset + r1 * pc.src_stride1 + c1;
        uint hi = F16_LOW(F16_LOAD(b_src.data, s1));
        b_dst.data[d0 >> 1u] = (hi << 16u) | lo;
    } else {
        uint prev = b_dst.data[d0 >> 1u];
        b_dst.data[d0 >> 1u] = (prev & 0xFFFF0000u) | lo;
    }
}
