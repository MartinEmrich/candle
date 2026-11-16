#version 450
// Sum the split-K f32 partial planes and pack to f16:
//   out[i] = f16(sum_{ks} partial[ks*split_stride + i])
// One thread per u32 output word (two adjacent elements). numel must be even.

#include "bf16_common.glslh"

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer Partial {
    float data[];
} b_p;

layout(set = 0, binding = 1) buffer Dst {
    uint data[];
} b_o;

layout(push_constant) uniform PushConstants {
    uint numel;
    uint k_splits;
    uint split_stride;
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
    if (i0 >= pc.numel) {
        return;
    }
    float s0 = 0.0;
    for (uint ks = 0u; ks < pc.k_splits; ++ks) {
        s0 += b_p.data[ks * pc.split_stride + i0];
    }
    uint lo = BF16_LOW(s0);
    if (i0 + 1u < pc.numel) {
        float s1 = 0.0;
        for (uint ks = 0u; ks < pc.k_splits; ++ks) {
            s1 += b_p.data[ks * pc.split_stride + i0 + 1u];
        }
        uint hi = BF16_LOW(s1);
        b_o.data[i0 >> 1u] = (hi << 16u) | lo;
    } else {
        uint prev = b_o.data[i0 >> 1u];
        b_o.data[i0 >> 1u] = (prev & 0xFFFF0000u) | lo;
    }
}
