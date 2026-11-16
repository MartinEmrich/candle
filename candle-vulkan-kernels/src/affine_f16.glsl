#version 450
// y = x * mul + add (f16 storage, f32 compute), strided source (rank <= 4),
// contiguous output. One thread per u32 word (two adjacent elements).

#include "f16_common.glslh"

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer Src {
    uint data[];
} b_src;

layout(set = 0, binding = 1) buffer Dst {
    uint data[];
} b_dst;

layout(push_constant) uniform PushConstants {
    uint src_offset;  // in elements
    uint dst_offset;  // in elements
    uint numel;
    float mul;
    float add;
    uint rank;
    uint dims[4];
    uint strides[4];
} pc;

uint global_index() {
    uint idx = gl_GlobalInvocationID.x;
    idx += gl_NumWorkGroups.x * gl_WorkGroupSize.x * gl_WorkGroupID.y;
    idx += gl_NumWorkGroups.x * gl_NumWorkGroups.y * gl_WorkGroupSize.x * gl_WorkGroupID.z;
    return idx;
}

uint decomp(uint idx) {
    uint off = pc.src_offset;
    uint rem = idx;
    for (int d = int(pc.rank) - 1; d >= 0; --d) {
        uint coord = rem % pc.dims[d];
        rem /= pc.dims[d];
        off += coord * pc.strides[d];
    }
    return off;
}

float af(float x) {
    return x * pc.mul + pc.add;
}

void main() {
    uint w = global_index();
    uint i0 = w * 2u;
    if (i0 >= pc.numel) {
        return;
    }
    uint lo = F16_LOW(af(F16_LOAD(b_src.data, decomp(i0))));
    if (i0 + 1u < pc.numel) {
        uint hi = F16_LOW(af(F16_LOAD(b_src.data, decomp(i0 + 1u))));
        b_dst.data[(pc.dst_offset + i0) >> 1u] = (hi << 16u) | lo;
    } else {
        uint wi = (pc.dst_offset + i0) >> 1u;
        uint prev = b_dst.data[wi];
        b_dst.data[wi] = (prev & 0xFFFF0000u) | lo;
    }
}
