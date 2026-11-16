#version 450
// index_select along one dim (f16 storage, f32 compute). ids read as u32 words
// (i64 ids use stride 2, low word holds the value for ids < 2^31). Each thread
// handles one u32 output word (two adjacent elements), gathering each element's
// source address and packing the two 16-bit values into one word.

#include "bf16_common.glslh"

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer Src {
    uint data[];
} b_src;

layout(set = 0, binding = 1) readonly buffer Ids {
    uint data[];
} b_ids;

layout(set = 0, binding = 2) buffer Dst {
    uint data[];
} b_dst;

layout(push_constant) uniform PushConstants {
    uint src_offset;
    uint dst_offset;
    uint numel;
    uint rank;
    uint dims[4];        // output dims (dims[dim] == ids_len)
    uint src_strides[4]; // source strides
    uint ids_offset;     // in u32 words
    uint ids_stride;     // in u32 words (1 for u32, 2 for i64)
    uint dim;
} pc;

uint global_index() {
    uint idx = gl_GlobalInvocationID.x;
    idx += gl_NumWorkGroups.x * gl_WorkGroupSize.x * gl_WorkGroupID.y;
    idx += gl_NumWorkGroups.x * gl_NumWorkGroups.y * gl_WorkGroupSize.x * gl_WorkGroupID.z;
    return idx;
}

// Source element address (in elements) for output element i.
uint gather_addr(uint i) {
    uint ostride[4];
    uint s = 1u;
    for (int d = int(pc.rank) - 1; d >= 0; --d) {
        ostride[d] = s;
        s *= pc.dims[d];
    }
    uint coord_dim = (i / ostride[pc.dim]) % pc.dims[pc.dim];
    uint id = b_ids.data[pc.ids_offset + coord_dim * pc.ids_stride];
    uint addr = pc.src_offset + id * pc.src_strides[pc.dim];
    uint rem = i;
    for (int d = int(pc.rank) - 1; d >= 0; --d) {
        uint coord = rem % pc.dims[d];
        rem /= pc.dims[d];
        if (d != int(pc.dim)) {
            addr += coord * pc.src_strides[d];
        }
    }
    return addr;
}

void main() {
    uint w = global_index();
    uint i0 = w * 2u;
    if (i0 >= pc.numel) {
        return;
    }
    uint lo = BF16_LOW(BF16_LOAD(b_src.data, gather_addr(i0)));
    if (i0 + 1u < pc.numel) {
        uint hi = BF16_LOW(BF16_LOAD(b_src.data, gather_addr(i0 + 1u)));
        b_dst.data[(pc.dst_offset + i0) >> 1u] = (hi << 16u) | lo;
    } else {
        uint wi = (pc.dst_offset + i0) >> 1u;
        uint prev = b_dst.data[wi];
        b_dst.data[wi] = (prev & 0xFFFF0000u) | lo;
    }
}
