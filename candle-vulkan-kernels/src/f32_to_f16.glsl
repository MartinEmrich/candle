#version 450
// f32 -> f16 cast. Each thread packs two f16 outputs into one u32 word.
// Requires pc.dst_offset even (true for freshly allocated f16 storage).

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer Src {
    float data[];
} b_src;

layout(set = 0, binding = 1) buffer Dst {
    uint data[];
} b_dst;

layout(push_constant) uniform PushConstants {
    uint src_offset;
    uint dst_offset;
    uint numel;
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

uint decomp(uint idx, uint dims[4], uint strides[4], uint rank, uint offset) {
    uint off = offset;
    uint rem = idx;
    for (int d = int(rank) - 1; d >= 0; --d) {
        uint coord = rem % dims[d];
        rem /= dims[d];
        off += coord * strides[d];
    }
    return off;
}

void main() {
    uint w = global_index();
    uint i0 = w * 2u;
    if (i0 >= pc.numel) {
        return;
    }
    float v0 = b_src.data[decomp(i0, pc.dims, pc.strides, pc.rank, pc.src_offset)];
    float v1 = 0.0;
    if (i0 + 1u < pc.numel) {
        v1 = b_src.data[decomp(i0 + 1u, pc.dims, pc.strides, pc.rank, pc.src_offset)];
    }
    b_dst.data[(pc.dst_offset + i0) / 2u] = packHalf2x16(vec2(v0, v1));
}
