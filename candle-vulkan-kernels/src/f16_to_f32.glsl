#version 450
// f16 -> f32 cast. f16 storage is read as packed u32 words.

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer Src {
    uint data[];
} b_src;

layout(set = 0, binding = 1) buffer Dst {
    float data[];
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
    uint i = global_index();
    if (i >= pc.numel) {
        return;
    }
    uint si = decomp(i, pc.dims, pc.strides, pc.rank, pc.src_offset);
    vec2 h = unpackHalf2x16(b_src.data[si / 2u]);
    b_dst.data[pc.dst_offset + i] = h[si % 2u];
}
