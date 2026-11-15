#version 450
// where_cond: out = cond ? on_true : on_false. Same shape for all inputs.
// cond is bool storage (u8), read as packed bytes.

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer Cond {
    uint data[];
} b_cond;

layout(set = 0, binding = 1) readonly buffer True {
    float data[];
} b_true;

layout(set = 0, binding = 2) readonly buffer False {
    float data[];
} b_false;

layout(set = 0, binding = 3) buffer Dst {
    float data[];
} b_dst;

layout(push_constant) uniform PushConstants {
    uint cond_offset;
    uint true_offset;
    uint false_offset;
    uint dst_offset;
    uint numel;
    uint rank;
    uint dims[4];
    uint cond_strides[4];
    uint true_strides[4];
    uint false_strides[4];
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

bool cond_at(uint i) {
    uint byte_i = pc.cond_offset + decomp(i, pc.dims, pc.cond_strides, pc.rank, 0u);
    uint word = b_cond.data[byte_i / 4u];
    return ((word >> (8u * (byte_i % 4u))) & 0xffu) != 0u;
}

void main() {
    uint i = global_index();
    if (i >= pc.numel) {
        return;
    }
    uint ti = decomp(i, pc.dims, pc.true_strides, pc.rank, pc.true_offset);
    uint fi = decomp(i, pc.dims, pc.false_strides, pc.rank, pc.false_offset);
    b_dst.data[pc.dst_offset + i] = cond_at(i) ? b_true.data[ti] : b_false.data[fi];
}
