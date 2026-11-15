#version 450

// y = x * mul + add, with strided source (rank <= 4) and contiguous output.

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer Src {
    float data[];
} b_src;

layout(set = 0, binding = 1) buffer Dst {
    float data[];
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

void main() {
    uint idx = global_index();
    if (idx >= pc.numel) {
        return;
    }
    uint src_idx = pc.src_offset;
    uint rem = idx;
    for (int d = int(pc.rank) - 1; d >= 0; --d) {
        uint coord = rem % pc.dims[d];
        rem /= pc.dims[d];
        src_idx += coord * pc.strides[d];
    }
    b_dst.data[pc.dst_offset + idx] = b_src.data[src_idx] * pc.mul + pc.add;
}
