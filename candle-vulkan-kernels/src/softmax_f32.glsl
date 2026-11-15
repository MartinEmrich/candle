#version 450
// Row softmax over the last dim (contiguous input). One workgroup per row.

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer Src {
    float data[];
} b_src;

layout(set = 0, binding = 1) buffer Dst {
    float data[];
} b_dst;

layout(push_constant) uniform PushConstants {
    uint src_offset;
    uint dst_offset;
    uint n_rows;
    uint n_cols;
} pc;

shared float sh[256];

uint workgroup_id() {
    return gl_WorkGroupID.x + gl_NumWorkGroups.x *
        (gl_WorkGroupID.y + gl_NumWorkGroups.y * gl_WorkGroupID.z);
}

void main() {
    uint row = workgroup_id();
    uint tid = gl_LocalInvocationIndex.x;
    if (row >= pc.n_rows) {
        return;
    }
    uint base = pc.src_offset + row * pc.n_cols;

    // pass 1: max
    float acc = -(1.0 / 0.0);
    for (uint i = tid; i < pc.n_cols; i += 256u) {
        acc = max(acc, b_src.data[base + i]);
    }
    sh[tid] = acc;
    barrier();
    for (uint s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) {
            sh[tid] = max(sh[tid], sh[tid + s]);
        }
        barrier();
    }
    float mx = sh[0];
    barrier();

    // pass 2: sum of exp
    acc = 0.0;
    for (uint i = tid; i < pc.n_cols; i += 256u) {
        acc += exp(b_src.data[base + i] - mx);
    }
    sh[tid] = acc;
    barrier();
    for (uint s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) {
            sh[tid] = sh[tid] + sh[tid + s];
        }
        barrier();
    }
    float ssum = sh[0];
    barrier();

    // pass 3: normalize
    for (uint i = tid; i < pc.n_cols; i += 256u) {
        b_dst.data[pc.dst_offset + row * pc.n_cols + i] =
            exp(b_src.data[base + i] - mx) / ssum;
    }
}
