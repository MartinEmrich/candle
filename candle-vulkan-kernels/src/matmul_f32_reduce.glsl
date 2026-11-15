#version 450
// Sum the split-K partial planes: out[i] = sum_{ks} partial[ks*split_stride + i].
// numel = b * m * n (one output element per thread).

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer Partial {
    float data[];
} b_p;

layout(set = 0, binding = 1) buffer Dst {
    float data[];
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
    uint i = global_index();
    if (i >= pc.numel) {
        return;
    }
    float s = 0.0;
    for (uint ks = 0u; ks < pc.k_splits; ++ks) {
        s += b_p.data[ks * pc.split_stride + i];
    }
    b_o.data[i] = s;
}
