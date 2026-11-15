#version 450
// 2D strided copy (contiguous inner dim), f32. Used for narrow/cat paths.

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
    uint i = global_index();
    uint n = pc.d1 * pc.d2;
    if (i >= n) {
        return;
    }
    uint r = i / pc.d2;
    uint c = i % pc.d2;
    b_dst.data[pc.dst_offset + r * pc.dst_stride1 + c] =
        b_src.data[pc.src_offset + r * pc.src_stride1 + c];
}
