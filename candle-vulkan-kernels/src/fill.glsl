#version 450

// Fill a u32-element view of a buffer with a constant 32-bit pattern.
// Used as the first end-to-end shader for the Vulkan backend; pattern
// replication covers zero/one initialization for all candle dtypes whose
// element size divides 4.

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer Data {
    uint data[];
} b_data;

layout(push_constant) uniform PushConstants {
    uint offset; // in u32 elements
    uint numel;  // in u32 elements
    uint value;
} pc;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx < pc.numel) {
        b_data.data[pc.offset + idx] = pc.value;
    }
}
