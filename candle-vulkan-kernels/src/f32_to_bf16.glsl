#version 450
// f32 -> bf16 cast. Each thread packs two bf16 outputs into one u32 word
// (little-endian: element 0 in the low half). Uses round-to-nearest-even.
// Requires pc.dst_offset even (true for freshly allocated bf16 storage).

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

uint f32_to_bf16_bits(float v) {
    uint x = floatBitsToUint(v);
    // NaN must stay NaN (avoid the rounding turning a quiet NaN into something else).
    if ((x & 0x7FFFFFFFu) > 0x7F800000u) {
        return (x >> 16u) | 0x0040u;
    }
    uint lsb = (x >> 16u) & 1u;
    uint rounded = x + 0x7FFFu + lsb;
    return rounded >> 16u;
}

void main() {
    uint w = global_index();
    uint i0 = w * 2u;
    if (i0 >= pc.numel) {
        return;
    }
    float v0 = b_src.data[decomp(i0, pc.dims, pc.strides, pc.rank, pc.src_offset)];
    uint lo = f32_to_bf16_bits(v0);
    if (i0 + 1u < pc.numel) {
        float v1 = b_src.data[decomp(i0 + 1u, pc.dims, pc.strides, pc.rank, pc.src_offset)];
        uint hi = f32_to_bf16_bits(v1);
        b_dst.data[(pc.dst_offset + i0) / 2u] = (hi << 16u) | lo;
    } else {
        // Odd tail: only the low half is meaningful; preserve the other half.
        uint word_idx = (pc.dst_offset + i0) / 2u;
        uint prev = b_dst.data[word_idx];
        b_dst.data[word_idx] = (prev & 0xFFFF0000u) | lo;
    }
}
