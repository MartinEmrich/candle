#version 450
// Row reduction along one axis: one workgroup per output element.
// pc.dims = src dims with dims[axis] = 1; pc.strides = src strides with
// strides[axis] = 0, so decomp(row) yields the row base in the source.

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
    uint numel_out;
    uint dim_len;
    uint reduce_stride;
    uint rank;
    uint dims[4];
    uint strides[4];
    uint op; // 0 = sum, 1 = min, 2 = max
    uint mode; // 0 = strided row, 1 = full reduction (decomp each index)
} pc;

shared float sh[256];

uint workgroup_id() {
    return gl_WorkGroupID.x + gl_NumWorkGroups.x *
        (gl_WorkGroupID.y + gl_NumWorkGroups.y * gl_WorkGroupID.z);
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
    uint row = workgroup_id();
    uint tid = gl_LocalInvocationIndex.x;
    if (row >= pc.numel_out) {
        return;
    }
    uint base = decomp(row, pc.dims, pc.strides, pc.rank, pc.src_offset);
    float acc = pc.op == 1u ? 1.0 / 0.0 : (pc.op == 2u ? -(1.0 / 0.0) : 0.0);
    for (uint i = tid; i < pc.dim_len; i += 256u) {
        uint addr = pc.mode == 1u
            ? decomp(i, pc.dims, pc.strides, pc.rank, pc.src_offset)
            : base + i * pc.reduce_stride;
        float v = b_src.data[addr];
        if (pc.op == 0u) {
            acc += v;
        } else if (pc.op == 1u) {
            acc = min(acc, v);
        } else {
            acc = max(acc, v);
        }
    }
    sh[tid] = acc;
    barrier();
    for (uint s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) {
            float a = sh[tid];
            float b = sh[tid + s];
            sh[tid] = pc.op == 0u ? a + b : (pc.op == 1u ? min(a, b) : max(a, b));
        }
        barrier();
    }
    if (tid == 0u) {
        b_dst.data[pc.dst_offset + row] = sh[0];
    }
}
