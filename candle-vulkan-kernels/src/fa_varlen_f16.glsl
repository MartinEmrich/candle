#version 450
// Flash attention, variable-length, f16 storage + f32 compute (FA_SCALAR style).
// Same algorithm as fa_varlen_f32 but q/k/v/o are packed 2x f16 per u32; values
// are unpacked to f32 registers for the online softmax and packed back on store.
// Requires head_size even (models use 64/128/80/...). ALiBi is not supported in
// the f16 path (dispatch rejects it).
//
// q,k,v,o are contiguous [total, num_heads, head_size] (varlen packing).

#include "f16_common.glslh"

layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer Q { uint data[]; } b_q;
layout(set = 0, binding = 1) readonly buffer K { uint data[]; } b_k;
layout(set = 0, binding = 2) readonly buffer V { uint data[]; } b_v;
layout(set = 0, binding = 3) buffer O { uint data[]; } b_o;
layout(set = 0, binding = 4) readonly buffer CQ { uint data[]; } b_cq;
layout(set = 0, binding = 5) readonly buffer CK { uint data[]; } b_ck;
layout(set = 0, binding = 6) readonly buffer ALIBI { float data[]; } b_alibi;

layout(push_constant) uniform PushConstants {
    uint head_size;
    uint num_heads;
    uint num_heads_k;
    uint total_q;
    uint q_off;
    uint k_off;
    uint v_off;
    uint o_off;
    uint cq_off;
    uint ck_off;
    uint batch;
    int window_left;   // -1 = unbounded
    int window_right;  // -1 = unbounded (causal uses 0)
    uint has_alibi;    // 0/1 (must be 0 for f16)
    float scale;
    float softcap;     // 0 = disabled
} pc;

void main() {
    uint q_token = gl_WorkGroupID.x * 32u + gl_LocalInvocationID.x;
    uint head = gl_WorkGroupID.y;
    if (q_token >= pc.total_q) {
        return;
    }

    uint seq = pc.batch - 1u;
    for (uint i = 0u; i < pc.batch; ++i) {
        if (q_token < b_cq.data[pc.cq_off + i + 1u]) {
            seq = i;
            break;
        }
    }
    uint q_start = b_cq.data[pc.cq_off + seq];
    uint q_end = b_cq.data[pc.cq_off + seq + 1u];
    if (q_token >= q_end) {
        return;
    }
    uint q_pos = q_token - q_start;

    uint k_start = b_ck.data[pc.ck_off + seq];
    uint k_end = b_ck.data[pc.ck_off + seq + 1u];
    uint k_len = k_end - k_start;

    uint k_lo = 0u;
    uint k_hi = k_len;
    if (pc.window_left >= 0) {
        k_lo = (q_pos > uint(pc.window_left)) ? (q_pos - uint(pc.window_left)) : 0u;
    }
    if (pc.window_right >= 0) {
        uint hi = q_pos + uint(pc.window_right) + 1u;
        k_hi = min(k_len, hi);
    }

    uint kv_head = head / (pc.num_heads / pc.num_heads_k);
    uint kv_stride = pc.num_heads_k * pc.head_size;
    uint q_stride = pc.num_heads * pc.head_size;

    uint q_base = pc.q_off + q_token * q_stride + head * pc.head_size;
    uint k_base = pc.k_off + k_start * kv_stride + kv_head * pc.head_size;
    uint v_base = pc.v_off + k_start * kv_stride + kv_head * pc.head_size;
    uint o_base = pc.o_off + q_token * q_stride + head * pc.head_size;

    float m = -(1.0 / 0.0);
    float l = 0.0;
    float acc[256];
    for (int d = 0; d < 256; ++d) {
        acc[d] = 0.0;
    }

    for (uint kt = k_lo; kt < k_hi; ++kt) {
        uint kb = k_base + kt * kv_stride;
        uint vb = v_base + kt * kv_stride;

        float s = 0.0;
        for (uint d = 0u; d < pc.head_size; ++d) {
            s += F16_LOAD(b_q.data, q_base + d) * F16_LOAD(b_k.data, kb + d);
        }
        s *= pc.scale;
        if (pc.softcap > 0.0) {
            s = pc.softcap * tanh(s / pc.softcap);
        }

        float m_new = max(m, s);
        float corr = exp(m - m_new);
        float p = exp(s - m_new);
        l = l * corr + p;
        for (uint d = 0u; d < pc.head_size; ++d) {
            acc[d] = acc[d] * corr + p * F16_LOAD(b_v.data, vb + d);
        }
        m = m_new;
    }

    float inv_l = (l > 0.0) ? (1.0 / l) : 0.0;
    for (uint d = 0u; d < pc.head_size; d += 2u) {
        uint e0 = o_base + d;
        if (d + 1u < pc.head_size) {
            uint word = e0 >> 1u;
            uint lo = F16_LOW(acc[d] * inv_l);
            uint hi = F16_LOW(acc[d + 1u] * inv_l);
            b_o.data[word] = (hi << 16u) | lo;
        } else {
            uint word = e0 >> 1u;
            uint prev = b_o.data[word];
            b_o.data[word] = (prev & 0xFFFF0000u) | F16_LOW(acc[d] * inv_l);
        }
    }
}
