#version 450

// Flash attention, variable-length, f32 compute (FA_SCALAR style).
//
// One thread computes one (query token, head) output row via online softmax,
// looping over the key/value tokens of that token's sequence. Supports
// variable-length batching (cu_seqlens), causal / sliding-window masking,
// ALiBi slopes and logit soft-capping. Grouped-query attention is handled by
// mapping each query head onto a (shared) key/value head.
//
// q,k,v,o are contiguous [total, num_heads, head_size] (varlen packing).

layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer Q { float data[]; } b_q;
layout(set = 0, binding = 1) readonly buffer K { float data[]; } b_k;
layout(set = 0, binding = 2) readonly buffer V { float data[]; } b_v;
layout(set = 0, binding = 3) buffer O { float data[]; } b_o;
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
    uint has_alibi;    // 0/1
    float scale;
    float softcap;     // 0 = disabled
} pc;

void main() {
    uint q_token = gl_WorkGroupID.x * 32u + gl_LocalInvocationID.x;
    uint head = gl_WorkGroupID.y;
    if (q_token >= pc.total_q) {
        return;
    }

    // Locate the sequence this query token belongs to via cu_seqlens_q.
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
        return; // padding guard
    }
    uint q_pos = q_token - q_start;

    uint k_start = b_ck.data[pc.ck_off + seq];
    uint k_end = b_ck.data[pc.ck_off + seq + 1u];
    uint k_len = k_end - k_start;

    // Restrict the key range to the causal / sliding window.
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

    float alibi_slope = pc.has_alibi != 0u ? b_alibi.data[head] : 0.0;

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
            s += b_q.data[q_base + d] * b_k.data[kb + d];
        }
        s *= pc.scale;
        if (pc.softcap > 0.0) {
            s = pc.softcap * tanh(s / pc.softcap);
        }
        if (pc.has_alibi != 0u) {
            s += alibi_slope * float(int(q_pos) - int(kt));
        }

        float m_new = max(m, s);
        float corr = exp(m - m_new);
        float p = exp(s - m_new);
        l = l * corr + p;
        for (uint d = 0u; d < pc.head_size; ++d) {
            acc[d] = acc[d] * corr + p * b_v.data[vb + d];
        }
        m = m_new;
    }

    float inv_l = (l > 0.0) ? (1.0 / l) : 0.0;
    for (uint d = 0u; d < pc.head_size; ++d) {
        b_o.data[o_base + d] = acc[d] * inv_l;
    }
}
