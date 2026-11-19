#version 450
// Batched tiled GEMM using VK_KHR_cooperative_matrix hardware tiles:
//   out[b] = l[b] @ r[b]   (f16 storage, f32 accumulate).
//
// Same push constants and (packed-f16) buffers as matmul_f16 / matmul_f16c, but
// each workgroup is ONE subgroup that computes a single 16x16 output tile with the
// cooperative-matrix unit: it loads 16x16 A and B tiles straight from the packed
// f16 storage, chains coopMatMulAdd over K in steps of 16 into an f32 accumulator,
// then stores the f32 tile to shared and converts it to packed f16. The f32
// accumulator gives the same accuracy class as the tiled f16c path (cosine ~1e-5 vs
// f32); the output is byte-identical in layout to the other f16 kernels.
//
// Tiling is 16x16 (the universally supported cooperative-matrix shape; the host
// only selects this kernel when a CooperativeMatrixPropertiesKHR entry matches
// m=n=k=16, a=b=f16, c=result=f32, scope=subgroup). Because the tiles are 4x
// finer than the 64x64 f16c tiling, a skinny GEMM (small M) already spawns enough
// workgroups to fill the GPU, so this path is used INSTEAD of split-K.
//
// Loads use the matrix layout (row/col-major) implied by the element strides, so
// both contiguous and transposed-of-contiguous operands work: A[m,k] is row-major
// when lk==1 (row pitch lm) else col-major when lm==1 (pitch lk); B[k,n] is
// row-major when rn==1 (pitch rk) else col-major when rk==1 (pitch rn). n and k
// are multiples of 16 (all model widths); the M remainder is handled by shifting
// the last M-tile back so its 16-row window stays in-bounds (recomputing a few
// rows identically). The host only dispatches this when m>=16, n%16==0, k%16==0
// and both operands are row- or col-major, and only after a runtime benchmark plus
// an output-correctness check vs f32 pass on the actual device.

#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require

layout(local_size_x = 32) in;  // one subgroup (cooperative-matrix scope = Subgroup)

layout(set = 0, binding = 0) readonly buffer Lhs {
    float16_t data[];
} b_l;

layout(set = 0, binding = 1) readonly buffer Rhs {
    float16_t data[];
} b_r;

layout(set = 0, binding = 2) buffer Dst {
    float16_t data[];
} b_o;

layout(set = 0, binding = 3) readonly buffer Zero {
    float data[];
} b_z;

layout(push_constant) uniform PushConstants {
    uint l_offset;
    uint r_offset;
    uint o_offset;
    uint m;
    uint n;
    uint k;
    uint lb0;
    uint lb1;
    uint lm;
    uint lk;
    uint rb0;
    uint rb1;
    uint rk;
    uint rn;
    uint ob;
    uint b0_count;
} pc;

// 16x16 f32 staging for the accumulator, converted to packed f16 on the way out.
shared float stage[256];

void main() {
    uint b = gl_WorkGroupID.z;
    uint b0 = b / pc.b0_count;
    uint b1 = b % pc.b0_count;
    uint m0 = gl_WorkGroupID.x * 16u;
    uint n0 = gl_WorkGroupID.y * 16u;
    // Shift the last M-tile back so its 16-row window is in-bounds (host guarantees
    // m >= 16). Rows in the overlap are recomputed identically (benign race).
    if (m0 + 16u > pc.m) {
        m0 = pc.m - 16u;
    }
    uint lbase = pc.l_offset + b0 * pc.lb0 + b1 * pc.lb1;
    uint rbase = pc.r_offset + b0 * pc.rb0 + b1 * pc.rb1;

    // A[m,k]: row-major if lk==1 (pitch lm), else col-major if lm==1 (pitch lk).
    // B[k,n]: row-major if rn==1 (pitch rk), else col-major if rk==1 (pitch rn).
    // The matrix-layout operand must be a compile-time constant, so branch on the
    // stride-implied layout and pass a literal 0 (row-major) / 1 (col-major).
    bool a_row = (pc.lk == 1u);
    bool b_row = (pc.rn == 1u);

    coopmat<float, gl_ScopeSubgroup, 16, 16, 2> c;
    coopMatLoad(c, b_z.data, 0u, 16u, 0);  // zero the accumulator

    for (uint kt = 0u; kt < pc.k; kt += 16u) {
        coopmat<float16_t, gl_ScopeSubgroup, 16, 16, 0> a;
        coopmat<float16_t, gl_ScopeSubgroup, 16, 16, 1> bb;
        uint a_elem = lbase + m0 * pc.lm + kt * pc.lk;
        uint b_elem = rbase + kt * pc.rk + n0 * pc.rn;
        if (a_row) { coopMatLoad(a, b_l.data, a_elem, pc.lm, 0); }
        else       { coopMatLoad(a, b_l.data, a_elem, pc.lk, 1); }
        if (b_row) { coopMatLoad(bb, b_r.data, b_elem, pc.rk, 0); }
        else       { coopMatLoad(bb, b_r.data, b_elem, pc.rn, 1); }
        c = coopMatMulAdd(a, bb, c);
    }

    // f32 accumulator -> shared -> packed f16 output.
    coopMatStore(c, stage, 0u, 16u, 0);
    barrier();
    uint tid = gl_LocalInvocationID.x;
    for (uint i = tid; i < 256u; i += 32u) {
        uint r = i / 16u;
        uint cc = i % 16u;
        uint gr = m0 + r;
        uint gc = n0 + cc;
        if (gr < pc.m && gc < pc.n) {
            uint idx = pc.o_offset + b * pc.ob + gr * pc.n + gc;
            b_o.data[idx] = float16_t(stage[r * 16u + cc]);
        }
    }
}
