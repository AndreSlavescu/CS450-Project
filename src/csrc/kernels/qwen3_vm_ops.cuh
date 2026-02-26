#pragma once

#include "qwen3.cuh"
#include "qwen3_types.cuh"
#include "utils.cuh"
#include "rmsnorm.cuh"
#include "silu.cuh"

__device__ __forceinline__ void vm_barrier_wait(unsigned int* __restrict__ barriers, int barrier_id,
                                                unsigned int expected_count) {
    if (barrier_id < 0)
        return;
    __syncthreads();
    if (threadIdx.x == 0) {
        volatile unsigned int* vc = &barriers[barrier_id];
        while (*vc < expected_count) {
        }
        asm volatile("fence.acq_rel.gpu;" ::: "memory");
    }
    __syncthreads();
}

__device__ __forceinline__ void vm_barrier_signal(unsigned int* __restrict__ barriers, int barrier_id) {
    if (barrier_id < 0)
        return;
    __syncthreads();
    if (threadIdx.x == 0) {
        asm volatile("fence.acq_rel.gpu;" ::: "memory");
        atomicAdd(&barriers[barrier_id], 1u);
    }
}

#define VM_WAIT_BAR(instr) ((instr)[29])
#define VM_WAIT_COUNT(instr) ((unsigned int)(instr)[30])
#define VM_SIGNAL_BAR(instr) ((instr)[31])

struct VMState {
    float* hidden;
    float* g_q;
    float* g_k;
    float* g_v;
    float* g_attn_out;
    float* g_attn_partial;
    float* g_attn_lse;
    float* g_silu;
    float* logits;
    const Qwen3LayerWeights* __restrict__ layers;
    const float* __restrict__ cos_cache;
    const float* __restrict__ sin_cache;
    unsigned int* barriers;
    int pos_id;
    int max_seq_len;
    float attn_scale;
};

__device__ __forceinline__ void vm_gemv_rows(float* __restrict__ output, const float4* __restrict__ s_input,
                                             const __nv_bfloat16* __restrict__ weight, int start_row, int end_row,
                                             int input_dim, int warp_id, int num_warps, int lane) {
    for (int row = start_row + warp_id; row < end_row; row += num_warps) {
        const uint4* row8 = reinterpret_cast<const uint4*>(weight + (long long)row * input_dim);
        float dot = 0.f;
        for (int k = lane; k < input_dim / 8; k += 32) {
            uint4 raw = __ldcg(row8 + k);
            float2 f01 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw.x));
            float2 f23 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw.y));
            float2 f45 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw.z));
            float2 f67 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw.w));
            float4 x0 = s_input[k * 2], x1 = s_input[k * 2 + 1];
            dot += f01.x * x0.x + f01.y * x0.y + f23.x * x0.z + f23.y * x0.w + f45.x * x1.x + f45.y * x1.y +
                   f67.x * x1.z + f67.y * x1.w;
        }
        for (int s = 16; s > 0; s >>= 1)
            dot += __shfl_down_sync(0xffffffff, dot, s);
        if (lane == 0)
            output[row] = dot;
    }
}

__device__ __forceinline__ void vm_gemv_rows_residual(float* __restrict__ output, const float4* __restrict__ s_input,
                                                      const __nv_bfloat16* __restrict__ weight, int start_row,
                                                      int end_row, int input_dim, int warp_id, int num_warps,
                                                      int lane) {
    for (int row = start_row + warp_id; row < end_row; row += num_warps) {
        const uint4* row8 = reinterpret_cast<const uint4*>(weight + (long long)row * input_dim);
        float dot = 0.f;
        for (int k = lane; k < input_dim / 8; k += 32) {
            uint4 raw = __ldcg(row8 + k);
            float2 f01 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw.x));
            float2 f23 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw.y));
            float2 f45 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw.z));
            float2 f67 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw.w));
            float4 x0 = s_input[k * 2], x1 = s_input[k * 2 + 1];
            dot += f01.x * x0.x + f01.y * x0.y + f23.x * x0.z + f23.y * x0.w + f45.x * x1.x + f45.y * x1.y +
                   f67.x * x1.z + f67.y * x1.w;
        }
        for (int s = 16; s > 0; s >>= 1)
            dot += __shfl_down_sync(0xffffffff, dot, s);
        if (lane == 0)
            output[row] += dot;
    }
}

__device__ __forceinline__ void
vm_gemv_rows_upgate_silu(float* __restrict__ silu_out, const float4* __restrict__ s_input,
                         const __nv_bfloat16* __restrict__ gate_w, const __nv_bfloat16* __restrict__ up_w,
                         int start_row, int end_row, int input_dim, int warp_id, int num_warps, int lane) {
    for (int row = start_row + warp_id; row < end_row; row += num_warps) {
        const uint4* gr8 = reinterpret_cast<const uint4*>(gate_w + (long long)row * input_dim);
        const uint4* ur8 = reinterpret_cast<const uint4*>(up_w + (long long)row * input_dim);
        float gate_dot = 0.f, up_dot = 0.f;
        for (int k = lane; k < input_dim / 8; k += 32) {
            float4 x0 = s_input[k * 2], x1 = s_input[k * 2 + 1];
            uint4 graw = __ldcg(gr8 + k);
            float2 gf01 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&graw.x));
            float2 gf23 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&graw.y));
            float2 gf45 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&graw.z));
            float2 gf67 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&graw.w));
            gate_dot += gf01.x * x0.x + gf01.y * x0.y + gf23.x * x0.z + gf23.y * x0.w + gf45.x * x1.x + gf45.y * x1.y +
                        gf67.x * x1.z + gf67.y * x1.w;
            uint4 uraw = __ldcg(ur8 + k);
            float2 uf01 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&uraw.x));
            float2 uf23 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&uraw.y));
            float2 uf45 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&uraw.z));
            float2 uf67 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&uraw.w));
            up_dot += uf01.x * x0.x + uf01.y * x0.y + uf23.x * x0.z + uf23.y * x0.w + uf45.x * x1.x + uf45.y * x1.y +
                      uf67.x * x1.z + uf67.y * x1.w;
        }
        for (int s = 16; s > 0; s >>= 1) {
            gate_dot += __shfl_down_sync(0xffffffff, gate_dot, s);
            up_dot += __shfl_down_sync(0xffffffff, up_dot, s);
        }
        if (lane == 0)
            silu_out[row] = kernels::silu_multiply(gate_dot, up_dot);
    }
}

__device__ __forceinline__ void vm_gemv_rows_residual_global(float* __restrict__ output,
                                                             const float* __restrict__ global_input,
                                                             const __nv_bfloat16* __restrict__ weight, int start_row,
                                                             int end_row, int input_dim, int warp_id, int num_warps,
                                                             int lane) {
    const float4* input4 = reinterpret_cast<const float4*>(global_input);
    for (int row = start_row + warp_id; row < end_row; row += num_warps) {
        const uint4* row8 = reinterpret_cast<const uint4*>(weight + (long long)row * input_dim);
        float dot = 0.f;
        for (int k = lane; k < input_dim / 8; k += 32) {
            uint4 raw = __ldcg(row8 + k);
            float2 f01 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw.x));
            float2 f23 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw.y));
            float2 f45 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw.z));
            float2 f67 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw.w));
            float4 x0 = __ldcg(input4 + k * 2), x1 = __ldcg(input4 + k * 2 + 1);
            dot += f01.x * x0.x + f01.y * x0.y + f23.x * x0.z + f23.y * x0.w + f45.x * x1.x + f45.y * x1.y +
                   f67.x * x1.z + f67.y * x1.w;
        }
        for (int s = 16; s > 0; s >>= 1)
            dot += __shfl_down_sync(0xffffffff, dot, s);
        if (lane == 0)
            output[row] += dot;
    }
}

__device__ void vm_exec_qkv(const int* __restrict__ instr, VMState& st) {
    constexpr int hidden_size = QWEN3_1_7B.hidden_size;
    constexpr int q_dim = QWEN3_1_7B.num_attention_heads * QWEN3_1_7B.head_dim;
    constexpr int kv_dim = QWEN3_1_7B.num_key_value_heads * QWEN3_1_7B.head_dim;
    constexpr int qkv_dim = q_dim + 2 * kv_dim;

    const int layer = instr[1];
    const int start_blk = instr[2];
    const int end_blk = instr[3];
    const int blk_size = 16;

    const Qwen3LayerWeights& w = st.layers[layer];

    vm_barrier_wait(st.barriers, VM_WAIT_BAR(instr), VM_WAIT_COUNT(instr));

    extern __shared__ char smem[];
    float* s_post_ln = reinterpret_cast<float*>(smem);
    float* s_reduce = s_post_ln + hidden_size;

    int tid = threadIdx.x, nt = blockDim.x;
    int lane_id = tid % kernels::WARP_SIZE;
    int warp_id = tid / kernels::WARP_SIZE;
    int num_warps = nt / kernels::WARP_SIZE;

    kernels::rmsnorm(s_post_ln, st.hidden, w.attn_ln_w, hidden_size, s_reduce, tid, nt, lane_id, warp_id, num_warps);
    __syncthreads();

    const int start_row = start_blk * blk_size;
    const int end_row = end_blk * blk_size;
    const float4* input4 = reinterpret_cast<const float4*>(s_post_ln);
    const int lane = tid & 31;

    for (int row = start_row + warp_id; row < end_row; row += num_warps) {
        const uint4* row8 = reinterpret_cast<const uint4*>(w.qkv_w + (long long)row * hidden_size);
        float dot = 0.f;
        for (int k = lane; k < hidden_size / 8; k += 32) {
            uint4 raw = __ldcg(row8 + k);
            float2 f01 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw.x));
            float2 f23 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw.y));
            float2 f45 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw.z));
            float2 f67 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw.w));
            float4 x0 = input4[k * 2], x1 = input4[k * 2 + 1];
            dot += f01.x * x0.x + f01.y * x0.y + f23.x * x0.z + f23.y * x0.w + f45.x * x1.x + f45.y * x1.y +
                   f67.x * x1.z + f67.y * x1.w;
        }
        for (int s = 16; s > 0; s >>= 1)
            dot += __shfl_down_sync(0xffffffff, dot, s);
        if (lane == 0) {
            if (row < q_dim)
                st.g_q[row] = dot;
            else if (row < q_dim + kv_dim)
                st.g_k[row - q_dim] = dot;
            else
                st.g_v[row - q_dim - kv_dim] = dot;
        }
    }

    vm_barrier_signal(st.barriers, VM_SIGNAL_BAR(instr));
}

__device__ void vm_exec_qknorm_rope_cache(const int* __restrict__ instr, VMState& st) {
    constexpr int q_dim = QWEN3_1_7B.num_attention_heads * QWEN3_1_7B.head_dim;
    constexpr int kv_dim = QWEN3_1_7B.num_key_value_heads * QWEN3_1_7B.head_dim;
    constexpr int num_q_heads = QWEN3_1_7B.num_attention_heads;
    constexpr int num_kv_heads = QWEN3_1_7B.num_key_value_heads;
    constexpr int head_dim = QWEN3_1_7B.head_dim;
    constexpr int half_head = head_dim / 2;

    const int layer = instr[1];
    const Qwen3LayerWeights& w = st.layers[layer];

    vm_barrier_wait(st.barriers, VM_WAIT_BAR(instr), VM_WAIT_COUNT(instr));

    extern __shared__ char smem[];
    float* s_reduce = reinterpret_cast<float*>(smem);

    int tid = threadIdx.x, nt = blockDim.x;
    int lane_id = tid % kernels::WARP_SIZE;
    int warp_id = tid / kernels::WARP_SIZE;
    int num_warps = nt / kernels::WARP_SIZE;

    kernels::rmsnorm_per_head(st.g_q, w.q_norm_w, num_q_heads, head_dim, s_reduce, tid, nt, lane_id, warp_id,
                              num_warps);

    kernels::rmsnorm_per_head(st.g_k, w.k_norm_w, num_kv_heads, head_dim, s_reduce, tid, nt, lane_id, warp_id,
                              num_warps);

    const float* cos_pos = st.cos_cache + (long long)st.pos_id * head_dim;
    const float* sin_pos = st.sin_cache + (long long)st.pos_id * head_dim;

    for (int h = 0; h < num_q_heads; h++) {
        int off = h * head_dim;
        for (int i = tid; i < half_head; i += nt) {
            float x0 = st.g_q[off + i], x1 = st.g_q[off + i + half_head];
            float c = cos_pos[i], s = sin_pos[i];
            st.g_q[off + i] = x0 * c - x1 * s;
            st.g_q[off + i + half_head] = x1 * c + x0 * s;
        }
    }

    for (int h = 0; h < num_kv_heads; h++) {
        int off = h * head_dim;
        for (int i = tid; i < half_head; i += nt) {
            float x0 = st.g_k[off + i], x1 = st.g_k[off + i + half_head];
            float c = cos_pos[i], s = sin_pos[i];
            st.g_k[off + i] = x0 * c - x1 * s;
            st.g_k[off + i + half_head] = x1 * c + x0 * s;
        }
    }
    __syncthreads();

    for (int i = tid; i < kv_dim; i += nt) {
        w.k_cache[(long long)st.pos_id * kv_dim + i] = st.g_k[i];
        w.v_cache[(long long)st.pos_id * kv_dim + i] = st.g_v[i];
    }
    __syncthreads();

    vm_barrier_signal(st.barriers, VM_SIGNAL_BAR(instr));
}

__device__ void vm_exec_flash_decode(const int* __restrict__ instr, VMState& st) {
    constexpr int NUM_Q_HEADS = QWEN3_1_7B.num_attention_heads;
    constexpr int NUM_KV_HEADS = QWEN3_1_7B.num_key_value_heads;
    constexpr int HEAD_DIM = QWEN3_1_7B.head_dim;
    constexpr int KV_DIM = NUM_KV_HEADS * HEAD_DIM;
    constexpr int GQA_RATIO = QWEN3_1_7B.gqa_ratio();
    constexpr int ELEMS_PER_LANE = HEAD_DIM / kernels::WARP_SIZE;

    const int layer = instr[1];
    const int kv_head = instr[2];
    const int num_partials = instr[3];
    const int partial_idx = instr[4];

    const Qwen3LayerWeights& w = st.layers[layer];

    vm_barrier_wait(st.barriers, VM_WAIT_BAR(instr), VM_WAIT_COUNT(instr));

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int num_warps = blockDim.x >> 5;

    extern __shared__ char smem_raw[];
    float* s_o = reinterpret_cast<float*>(smem_raw);
    float* s_m = s_o + num_warps * HEAD_DIM;
    float* s_d = s_m + num_warps;

    const int seq_len = st.pos_id + 1;
    const int chunk = (seq_len + num_partials - 1) / num_partials;
    const int kv_start = partial_idx * chunk;
    const int kv_end = min(kv_start + chunk, seq_len);

    for (int gqa_i = 0; gqa_i < GQA_RATIO; gqa_i++) {
        const int q_h = kv_head * GQA_RATIO + gqa_i;
        const float* q_head = st.g_q + q_h * HEAD_DIM;

        const int warp_chunk = (kv_end - kv_start + num_warps - 1) / num_warps;
        const int w_start = kv_start + warp * warp_chunk;
        const int w_end = min(w_start + warp_chunk, kv_end);

        float m_w = -INFINITY;
        float d_w = 0.0f;
        float o_w[ELEMS_PER_LANE] = {};

        for (int pos = w_start; pos < w_end; pos++) {
            const float* k = w.k_cache + (long long)pos * KV_DIM + kv_head * HEAD_DIM;
            const float* v = w.v_cache + (long long)pos * KV_DIM + kv_head * HEAD_DIM;

            float dot = 0.0f;
#pragma unroll
            for (int d = lane; d < HEAD_DIM; d += kernels::WARP_SIZE)
                dot += q_head[d] * k[d];
#pragma unroll
            for (int s = 16; s > 0; s >>= 1)
                dot += __shfl_down_sync(0xffffffff, dot, s);
            const float score = __shfl_sync(0xffffffff, dot, 0) * st.attn_scale;

            const float m_new = fmaxf(m_w, score);
            const float alpha = kernels::fast_exp_approx(m_w - m_new);
            const float beta = kernels::fast_exp_approx(score - m_new);
            d_w = d_w * alpha + beta;
            m_w = m_new;

            const float4 v4 = *reinterpret_cast<const float4*>(v + lane * ELEMS_PER_LANE);
            o_w[0] = o_w[0] * alpha + beta * v4.x;
            o_w[1] = o_w[1] * alpha + beta * v4.y;
            o_w[2] = o_w[2] * alpha + beta * v4.z;
            o_w[3] = o_w[3] * alpha + beta * v4.w;
        }

#pragma unroll
        for (int i = 0; i < ELEMS_PER_LANE; i++)
            s_o[warp * HEAD_DIM + lane * ELEMS_PER_LANE + i] = o_w[i];
        if (lane == 0) {
            s_m[warp] = m_w;
            s_d[warp] = d_w;
        }
        __syncthreads();

        if (warp == 0) {
            float m_acc = s_m[0];
            float d_acc = s_d[0];
            float o_acc[ELEMS_PER_LANE];
#pragma unroll
            for (int i = 0; i < ELEMS_PER_LANE; i++)
                o_acc[i] = s_o[lane * ELEMS_PER_LANE + i];

            for (int w2 = 1; w2 < num_warps; w2++) {
                const float m_w2 = s_m[w2];
                const float d_w2 = s_d[w2];
                const float m_new = fmaxf(m_acc, m_w2);
                const float a1 = kernels::fast_exp_approx(m_acc - m_new);
                const float a2 = kernels::fast_exp_approx(m_w2 - m_new);
                d_acc = d_acc * a1 + d_w2 * a2;
                m_acc = m_new;
#pragma unroll
                for (int i = 0; i < ELEMS_PER_LANE; i++)
                    o_acc[i] = o_acc[i] * a1 + s_o[w2 * HEAD_DIM + lane * ELEMS_PER_LANE + i] * a2;
            }

            if (num_partials == 1) {
                const float inv_d = kernels::approx_reciprocal(d_acc);
                float* out = st.g_attn_out + q_h * HEAD_DIM;
#pragma unroll
                for (int i = 0; i < ELEMS_PER_LANE; i++)
                    out[lane * ELEMS_PER_LANE + i] = o_acc[i] * inv_d;
            } else {
                float* p_out = st.g_attn_partial + (long long)q_h * 24 * HEAD_DIM + partial_idx * HEAD_DIM;
#pragma unroll
                for (int i = 0; i < ELEMS_PER_LANE; i++)
                    p_out[lane * ELEMS_PER_LANE + i] = o_acc[i];
                if (lane == 0)
                    st.g_attn_lse[q_h * 24 + partial_idx] = m_acc + logf(d_acc);
            }
        }
        __syncthreads();
    }

    vm_barrier_signal(st.barriers, VM_SIGNAL_BAR(instr));
}

__device__ void vm_exec_attn_reduction(const int* __restrict__ instr, VMState& st) {
    constexpr int HEAD_DIM = QWEN3_1_7B.head_dim;
    constexpr int GQA_RATIO = QWEN3_1_7B.gqa_ratio();
    constexpr int ELEMS_PER_LANE = HEAD_DIM / kernels::WARP_SIZE;

    const int layer = instr[1];
    const int head_start = instr[2];
    const int num_partials = instr[3];
    const int is_terminal = instr[4];
    const int list_len = instr[5];
    const int* red_list = &instr[6];

    vm_barrier_wait(st.barriers, VM_WAIT_BAR(instr), VM_WAIT_COUNT(instr));

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    for (int gqa_i = warp; gqa_i < GQA_RATIO; gqa_i += (blockDim.x >> 5)) {
        const int qh = head_start + gqa_i;
        if (qh >= QWEN3_1_7B.num_attention_heads)
            break;

        float m_acc = st.g_attn_lse[qh * 24 + red_list[0]];
        float o_acc[ELEMS_PER_LANE];
        const float* p0 = st.g_attn_partial + (long long)qh * 24 * HEAD_DIM + red_list[0] * HEAD_DIM;
#pragma unroll
        for (int i = 0; i < ELEMS_PER_LANE; i++)
            o_acc[i] = p0[lane * ELEMS_PER_LANE + i];
        float d_acc = 1.0f;

        for (int r = 1; r < list_len; r++) {
            const int pidx = red_list[r];
            float lse_r = st.g_attn_lse[qh * 24 + pidx];
            const float* pr = st.g_attn_partial + (long long)qh * 24 * HEAD_DIM + pidx * HEAD_DIM;
            float m_new = fmaxf(m_acc, lse_r);
            float a1 = kernels::fast_exp_approx(m_acc - m_new);
            float a2 = kernels::fast_exp_approx(lse_r - m_new);
            d_acc = d_acc * a1 + a2;
            m_acc = m_new;
#pragma unroll
            for (int i = 0; i < ELEMS_PER_LANE; i++)
                o_acc[i] = o_acc[i] * a1 + pr[lane * ELEMS_PER_LANE + i] * a2;
        }

        if (is_terminal) {
            float inv_d = kernels::approx_reciprocal(d_acc);
            float* out = st.g_attn_out + qh * HEAD_DIM;
#pragma unroll
            for (int i = 0; i < ELEMS_PER_LANE; i++)
                out[lane * ELEMS_PER_LANE + i] = o_acc[i] * inv_d;
        }
    }
    __syncthreads();

    vm_barrier_signal(st.barriers, VM_SIGNAL_BAR(instr));
}

__device__ void vm_exec_oproj_residual(const int* __restrict__ instr, VMState& st) {
    constexpr int hidden_size = QWEN3_1_7B.hidden_size;
    const int layer = instr[1];
    const int start_blk = instr[2];
    const int end_blk = instr[3];
    const int blk_size = 8;

    const Qwen3LayerWeights& w = st.layers[layer];

    vm_barrier_wait(st.barriers, VM_WAIT_BAR(instr), VM_WAIT_COUNT(instr));

    extern __shared__ char smem[];
    float* s_attn = reinterpret_cast<float*>(smem);

    int tid = threadIdx.x, nt = blockDim.x;
    int warp_id = tid / kernels::WARP_SIZE;
    int num_warps = nt / kernels::WARP_SIZE;
    int lane = tid & 31;

    for (int i = tid; i < hidden_size; i += nt)
        s_attn[i] = st.g_attn_out[i];
    __syncthreads();

    const int start_row = start_blk * blk_size;
    const int end_row = end_blk * blk_size;
    const float4* input4 = reinterpret_cast<const float4*>(s_attn);

    vm_gemv_rows_residual(st.hidden, input4, w.o_proj_w, start_row, end_row, hidden_size, warp_id, num_warps, lane);

    vm_barrier_signal(st.barriers, VM_SIGNAL_BAR(instr));
}

__device__ void vm_exec_upgate_silu(const int* __restrict__ instr, VMState& st) {
    constexpr int hidden_size = QWEN3_1_7B.hidden_size;
    constexpr int intermediate_size = QWEN3_1_7B.intermediate_size;
    const int layer = instr[1];
    const int num_blk_ids = instr[2];
    const int* blk_ids = &instr[3];
    const int blk_size = 16;

    const Qwen3LayerWeights& w = st.layers[layer];

    vm_barrier_wait(st.barriers, VM_WAIT_BAR(instr), VM_WAIT_COUNT(instr));

    extern __shared__ char smem[];
    float* s_post_ln = reinterpret_cast<float*>(smem);
    float* s_reduce = s_post_ln + hidden_size;

    int tid = threadIdx.x, nt = blockDim.x;
    int lane_id = tid % kernels::WARP_SIZE;
    int warp_id = tid / kernels::WARP_SIZE;
    int num_warps = nt / kernels::WARP_SIZE;
    int lane = tid & 31;

    kernels::rmsnorm(s_post_ln, st.hidden, w.mlp_ln_w, hidden_size, s_reduce, tid, nt, lane_id, warp_id, num_warps);
    __syncthreads();

    const float4* input4 = reinterpret_cast<const float4*>(s_post_ln);

    for (int b = 0; b < num_blk_ids; b++) {
        int blk = blk_ids[b];
        int start_row = blk * blk_size;
        int end_row = start_row + blk_size;
        if (end_row > intermediate_size)
            end_row = intermediate_size;
        vm_gemv_rows_upgate_silu(st.g_silu, input4, w.gate_w, w.up_w, start_row, end_row, hidden_size, warp_id,
                                 num_warps, lane);
    }

    vm_barrier_signal(st.barriers, VM_SIGNAL_BAR(instr));
}

__device__ void vm_exec_downproj_residual(const int* __restrict__ instr, VMState& st) {
    constexpr int hidden_size = QWEN3_1_7B.hidden_size;
    constexpr int intermediate_size = QWEN3_1_7B.intermediate_size;
    const int layer = instr[1];
    const int start_blk = instr[2];
    const int end_blk = instr[3];

    const int blk_size = 8;

    const Qwen3LayerWeights& w = st.layers[layer];

    vm_barrier_wait(st.barriers, VM_WAIT_BAR(instr), VM_WAIT_COUNT(instr));

    int tid = threadIdx.x;
    int warp_id = tid / kernels::WARP_SIZE;
    int num_warps = blockDim.x / kernels::WARP_SIZE;
    int lane = tid & 31;

    const int start_row = start_blk * blk_size;
    const int end_row = end_blk * blk_size;

    vm_gemv_rows_residual_global(st.hidden, st.g_silu, w.down_proj_w, start_row, end_row, intermediate_size, warp_id,
                                 num_warps, lane);

    vm_barrier_signal(st.barriers, VM_SIGNAL_BAR(instr));
}

__device__ void vm_exec_lm_head(const int* __restrict__ instr, VMState& st, const float* __restrict__ norm_w,
                                const __nv_bfloat16* __restrict__ lm_head_w, int vocab_size) {
    constexpr int hidden_size = QWEN3_1_7B.hidden_size;
    const int start_blk = instr[1];
    const int end_blk = instr[2];
    const int blk_size = 16;

    vm_barrier_wait(st.barriers, VM_WAIT_BAR(instr), VM_WAIT_COUNT(instr));

    extern __shared__ char smem[];
    float* s_post_ln = reinterpret_cast<float*>(smem);
    float* s_reduce = s_post_ln + hidden_size;

    int tid = threadIdx.x, nt = blockDim.x;
    int lane_id = tid % kernels::WARP_SIZE;
    int warp_id = tid / kernels::WARP_SIZE;
    int num_warps = nt / kernels::WARP_SIZE;
    int lane = tid & 31;

    kernels::rmsnorm(s_post_ln, st.hidden, norm_w, hidden_size, s_reduce, tid, nt, lane_id, warp_id, num_warps);
    __syncthreads();

    const int start_row = start_blk * blk_size;
    const int end_row = min(end_blk * blk_size, vocab_size);
    const float4* input4 = reinterpret_cast<const float4*>(s_post_ln);

    vm_gemv_rows(st.logits, input4, lm_head_w, start_row, end_row, hidden_size, warp_id, num_warps, lane);

    vm_barrier_signal(st.barriers, VM_SIGNAL_BAR(instr));
}
