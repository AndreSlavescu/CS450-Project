#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

struct Qwen3Config {
    int hidden_size;
    int intermediate_size;
    int num_attention_heads;
    int num_key_value_heads;
    int head_dim;
    int num_hidden_layers;
    int vocab_size;
    int max_position_embeddings;
    float rope_theta;
    float rms_norm_eps;

    __host__ __device__ int gqa_ratio() const { return num_attention_heads / num_key_value_heads; }
    __host__ __device__ int qkv_output_dim() const {
        return (num_attention_heads + 2 * num_key_value_heads) * head_dim;
    }
    __host__ __device__ float attn_scale() const { return 1.0f / sqrtf(static_cast<float>(head_dim)); }
};

// Qwen3-1.7B
constexpr Qwen3Config QWEN3_1_7B = {
    .hidden_size = 2048,
    .intermediate_size = 6144,
    .num_attention_heads = 16,
    .num_key_value_heads = 8,
    .head_dim = 128,
    .num_hidden_layers = 28,
    .vocab_size = 151936,
    .max_position_embeddings = 40960,
    .rope_theta = 1000000.0f,
    .rms_norm_eps = 1e-6f,
};

// Qwen3-8B
constexpr Qwen3Config QWEN3_8B = {
    .hidden_size = 4096,
    .intermediate_size = 12288,
    .num_attention_heads = 32,
    .num_key_value_heads = 8,
    .head_dim = 128,
    .num_hidden_layers = 36,
    .vocab_size = 151936,
    .max_position_embeddings = 40960,
    .rope_theta = 1000000.0f,
    .rms_norm_eps = 1e-6f,
};
