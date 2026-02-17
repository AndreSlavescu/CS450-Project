#pragma once

// Qwen3-1.7B model dimensions
static constexpr int HIDDEN_DIM       = 2048;
static constexpr int NUM_Q_HEADS      = 16;
static constexpr int NUM_KV_HEADS     = 8;
static constexpr int HEAD_DIM         = 128;
static constexpr int HALF_HEAD_DIM    = HEAD_DIM / 2;
static constexpr int GQA_RATIO        = NUM_Q_HEADS / NUM_KV_HEADS;  // 2
static constexpr int Q_DIM            = NUM_Q_HEADS * HEAD_DIM;      // 2048
static constexpr int K_DIM            = NUM_KV_HEADS * HEAD_DIM;     // 1024
static constexpr int V_DIM            = NUM_KV_HEADS * HEAD_DIM;     // 1024
static constexpr int QKV_DIM          = Q_DIM + K_DIM + V_DIM;       // 4096
static constexpr int INTERMEDIATE_DIM = 6144;
static constexpr int VOCAB_SIZE       = 151936;
static constexpr int NUM_LAYERS       = 28;
static constexpr float EPS            = 1e-6f;
