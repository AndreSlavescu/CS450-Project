cmake_minimum_required(VERSION 3.18)
project(cs450_csrc LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD 20)

add_library(gpu_profiler INTERFACE)
target_include_directories(gpu_profiler INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/profiler)

# Kernel headers (config + TP/SP + ZigZag attention)
add_library(qwen3_kernels INTERFACE)
target_include_directories(qwen3_kernels INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/kernels)
