#include "gpu_profiler.cuh"

enum : int {
    EV_INIT = 0,
    EV_INIT_END = 1,
    EV_COMPUTE = 2,
    EV_COMPUTE_END = 3,
    EV_REDUCE = 4,
    EV_REDUCE_END = 5,
    EV_DONE = 6,
};

__global__ void profiled_kernel(float* output, int n, profiler::event_record* g_events, int* g_counts) {
    __shared__ profiler::block_state prof;
    __shared__ float smem[256];

    if (threadIdx.x == 0)
        prof.init();
    __syncthreads();

    if (threadIdx.x == 0)
        prof.record(EV_INIT);

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float val = 0.0f;
    if (idx < n) {
        val = sinf((float)idx * 0.001f);
    }
    smem[threadIdx.x] = val;
    __syncthreads();

    if (threadIdx.x == 0)
        prof.record(EV_INIT_END);

    if (threadIdx.x == 0)
        prof.record(EV_COMPUTE);

    for (int iter = 0; iter < 100; iter++) {
        val = val * 0.999f + smem[(threadIdx.x + iter) % blockDim.x] * 0.001f;
    }
    smem[threadIdx.x] = val;
    __syncthreads();

    if (threadIdx.x == 0)
        prof.record(EV_COMPUTE_END);

    if (threadIdx.x == 0)
        prof.record(EV_REDUCE);

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            smem[threadIdx.x] += smem[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
        prof.record(EV_REDUCE_END);

    if (threadIdx.x == 0) {
        output[blockIdx.x] = smem[0];
        prof.record(EV_DONE);
    }

    __syncthreads();
    if (threadIdx.x == 0) {
        prof.flush(g_events + blockIdx.x * profiler::config::MAX_EVENTS, g_counts + blockIdx.x);
    }
}

int main() {
    constexpr int N = 256 * 32;
    constexpr int NUM_BLOCKS = 32;
    constexpr int BLOCK_SIZE = 256;

    float* d_output;
    cudaMalloc(&d_output, NUM_BLOCKS * sizeof(float));

    profiler::host_buffer prof_buf;
    prof_buf.allocate(NUM_BLOCKS);

    printf("Launching profiled_kernel: %d blocks x %d threads\n", NUM_BLOCKS, BLOCK_SIZE);
    profiled_kernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_output, N, prof_buf.d_events, prof_buf.d_counts);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    profiler::event_names names;
    names.set(EV_INIT, "init");
    names.set(EV_INIT_END, "init_end");
    names.set(EV_COMPUTE, "compute");
    names.set(EV_COMPUTE_END, "compute_end");
    names.set(EV_REDUCE, "reduce");
    names.set(EV_REDUCE_END, "reduce_end");
    names.set(EV_DONE, "done");

    prof_buf.print_report(&names);

    prof_buf.export_perfetto_json("trace.json", &names, /*paired=*/true);

    float h_output[NUM_BLOCKS];
    cudaMemcpy(h_output, d_output, NUM_BLOCKS * sizeof(float), cudaMemcpyDeviceToHost);
    printf("\nFirst 4 block results: %.6f, %.6f, %.6f, %.6f\n", h_output[0], h_output[1], h_output[2], h_output[3]);

    prof_buf.free();
    cudaFree(d_output);

    printf("\nDone. Load trace.json at https://ui.perfetto.dev\n");
    return 0;
}
