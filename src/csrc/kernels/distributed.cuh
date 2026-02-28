#pragma once

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "multimem.cuh"

#define CU_CHECK(cmd)                                                                                                  \
    do {                                                                                                               \
        CUresult e = cmd;                                                                                              \
        if (e != CUDA_SUCCESS) {                                                                                       \
            const char* err_str;                                                                                       \
            cuGetErrorString(e, &err_str);                                                                             \
            printf("CUDA driver error %s:%d '%s'\n", __FILE__, __LINE__, err_str);                                     \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)

struct MulticastConfig {
    CUmemGenericAllocationHandle mc_handle;
    CUdeviceptr mc_addr;
    CUdeviceptr local_addr;
    CUmemGenericAllocationHandle* phys_handles;
    size_t buf_size;
    size_t granularity;
    int num_devices;
    bool initialized;
};

struct DistributedState {
    MulticastConfig data_mc;
    MulticastConfig flag_mc;
    int world_size;
    int local_rank;
    CUdevice* devices;
};

inline size_t align_up(size_t val, size_t alignment) {
    return (val + alignment - 1) & ~(alignment - 1);
}

inline MulticastConfig create_multicast_buffer(int num_devices, const CUdevice* devices, size_t requested_size) {
    MulticastConfig cfg = {};
    cfg.num_devices = num_devices;

    CUmulticastObjectProp mc_prop = {};
    mc_prop.numDevices = static_cast<unsigned int>(num_devices);
    mc_prop.size = requested_size;
    mc_prop.handleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

    CU_CHECK(cuMulticastGetGranularity(&cfg.granularity, &mc_prop, CU_MULTICAST_GRANULARITY_RECOMMENDED));
    cfg.buf_size = align_up(requested_size, cfg.granularity);
    mc_prop.size = cfg.buf_size;

    CU_CHECK(cuMulticastCreate(&cfg.mc_handle, &mc_prop));

    for (int i = 0; i < num_devices; i++) {
        CU_CHECK(cuMulticastAddDevice(cfg.mc_handle, devices[i]));
    }

    cfg.phys_handles = new CUmemGenericAllocationHandle[num_devices];
    for (int i = 0; i < num_devices; i++) {
        CUmemAllocationProp alloc_prop = {};
        alloc_prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        alloc_prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        alloc_prop.location.id = static_cast<int>(devices[i]);
        alloc_prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
        CU_CHECK(cuMemCreate(&cfg.phys_handles[i], cfg.buf_size, &alloc_prop, 0));
        CU_CHECK(cuMulticastBindMem(cfg.mc_handle, 0, cfg.phys_handles[i], 0, cfg.buf_size, 0));
    }

    CU_CHECK(cuMemAddressReserve(&cfg.mc_addr, cfg.buf_size, cfg.granularity, 0, 0));
    CU_CHECK(cuMemMap(cfg.mc_addr, cfg.buf_size, 0, cfg.mc_handle, 0));

    CU_CHECK(cuMemAddressReserve(&cfg.local_addr, cfg.buf_size, cfg.granularity, 0, 0));

    CUmemAccessDesc access_desc = {};
    access_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    for (int i = 0; i < num_devices; i++) {
        access_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        access_desc.location.id = static_cast<int>(devices[i]);
        CU_CHECK(cuMemSetAccess(cfg.mc_addr, cfg.buf_size, &access_desc, 1));
    }

    cfg.initialized = true;
    return cfg;
}

inline void bind_local_view(MulticastConfig& cfg, int rank, CUdevice device) {
    CU_CHECK(cuMemMap(cfg.local_addr, cfg.buf_size, 0, cfg.phys_handles[rank], 0));
    CUmemAccessDesc access_desc = {};
    access_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    access_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access_desc.location.id = static_cast<int>(device);
    CU_CHECK(cuMemSetAccess(cfg.local_addr, cfg.buf_size, &access_desc, 1));
}

inline void destroy_multicast_buffer(MulticastConfig& cfg) {
    if (!cfg.initialized)
        return;
    cuMemUnmap(cfg.local_addr, cfg.buf_size);
    cuMemAddressFree(cfg.local_addr, cfg.buf_size);
    cuMemUnmap(cfg.mc_addr, cfg.buf_size);
    cuMemAddressFree(cfg.mc_addr, cfg.buf_size);
    for (int i = 0; i < cfg.num_devices; i++) {
        cuMemRelease(cfg.phys_handles[i]);
    }
    cuMulticastDestroy(cfg.mc_handle);
    delete[] cfg.phys_handles;
    cfg.initialized = false;
}

inline DistributedState create_distributed_state(int world_size, int local_rank, size_t data_buf_size,
                                                 int num_sms_for_flags) {
    DistributedState state = {};
    state.world_size = world_size;
    state.local_rank = local_rank;

    state.devices = new CUdevice[world_size];
    for (int i = 0; i < world_size; i++) {
        CU_CHECK(cuDeviceGet(&state.devices[i], i));
    }

    state.data_mc = create_multicast_buffer(world_size, state.devices, data_buf_size);
    size_t flag_size = static_cast<size_t>(num_sms_for_flags) * sizeof(uint32_t);
    state.flag_mc = create_multicast_buffer(world_size, state.devices, flag_size);

    bind_local_view(state.data_mc, local_rank, state.devices[local_rank]);
    bind_local_view(state.flag_mc, local_rank, state.devices[local_rank]);

    cudaMemset(reinterpret_cast<void*>(state.flag_mc.local_addr), 0, state.flag_mc.buf_size);
    cudaDeviceSynchronize();

    return state;
}

inline void destroy_distributed_state(DistributedState& state) {
    destroy_multicast_buffer(state.data_mc);
    destroy_multicast_buffer(state.flag_mc);
    delete[] state.devices;
}
