#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include <string>
#include <algorithm>
#include <unordered_map>

/*
gpu_profiler.cuh — Dual-clock CUDA profiler with Perfetto export.

- Timestamps from %globaltimer (nanoseconds, global across all SMs)
  and clock64() (SM-local cycles).
- Perfetto (Chrome Trace Event Format) JSON export, ready for ui.perfetto.dev.
- Zero-overhead when disabled. Block-, warp-, and region-based APIs. Host-side helpers for export & reporting.
*/

namespace profiler {

struct config {
    static constexpr int MAX_EVENTS = 64;
    static constexpr bool ENABLED = true;
};

struct alignas(8) event_record {
    uint64_t global_ns;
    uint64_t sm_cycles;
    int      event_id;
    int      sm_id;
};

__device__ __forceinline__ uint64_t read_globaltimer() {
    uint64_t ns;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(ns));
    return ns;
}

__device__ __forceinline__ uint64_t read_clock64() {
    return clock64();
}

__device__ __forceinline__ uint32_t read_smid() {
    uint32_t id;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(id));
    return id;
}

struct block_state {
    event_record events[config::MAX_EVENTS];
    int count;

    __device__ __forceinline__ void init() {
        if constexpr (!config::ENABLED) return;
        count = 0;
    }

    __device__ __forceinline__ int record(int event_id) {
        if constexpr (!config::ENABLED) return -1;

        int slot = atomicAdd(&count, 1);
        if (slot >= config::MAX_EVENTS) return -1;

        uint64_t gns  = read_globaltimer();
        uint64_t cyc  = read_clock64();

        events[slot].global_ns  = gns;
        events[slot].sm_cycles  = cyc;
        events[slot].event_id   = event_id;
        events[slot].sm_id      = static_cast<int>(read_smid());

        return slot;
    }

    __device__ __forceinline__ void flush(event_record* dst, int* out_count) {
        if constexpr (!config::ENABLED) return;
        int n = min(count, config::MAX_EVENTS);
        *out_count = n;
        for (int i = 0; i < n; i++) {
            dst[i] = events[i];
        }
    }
};

struct warp_state {
    event_record events[config::MAX_EVENTS];
    int count;

    __device__ __forceinline__ void init() {
        if constexpr (!config::ENABLED) return;
        count = 0;
    }

    __device__ __forceinline__ int record(int event_id) {
        if constexpr (!config::ENABLED) return -1;
        int slot = count;
        if (slot >= config::MAX_EVENTS) return -1;
        uint64_t gns  = read_globaltimer();
        uint64_t cyc  = read_clock64();

        events[slot].global_ns  = gns;
        events[slot].sm_cycles  = cyc;
        events[slot].event_id   = event_id;
        events[slot].sm_id      = static_cast<int>(read_smid());

        count = slot + 1;
        return slot;
    }

    __device__ __forceinline__ void flush(event_record* dst, int* out_count) {
        if constexpr (!config::ENABLED) return;
        int n = min(count, config::MAX_EVENTS);
        *out_count = n;
        for (int i = 0; i < n; i++) {
            dst[i] = events[i];
        }
    }
};

template <int N>
struct region_timer {
    uint64_t begin_ns[N];
    uint64_t end_ns[N];
    uint64_t begin_cyc[N];
    uint64_t end_cyc[N];

    __device__ __forceinline__ void begin(int region) {
        if constexpr (!config::ENABLED) return;
        begin_ns[region]  = read_globaltimer();
        begin_cyc[region] = read_clock64();
    }

    __device__ __forceinline__ void end(int region) {
        if constexpr (!config::ENABLED) return;
        end_cyc[region] = read_clock64();
        end_ns[region]  = read_globaltimer();
    }

    __device__ __forceinline__ uint64_t global_ns_elapsed(int region) const {
        return end_ns[region] - begin_ns[region];
    }

    __device__ __forceinline__ uint64_t sm_cycles_elapsed(int region) const {
        return end_cyc[region] - begin_cyc[region];
    }
};

struct event_names {
    std::unordered_map<int, std::string> map;

    void set(int event_id, const char* name) { map[event_id] = name; }

    const char* get(int event_id) const {
        auto it = map.find(event_id);
        if (it != map.end()) return it->second.c_str();
        return nullptr;
    }

    std::string get_or_default(int event_id) const {
        auto it = map.find(event_id);
        if (it != map.end()) return it->second;
        return "event_" + std::to_string(event_id);
    }
};

struct host_buffer {
    event_record* d_events = nullptr;
    int*          d_counts = nullptr;
    int           num_blocks = 0;

    void allocate(int blocks) {
        num_blocks = blocks;
        cudaMalloc(&d_events, blocks * config::MAX_EVENTS * sizeof(event_record));
        cudaMalloc(&d_counts, blocks * sizeof(int));
        cudaMemset(d_events, 0, blocks * config::MAX_EVENTS * sizeof(event_record));
        cudaMemset(d_counts, 0, blocks * sizeof(int));
    }

    void free() {
        if (d_events) { cudaFree(d_events); d_events = nullptr; }
        if (d_counts) { cudaFree(d_counts); d_counts = nullptr; }
    }

    __host__ __device__ event_record* block_events(int block_idx) {
        return d_events + block_idx * config::MAX_EVENTS;
    }

    __host__ __device__ int* block_count(int block_idx) {
        return d_counts + block_idx;
    }

    std::pair<std::vector<int>, std::vector<event_record>> download() {
        std::vector<int> h_counts(num_blocks);
        std::vector<event_record> h_events(num_blocks * config::MAX_EVENTS);

        cudaMemcpy(h_counts.data(), d_counts,
                   num_blocks * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_events.data(), d_events,
                   num_blocks * config::MAX_EVENTS * sizeof(event_record),
                   cudaMemcpyDeviceToHost);

        return {std::move(h_counts), std::move(h_events)};
    }

    void print_report(const event_names* names = nullptr) {
        if (!d_events || !d_counts || num_blocks == 0) return;
        auto [h_counts, h_events] = download();

        int total = 0;
        for (int b = 0; b < num_blocks; b++) total += h_counts[b];

        printf("===== GPU Profiler Report (%d events across %d blocks) =====\n",
               total, num_blocks);
        printf("%-6s %-8s %-20s %-6s %-18s %-18s\n",
               "Block", "EvtID", "Name", "SM", "GlobalTimer(ns)", "clock64(cyc)");
        printf("----------------------------------------------------------------------\n");

        for (int b = 0; b < num_blocks; b++) {
            int n = h_counts[b];
            event_record* base = h_events.data() + b * config::MAX_EVENTS;
            for (int i = 0; i < n; i++) {
                const char* name = names ? names->get(base[i].event_id) : nullptr;
                char name_buf[24];
                if (!name) {
                    snprintf(name_buf, sizeof(name_buf), "event_%d", base[i].event_id);
                    name = name_buf;
                }
                printf("%-6d %-8d %-20s %-6d %-18lu %-18lu\n",
                       b, base[i].event_id, name, base[i].sm_id,
                       (unsigned long)base[i].global_ns,
                       (unsigned long)base[i].sm_cycles);
            }
        }
        printf("======================================================================\n");
    }

    // Perfetto export: emits Chrome Trace Event Format JSON file.
    void export_perfetto_json(const char* path,
                              const event_names* names = nullptr,
                              bool paired = true) {
        if (!d_events || !d_counts || num_blocks == 0) return;
        auto [h_counts, h_events] = download();

        struct flat_event { int block; event_record ev; };
        std::vector<flat_event> all;
        for (int b = 0; b < num_blocks; b++) {
            int n = std::min(h_counts[b], config::MAX_EVENTS);
            event_record* base = h_events.data() + b * config::MAX_EVENTS;
            for (int i = 0; i < n; i++) {
                all.push_back({b, base[i]});
            }
        }
        std::sort(all.begin(), all.end(), [](const flat_event& a, const flat_event& b) {
            return a.ev.global_ns < b.ev.global_ns;
        });

        if (all.empty()) return;

        uint64_t t0 = all[0].ev.global_ns;

        FILE* f = fopen(path, "w");
        if (!f) {
            fprintf(stderr, "profiler: failed to open %s for writing\n", path);
            return;
        }

        fprintf(f, "{\"traceEvents\":[\n");
        fprintf(f,
            "{\"name\":\"process_name\",\"ph\":\"M\",\"pid\":0,\"tid\":0,"
            "\"args\":{\"name\":\"GPU\"}},\n");

        std::vector<int> sm_ids;
        for (auto& fe : all) {
            if (std::find(sm_ids.begin(), sm_ids.end(), fe.ev.sm_id) == sm_ids.end())
                sm_ids.push_back(fe.ev.sm_id);
        }
        std::sort(sm_ids.begin(), sm_ids.end());
        for (int sm : sm_ids) {
            fprintf(f,
                "{\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":0,\"tid\":%d,"
                "\"args\":{\"name\":\"SM %d\"}},\n", sm, sm);
        }

        bool first = true;
        for (size_t i = 0; i < all.size(); i++) {
            auto& fe = all[i];
            uint64_t ts_us = (fe.ev.global_ns - t0) / 1000;
            int eid = fe.ev.event_id;

            std::string ename;
            if (names) {
                ename = names->get_or_default(paired ? (eid & ~1) : eid);
            } else {
                ename = "event_" + std::to_string(paired ? (eid & ~1) : eid);
            }

            const char* ph;
            if (!paired) ph = "i";
            else if (eid % 2 == 0) ph = "B";
            else ph = "E";

            if (!first) fprintf(f, ",\n");
            first = false;

            fprintf(f,
                "{\"name\":\"%s\",\"cat\":\"gpu\",\"ph\":\"%s\","
                "\"ts\":%lu,\"pid\":0,\"tid\":%d,"
                "\"args\":{\"event_id\":%d,\"block\":%d,"
                "\"sm_cycles\":%lu,\"global_ns\":%lu}}",
                ename.c_str(), ph,
                (unsigned long)ts_us, fe.ev.sm_id,
                eid, fe.block,
                (unsigned long)fe.ev.sm_cycles,
                (unsigned long)fe.ev.global_ns);
        }

        fprintf(f, "\n]}\n");
        fclose(f);

        printf("profiler: wrote Perfetto trace to %s (%zu events)\n",
               path, all.size());
        printf("profiler: open https://ui.perfetto.dev and load the file\n");
    }
};

} // namespace profiler
