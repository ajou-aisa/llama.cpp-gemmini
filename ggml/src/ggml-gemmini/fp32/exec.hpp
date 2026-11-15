#pragma once

#include "ggml.h"
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#if !defined(__riscv)
void gemmini_fp32_model_issue(uint64_t, uint64_t, uint32_t);
void gemmini_fp32_model_fence();
#endif

namespace ggml::gemmini::fp32 {
// The native model and independent golden select the same documented RTL rule.
enum class Order : uint32_t { chunk_fma = 0, serial_fma = 1, chunk_reverse_fma = 2 };
void set_order(Order order);
Order get_order();
const char * order_name();

struct Counters {
    uint64_t linear = 0, qk = 0, pv = 0, gemms = 0, flushes = 0;
    uint64_t conversions = 0, cached_bytes = 0;
};
Counters counters();
void reset_counts();
void reset_weights();

struct ThreadReport {
    int cpu = -1;
    uint64_t jobs = 0, elements = 0, mismatch = 0;
};
// Exactly one command issuer and three helper lanes; the caller is the fourth lane.
void start_runtime(int issuer_cpu, const std::vector<int> & lane_cpus);
void stop_runtime();
std::vector<ThreadReport> runtime_report();

// A[M,K], B[K,N], C[M,N], all contiguous F32. Always WS, full-width C, no bias.
void matmul(const float * a, const float * b, float * c, size_t m, size_t n, size_t k);
bool supported(const ggml_tensor * node);
void execute(ggml_tensor * node);

// Graph zero is prefill. Graph s>0 consumes generated token s-1 (decode step s).
void begin_graph(uint64_t graph, const std::string & dump_directory);
}
