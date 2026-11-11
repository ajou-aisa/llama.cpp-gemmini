#pragma once

#if defined(GGML_GEMMINI_PB)
#include "ggml-gemmini.h"

#include <cstddef>
#include <cstdint>
#include <vector>

struct ggml_gemmini_args_t;

namespace ggml::gemmini::pb {
enum class Status { success, invalid_arguments, invalid_codes, invalid_scales, allocation_failure };

// Codes: A[M,K], B[J,K]. Scales: sA[M,ceil(K/32)], sB[ceil(K/32),J].
// Output strides count float elements, not bytes. Inputs and output must not alias.
Status execute(const int8_t * a, const float * s_a, const int8_t * b, const float * s_b,
               size_t m, size_t j, size_t k, float * output,
               size_t row_stride, size_t column_stride,
               ggml_gemmini_pb_node_type type = GGML_GEMMINI_PB_OTHER);
Status execute_linear(const ggml_gemmini_args_t & args);
#if defined(GGML_GEMMINI_PB_Q8_0)
// Quantizes rows of K floats with ggml's Q8_0 reference quantizer.
// Codes: rows-by-K. Scales: rows-by-(K/32), widened from the stored FP16 delta.
Status quantize_q8_0_rows(const float * data, size_t rows, size_t k,
                          std::vector<int8_t> & codes, std::vector<float> & scales);
// Runs one MUL_MAT node whose weight is Q8_0 and whose activation and output are F32.
Status execute_linear_q8_0(const ggml_tensor & node, const char * layer);
#endif
// Weight buffers call this when their contents or lifetime change.
void reset_validated_weights();

// Five-core mode, opt-in. Without start_runtime() every call runs single-threaded on the calling thread.
// With it, one issuer thread pinned to the Linux CPU of hart 0 runs every Gemmini command (flush and
// tiled_matmul_auto), and the CPU stages are split over lanes: lane 0 is the thread that calls into PB
// (the coordinator, which must not run on the issuer CPU), lanes 1..N are helper threads pinned to
// helper_cpus. Lanes own disjoint output rows or columns, so every output element still receives its
// K blocks in order with the same arithmetic. K blocks stay sequential on Gemmini.
struct RuntimeConfig {
    int issuer_cpu = -1;
    std::vector<int> helper_cpus;
};
Status start_runtime(const RuntimeConfig & config);
void stop_runtime();
bool runtime_active();

struct ThreadReport {
    const char * role;      // "issuer", "coordinator" or "helper"
    int lane;               // -1 for the issuer
    long tid;
    int cpu_assigned;       // -1 for the coordinator, which the caller pins
    int cpu_first, cpu_last;  // sched_getcpu() at the first and the latest session
    uint64_t cpu_mismatch;  // sessions off the assigned CPU; for the coordinator, sessions on the issuer CPU
    uint64_t sessions, phases, thread_cpu_ns;  // thread CPU time includes spin waiting
    uint64_t busy_prep_ns, busy_scale_ns, busy_quant_ns, busy_gemm_ns, wait_ns;
    uint64_t gemm_calls, flushes, packed_a_bytes, packed_b_bytes, scales_b, zeroed, scaled_elements, quantized_rows;
};
std::vector<ThreadReport> runtime_report();

// The first completions of the run in order: lane -1 is the issuer. Kinds: 0 flush, 1 GEMM,
// 2 zero and prepare block 0, 3 scale block `block` and prepare the next, 4 activation quantization.
struct RuntimeEvent {
    uint64_t session, seq;
    int lane, kind;
    uint64_t block, begin_ns, end_ns;
};
std::vector<RuntimeEvent> runtime_events();

// Wakes the lanes and the issuer for the duration of one PB node. Nested sessions are no-ops.
class Session {
  public:
    Session();
    ~Session();
    Session(const Session &) = delete;
    Session & operator=(const Session &) = delete;
  private:
    bool owner_ = false;
};

#if defined(GGML_GEMMINI_TESTING)
struct BlockObservation {
    size_t block, m, j, k;
    const int8_t * a;       // Packed M-by-k.
    const int8_t * b;       // Packed k-by-J, never transposed by hardware.
    const int32_t * acc;    // M-by-J.
    const float * s_a;      // M scales for this block.
    const float * s_b;      // J scales for this block.
    const float * output;   // FP32 accumulation after this block.
    size_t row_stride, column_stride;
};
using BlockObserver = void (*)(const BlockObservation &, void *);
void set_block_observer(BlockObserver observer, void * context);
uint64_t tiled_call_count();
uint64_t flush_count();
void reset_tiled_call_count();
uint64_t weight_validation_count();
#endif
} // namespace ggml::gemmini::pb
#endif
