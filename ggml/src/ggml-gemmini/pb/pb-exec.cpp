#include "pb-exec.hpp"

#if defined(GGML_GEMMINI_PB)
#include "../ggml-gemmini-args.h"
#include "../quants/common/weight_reader.hpp"
#include "ggml-quants.h"
#include <gemmini.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <ctime>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <set>
#include <stdexcept>
#include <string_view>
#include <system_error>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include <pthread.h>
#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace {
std::array<ggml_gemmini_pb_counters, GGML_GEMMINI_PB_NODE_TYPE_COUNT> counters{};

bool product(size_t a, size_t b, size_t & result) {
    return !__builtin_mul_overflow(a, b, &result);
}

ggml_gemmini_pb_node_type node_type(std::string_view name) {
    if (name == "lm_head" || name == "output.weight" || name == "unclassified.output.weight" ||
        name.find("token_embd.weight") != name.npos) return GGML_GEMMINI_PB_LM_HEAD;
    const std::pair<std::string_view, ggml_gemmini_pb_node_type> names[] = {
        {"attn.q_proj", GGML_GEMMINI_PB_Q}, {"attn_q.weight", GGML_GEMMINI_PB_Q},
        {"attn.k_proj", GGML_GEMMINI_PB_K}, {"attn_k.weight", GGML_GEMMINI_PB_K},
        {"attn.v_proj", GGML_GEMMINI_PB_V}, {"attn_v.weight", GGML_GEMMINI_PB_V},
        {"attn.out_proj", GGML_GEMMINI_PB_OUTPUT}, {"attn_output.weight", GGML_GEMMINI_PB_OUTPUT},
        {"mlp.gate_proj", GGML_GEMMINI_PB_FFN_GATE}, {"ffn_gate.weight", GGML_GEMMINI_PB_FFN_GATE},
        {"mlp.up_proj", GGML_GEMMINI_PB_FFN_UP}, {"ffn_up.weight", GGML_GEMMINI_PB_FFN_UP},
        {"mlp.down_proj", GGML_GEMMINI_PB_FFN_DOWN}, {"ffn_down.weight", GGML_GEMMINI_PB_FFN_DOWN},
    };
    for (const auto & entry : names) {
        if (name.find(entry.first) != name.npos) return entry.second;
    }
    return GGML_GEMMINI_PB_OTHER;
}
} // namespace

extern "C" bool ggml_gemmini_pb_get_counters(
        ggml_gemmini_pb_node_type type, ggml_gemmini_pb_counters * out) {
    if (!out || type < 0 || type >= GGML_GEMMINI_PB_NODE_TYPE_COUNT) return false;
    *out = counters[type];
    return true;
}

extern "C" void ggml_gemmini_pb_reset_counters(void) {
    counters = {};
}

namespace ggml::gemmini::pb {
static_assert(DIM == 32 && sizeof(elem_t) == 1 && sizeof(acc_t) == 4);

#if defined(GGML_GEMMINI_TESTING)
namespace {
BlockObserver observer = nullptr;
void * observer_context = nullptr;
uint64_t calls = 0;
uint64_t flushes = 0;
}
void set_block_observer(BlockObserver value, void * context) {
    observer = value;
    observer_context = context;
}
uint64_t tiled_call_count() { return calls; }
uint64_t flush_count() { return flushes; }
void reset_tiled_call_count() { calls = 0; }
#endif

namespace {
using WeightKey = std::tuple<const void *, size_t, size_t>;
std::set<WeightKey> validated_weights;
uint64_t weight_validations = 0;

struct ContiguousWeights {
    const int8_t * codes;
    const float * block_scales;
    size_t k;

    void pack(size_t block, size_t width, size_t j, int8_t * packed) const {
        pack_cols(block, width, j, 0, j, packed);
    }
    void pack_cols(size_t block, size_t width, size_t j, size_t c0, size_t c1, int8_t * packed) const {
        for (size_t lane = 0; lane < width; ++lane)
            for (size_t col = c0; col < c1; ++col)
                packed[lane * j + col] = codes[col * k + block * 32 + lane];
    }
    const float * scales(size_t block, size_t j, std::vector<float> &) const {
        return block_scales + block * j;
    }
    void scales_cols(size_t block, size_t j, size_t c0, size_t c1, float * out) const {
        for (size_t col = c0; col < c1; ++col) out[col] = block_scales[block * j + col];
    }
};

struct Hp1Weights {
    const block_q8_hp1 * blocks;
    size_t blocks_per_row;

    void pack(size_t block, size_t width, size_t j, int8_t * packed) const {
        pack_cols(block, width, j, 0, j, packed);
    }
    void pack_cols(size_t block, size_t width, size_t j, size_t c0, size_t c1, int8_t * packed) const {
        for (size_t col = c0; col < c1; ++col) {
            const int8_t * codes = blocks[col * blocks_per_row + block].qs;
            for (size_t lane = 0; lane < width; ++lane) packed[lane * j + col] = codes[lane];
        }
    }
    const float * scales(size_t block, size_t j, std::vector<float> & scratch) const {
        scales_cols(block, j, 0, j, scratch.data());
        return scratch.data();
    }
    void scales_cols(size_t block, size_t, size_t c0, size_t c1, float * out) const {
        for (size_t col = c0; col < c1; ++col) {
            const block_q8_hp1 & weight = blocks[col * blocks_per_row + block];
            out[col] = weight.m == INT16_MIN ? 0.0f : std::ldexp(weight.channel_scale, int(weight.m));
        }
    }
};

struct Q8_0Weights {
    const block_q8_0 * blocks;
    size_t blocks_per_row;

    void pack(size_t block, size_t width, size_t j, int8_t * packed) const {
        pack_cols(block, width, j, 0, j, packed);
    }
    void pack_cols(size_t block, size_t width, size_t j, size_t c0, size_t c1, int8_t * packed) const {
        for (size_t col = c0; col < c1; ++col) {
            const int8_t * codes = blocks[col * blocks_per_row + block].qs;
            for (size_t lane = 0; lane < width; ++lane) packed[lane * j + col] = codes[lane];
        }
    }
    const float * scales(size_t block, size_t j, std::vector<float> & scratch) const {
        scales_cols(block, j, 0, j, scratch.data());
        return scratch.data();
    }
    void scales_cols(size_t block, size_t, size_t c0, size_t c1, float * out) const {
        for (size_t col = c0; col < c1; ++col) out[col] = ggml_fp16_to_fp32(blocks[col * blocks_per_row + block].d);
    }
};

// The RoCC macros carry no memory clobber, so the compiler must be told that Gemmini accesses this buffer.
inline void gemmini_memory_barrier(const void * p) { asm volatile("" : : "r"(p) : "memory"); }

// ---- Five-core runtime (opt-in, see start_runtime) ----
constexpr size_t NONE = std::numeric_limits<size_t>::max();
constexpr size_t EVENT_CAP = 16384;
enum : int { KIND_FLUSH = 0, KIND_GEMM = 1, KIND_INIT = 2, KIND_SCALE_PREP = 3, KIND_QUANT = 4 };

inline uint64_t mono_ns() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return uint64_t(ts.tv_sec) * 1000000000ull + uint64_t(ts.tv_nsec);
}

inline void cpu_relax() {
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#else
    asm volatile("" : : : "memory");
#endif
}

bool pin_self(int cpu) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    return sched_setaffinity(0, sizeof(set), &set) == 0;
}

// Each counter has one writer, its own thread; others read it only between sessions.
struct Stat {
    std::atomic<uint64_t> v{0};
    void add(uint64_t x) { v.store(v.load(std::memory_order_relaxed) + x, std::memory_order_relaxed); }
    uint64_t get() const { return v.load(std::memory_order_relaxed); }
};

struct alignas(64) ThreadStats {
    Stat sessions, phases, busy_prep_ns, busy_scale_ns, busy_quant_ns, busy_gemm_ns, wait_ns, gemm_calls, flushes,
         packed_a_bytes, packed_b_bytes, scales_b, zeroed, scaled_elements, quantized_rows, cpu_mismatch;
    std::atomic<int> cpu_assigned{-1}, cpu_first{-1}, cpu_last{-1};
    std::atomic<long> tid{0};
};

void note_session(ThreadStats & st, int assigned, int forbidden) {
    const int cpu = sched_getcpu();
    if (st.cpu_first.load(std::memory_order_relaxed) < 0) st.cpu_first.store(cpu, std::memory_order_relaxed);
    st.cpu_last.store(cpu, std::memory_order_relaxed);
    st.sessions.add(1);
    if ((assigned >= 0 && cpu != assigned) || (forbidden >= 0 && cpu == forbidden)) st.cpu_mismatch.add(1);
}

using PhaseFn = void (*)(void * ctx, size_t lane, size_t lanes, ThreadStats & st);

struct IssuerRequest {
    int kind;
    size_t block, m, j, width;
    const int8_t * a;
    const int8_t * b;
    int32_t * c;
};

struct Runtime {
    int issuer_cpu = -1;
    std::vector<int> helper_cpus;
    size_t lanes = 1;                          // coordinator plus helpers
    std::vector<std::thread> threads;          // [0] issuer, [1..] helpers
    std::unique_ptr<ThreadStats[]> stats;      // [0] issuer, [1] coordinator, [2..] helpers
    std::mutex m;
    std::condition_variable cv;
    bool stop = false;
    std::atomic<uint64_t> session_seq{0};
    std::atomic<bool> in_session{false};
    alignas(64) std::atomic<uint64_t> phase_seq{0};
    alignas(64) std::atomic<uint64_t> phase_done{0};  // cumulative helper completions
    PhaseFn phase_fn = nullptr;
    void * phase_ctx = nullptr;
    int phase_kind = 0;
    uint64_t phase_block = 0;
    alignas(64) std::atomic<uint64_t> req_seq{0};
    alignas(64) std::atomic<uint64_t> req_done{0};
    IssuerRequest req{};
    std::atomic<int> ready{0}, failed{0};
    pthread_t coordinator{};
    std::atomic<bool> have_coordinator{false};
    std::unique_ptr<RuntimeEvent[]> events;
    std::atomic<size_t> event_count{0};
};

Runtime * runtime = nullptr;
thread_local int session_depth = 0;

void record_event(Runtime & r, int lane, int kind, uint64_t block, uint64_t begin, uint64_t end) {
    if (r.event_count.load(std::memory_order_relaxed) >= EVENT_CAP) return;
    const size_t n = r.event_count.fetch_add(1, std::memory_order_relaxed);
    if (n < EVENT_CAP) r.events[n] = {r.session_seq.load(std::memory_order_relaxed), n, lane, kind, block, begin, end};
}

bool wait_for_session(Runtime & r, uint64_t & seen) {
    std::unique_lock<std::mutex> lock(r.m);
    r.cv.wait(lock, [&] { return r.stop || r.session_seq.load(std::memory_order_relaxed) != seen; });
    if (r.stop) return false;
    seen = r.session_seq.load(std::memory_order_relaxed);
    return true;
}

// Runs every Gemmini command of the process while the runtime is active.
void issuer_main(Runtime & r) {
    ThreadStats & st = r.stats[0];
    st.tid.store(long(syscall(SYS_gettid)), std::memory_order_relaxed);
    st.cpu_assigned.store(r.issuer_cpu, std::memory_order_relaxed);
    if (!pin_self(r.issuer_cpu) || sched_getcpu() != r.issuer_cpu) r.failed.fetch_add(1);
    r.ready.fetch_add(1);
    uint64_t seen = 0, last = 0;
    while (wait_for_session(r, seen)) {
        note_session(st, r.issuer_cpu, -1);
        uint64_t mark = mono_ns();
        for (;;) {
            const uint64_t q = r.req_seq.load(std::memory_order_acquire);
            if (q != last) {
                const uint64_t begin = mono_ns();
                st.wait_ns.add(begin - mark);
                last = q;
                const IssuerRequest rq = r.req;
                std::atomic_thread_fence(std::memory_order_seq_cst);
                if (rq.kind == KIND_FLUSH) {
                    // Linux sfence.vma does not reach Gemmini's TLB, and per-call buffers can land on remapped pages.
                    gemmini_flush(0);
                    st.flushes.add(1);
#if defined(GGML_GEMMINI_TESTING)
                    ++flushes;
#endif
                } else {
                    gemmini_memory_barrier(rq.a);
                    gemmini_memory_barrier(rq.b);
                    tiled_matmul_auto(rq.m, rq.j, rq.width, rq.a, rq.b, nullptr, rq.c,
                                      rq.width, rq.j, 0, rq.j,
                                      MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
                                      NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
                                      false, false, false, true, false, 0, WS);
                    gemmini_memory_barrier(rq.c);
                    st.gemm_calls.add(1);
#if defined(GGML_GEMMINI_TESTING)
                    ++calls;
#endif
                }
                std::atomic_thread_fence(std::memory_order_seq_cst);
                const uint64_t end = mono_ns();
                st.busy_gemm_ns.add(end - begin);
                record_event(r, -1, rq.kind, rq.block, begin, end);
                r.req_done.store(q, std::memory_order_release);
                mark = end;
                continue;
            }
            if (!r.in_session.load(std::memory_order_acquire)) {
                st.wait_ns.add(mono_ns() - mark);
                break;
            }
            cpu_relax();
        }
    }
}

void helper_main(Runtime & r, size_t lane, int cpu) {
    ThreadStats & st = r.stats[lane + 1];
    st.tid.store(long(syscall(SYS_gettid)), std::memory_order_relaxed);
    st.cpu_assigned.store(cpu, std::memory_order_relaxed);
    if (!pin_self(cpu) || sched_getcpu() != cpu) r.failed.fetch_add(1);
    r.ready.fetch_add(1);
    uint64_t seen = 0, last = 0;
    while (wait_for_session(r, seen)) {
        note_session(st, cpu, -1);
        uint64_t mark = mono_ns();
        for (;;) {
            const uint64_t p = r.phase_seq.load(std::memory_order_acquire);
            if (p != last) {
                const uint64_t begin = mono_ns();
                st.wait_ns.add(begin - mark);
                last = p;
                r.phase_fn(r.phase_ctx, lane, r.lanes, st);
                const uint64_t end = mono_ns();
                record_event(r, int(lane), r.phase_kind, r.phase_block, begin, end);
                st.phases.add(1);
                r.phase_done.fetch_add(1, std::memory_order_release);
                mark = end;
                continue;
            }
            if (!r.in_session.load(std::memory_order_acquire)) {
                st.wait_ns.add(mono_ns() - mark);
                break;
            }
            cpu_relax();
        }
    }
}

// Coordinator: runs lane 0 of a phase, then waits until every helper finished it.
void run_phase(Runtime & r, PhaseFn fn, void * ctx, int kind, uint64_t block) {
    ThreadStats & st = r.stats[1];
    r.phase_fn = fn;
    r.phase_ctx = ctx;
    r.phase_kind = kind;
    r.phase_block = block;
    const uint64_t seq = r.phase_seq.load(std::memory_order_relaxed) + 1;
    r.phase_seq.store(seq, std::memory_order_release);
    const uint64_t begin = mono_ns();
    fn(ctx, 0, r.lanes, st);
    const uint64_t end = mono_ns();
    record_event(r, 0, kind, block, begin, end);
    st.phases.add(1);
    const uint64_t want = seq * (r.lanes - 1);
    while (r.phase_done.load(std::memory_order_acquire) != want) cpu_relax();
    st.wait_ns.add(mono_ns() - end);
}

// Coordinator: hands one Gemmini command to the issuer and waits for its completion.
void issue(Runtime & r, const IssuerRequest & rq) {
    ThreadStats & st = r.stats[1];
    r.req = rq;
    const uint64_t q = r.req_seq.load(std::memory_order_relaxed) + 1;
    r.req_seq.store(q, std::memory_order_release);
    const uint64_t begin = mono_ns();
    while (r.req_done.load(std::memory_order_acquire) != q) cpu_relax();
    st.wait_ns.add(mono_ns() - begin);
}

std::pair<size_t, size_t> split(size_t n, size_t lane, size_t lanes, size_t align) {
    const size_t units = (n + align - 1) / align;
    return {std::min(n, units * lane / lanes * align), std::min(n, units * (lane + 1) / lanes * align)};
}

template <class Weights>
struct BlockJob {
    const int8_t * a;
    const float * s_a;
    const Weights * weights;
    size_t m, j, k, blocks;
    float * output;
    size_t row_stride, column_stride;
    int8_t * pa[2];
    int8_t * pb[2];
    float * sa[2];
    float * sb[2];
    const int32_t * acc;
    bool zero;
    size_t scale_block, prep_block;
};

// One lane's share of a phase. Output elements are owned by one lane (whole rows when every lane gets
// one, else column ranges), so each element gets its K blocks in order, with the single-thread arithmetic.
// Preparing block b+1 writes the other slot, which block b's Gemmini command and scaling never read.
template <class Weights>
void block_phase(void * p, size_t lane, size_t lanes, ThreadStats & st) {
    const auto & job = *static_cast<const BlockJob<Weights> *>(p);
    const bool by_rows = job.m >= lanes;
    const auto rows = by_rows ? split(job.m, lane, lanes, 1) : std::pair<size_t, size_t>{0, job.m};
    const auto cols = by_rows ? std::pair<size_t, size_t>{0, job.j} : split(job.j, lane, lanes, 16);
    const uint64_t begin = mono_ns();
    uint64_t mid = begin;
    if (job.zero) {
        for (size_t i = rows.first; i < rows.second; ++i)
            for (size_t col = cols.first; col < cols.second; ++col)
                job.output[i * job.row_stride + col * job.column_stride] = +0.0f;
        st.zeroed.add((rows.second - rows.first) * (cols.second - cols.first));
    }
    if (job.scale_block != NONE) {
        const size_t s = job.scale_block % 2;
        const float * scales_a = job.sa[s];
        const float * scales_b = job.sb[s];
        for (size_t i = rows.first; i < rows.second; ++i) {
            for (size_t col = cols.first; col < cols.second; ++col) {
                float t = static_cast<float>(job.acc[i * job.j + col]);
#if defined(GGML_GEMMINI_PB_Q8_0)
                t = t * (scales_a[i] * scales_b[col]);
#else
                t = t * scales_a[i];
                t = t * scales_b[col];
#endif
                float & c = job.output[i * job.row_stride + col * job.column_stride];
                c = c + t;
            }
        }
        mid = mono_ns();
        st.busy_scale_ns.add(mid - begin);
        st.scaled_elements.add((rows.second - rows.first) * (cols.second - cols.first));
    }
    if (job.prep_block != NONE) {
        const size_t block = job.prep_block, s = block % 2, width = std::min(size_t{32}, job.k - block * 32);
        const auto arows = split(job.m, lane, lanes, 1);
        for (size_t i = arows.first; i < arows.second; ++i) {
            std::copy_n(job.a + i * job.k + block * 32, width, job.pa[s] + i * width);
            job.sa[s][i] = job.s_a[i * job.blocks + block];
        }
        const auto bcols = split(job.j, lane, lanes, 64);
        job.weights->pack_cols(block, width, job.j, bcols.first, bcols.second, job.pb[s]);
        job.weights->scales_cols(block, job.j, bcols.first, bcols.second, job.sb[s]);
        st.packed_a_bytes.add((arows.second - arows.first) * width);
        st.packed_b_bytes.add((bcols.second - bcols.first) * width);
        st.scales_b.add(bcols.second - bcols.first);
    }
    st.busy_prep_ns.add(mono_ns() - mid);
}

template <class Weights>
Status run_blocks_five(Runtime & r, const int8_t * a, const float * s_a, const Weights & weights,
                       size_t m, size_t j, size_t k, size_t blocks, size_t c_count,
                       float * output, size_t row_stride, size_t column_stride,
                       ggml_gemmini_pb_node_type type) {
    Session session;
    try {
        size_t packed_a_count, packed_b_count;
        if (!product(m, std::min(k, size_t{32}), packed_a_count) ||
            !product(j, std::min(k, size_t{32}), packed_b_count)) return Status::invalid_arguments;
        std::vector<int8_t> pa0(packed_a_count), pa1(packed_a_count), pb0(packed_b_count), pb1(packed_b_count);
        std::vector<int32_t> acc(c_count);
        std::vector<float> sa0(m), sa1(m), sb0(j), sb1(j);
        BlockJob<Weights> job{a, s_a, &weights, m, j, k, blocks, output, row_stride, column_stride,
                              {pa0.data(), pa1.data()}, {pb0.data(), pb1.data()},
                              {sa0.data(), sa1.data()}, {sb0.data(), sb1.data()}, acc.data(), true, NONE, 0};
        issue(r, {KIND_FLUSH, 0, 0, 0, 0, nullptr, nullptr, nullptr});
        run_phase(r, &block_phase<Weights>, &job, KIND_INIT, 0);
        job.zero = false;
        for (size_t block = 0; block < blocks; ++block) {
            const size_t s = block % 2, width = std::min(size_t{32}, k - block * 32);
            issue(r, {KIND_GEMM, block, m, j, width, job.pa[s], job.pb[s], acc.data()});
            ++counters[type].k_blocks;
            counters[type].int32_elements += c_count;
            job.scale_block = block;
            job.prep_block = block + 1 < blocks ? block + 1 : NONE;
            run_phase(r, &block_phase<Weights>, &job, KIND_SCALE_PREP, block);
#if defined(GGML_GEMMINI_TESTING)
            if (observer) observer({block, m, j, width, job.pa[s], job.pb[s], acc.data(),
                                    job.sa[s], job.sb[s], output, row_stride, column_stride}, observer_context);
#endif
        }
        ++counters[type].nodes;
        return Status::success;
    } catch (const std::bad_alloc &) {
        return Status::allocation_failure;
    } catch (const std::length_error &) {
        return Status::allocation_failure;
    }
}

#if defined(GGML_GEMMINI_PB_Q8_0)
struct QuantJob {
    const float * data;
    size_t rows, k, blocks;
    int8_t * codes;
    float * scales;
    block_q8_0 * scratch;  // one row of blocks per lane
    std::atomic<bool> nonfinite{false}, bad_scale{false};
};

void quant_phase(void * p, size_t lane, size_t lanes, ThreadStats & st) {
    auto & q = *static_cast<QuantJob *>(p);
    const auto rows = split(q.rows, lane, lanes, 1);
    const uint64_t begin = mono_ns();
    bool finite = true;
    for (size_t n = rows.first * q.k; finite && n < rows.second * q.k; ++n) finite = std::isfinite(q.data[n]);
    if (!finite) {
        q.nonfinite.store(true, std::memory_order_relaxed);
    } else {
        block_q8_0 * row_blocks = q.scratch + lane * q.blocks;
        for (size_t row = rows.first; row < rows.second; ++row) {
            quantize_row_q8_0_ref(q.data + row * q.k, row_blocks, int64_t(q.k));
            for (size_t block = 0; block < q.blocks; ++block) {
                const float scale = ggml_fp16_to_fp32(row_blocks[block].d);
                if (!std::isfinite(scale)) q.bad_scale.store(true, std::memory_order_relaxed);
                q.scales[row * q.blocks + block] = scale;
                std::copy_n(row_blocks[block].qs, QK8_0, q.codes + row * q.k + block * QK8_0);
            }
        }
        st.quantized_rows.add(rows.second - rows.first);
    }
    st.busy_quant_ns.add(mono_ns() - begin);
}
#endif

void shutdown_runtime(Runtime & r) {
    {
        std::lock_guard<std::mutex> lock(r.m);
        r.stop = true;
    }
    r.cv.notify_all();
    for (auto & t : r.threads)
        if (t.joinable()) t.join();
}

template <class Weights>
Status run_blocks(const int8_t * a, const float * s_a, const Weights & weights,
                  size_t m, size_t j, size_t k, size_t blocks, size_t c_count,
                  float * output, size_t row_stride, size_t column_stride,
                  ggml_gemmini_pb_node_type type) {
    if (runtime) return run_blocks_five(*runtime, a, s_a, weights, m, j, k, blocks, c_count,
                                        output, row_stride, column_stride, type);
    try {
        size_t packed_a_count, packed_b_count;
        if (!product(m, std::min(k, size_t{32}), packed_a_count) ||
            !product(j, std::min(k, size_t{32}), packed_b_count)) return Status::invalid_arguments;
        std::vector<int8_t> packed_a(packed_a_count), packed_b(packed_b_count);
        std::vector<int32_t> acc(c_count);
        std::vector<float> scales_a(m), scratch_b(j);
        for (size_t i = 0; i < m; ++i)
            for (size_t col = 0; col < j; ++col) output[i * row_stride + col * column_stride] = +0.0f;
        // Linux sfence.vma does not reach Gemmini's TLB, and these per-call buffers can land on remapped pages.
        gemmini_flush(0);
#if defined(GGML_GEMMINI_TESTING)
        ++flushes;
#endif
        for (size_t block = 0; block < blocks; ++block) {
            const size_t width = std::min(size_t{32}, k - block * 32);
            for (size_t i = 0; i < m; ++i) {
                std::copy_n(a + i * k + block * 32, width, packed_a.data() + i * width);
                scales_a[i] = s_a[i * blocks + block];
            }
            weights.pack(block, width, j, packed_b.data());
            const float * scales_b = weights.scales(block, j, scratch_b);
            gemmini_memory_barrier(packed_a.data());
            gemmini_memory_barrier(packed_b.data());
            tiled_matmul_auto(m, j, width, packed_a.data(), packed_b.data(), nullptr, acc.data(),
                              width, j, 0, j,
                              MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
                              NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
                              false, false, false, true, false, 0, WS);
            gemmini_memory_barrier(acc.data());
#if defined(GGML_GEMMINI_TESTING)
            ++calls;
#endif
            ++counters[type].k_blocks;
            counters[type].int32_elements += c_count;
            for (size_t i = 0; i < m; ++i) {
                for (size_t col = 0; col < j; ++col) {
                    float t = static_cast<float>(acc[i * j + col]);
#if defined(GGML_GEMMINI_PB_Q8_0)
                    // ggml_vec_dot_q8_0_q8_0 multiplies the two block scales before the integer sum.
                    t = t * (scales_a[i] * scales_b[col]);
#else
                    t = t * scales_a[i];
                    t = t * scales_b[col];
#endif
                    float & c = output[i * row_stride + col * column_stride];
                    c = c + t;
                }
            }
#if defined(GGML_GEMMINI_TESTING)
            if (observer) observer({block, m, j, width, packed_a.data(), packed_b.data(), acc.data(),
                                    scales_a.data(), scales_b, output, row_stride, column_stride}, observer_context);
#endif
        }
        ++counters[type].nodes;
        return Status::success;
    } catch (const std::bad_alloc &) {
        return Status::allocation_failure;
    } catch (const std::length_error &) {
        return Status::allocation_failure;
    }
}

Status validate_weights_once(const ggml_gemmini_args_t & args, size_t stored_blocks) {
    const WeightKey key{args.q8_hp1_blocks, args.J, args.K};
    if (validated_weights.count(key)) return Status::success;
    namespace qr = quants::wreader;
    namespace route = quants::wroute;
    const auto plan = route::resolve_weight_route_plan(args, route::WeightScaleInfoMode::CommonOutput);
    if (!plan.valid || plan.scale_domain != route::WeightScaleDomain::FloatingBlock ||
        qr::validate(args, plan) != qr::WeightReaderStatus::Success) return Status::invalid_arguments;
    for (size_t n = 0; n < stored_blocks; ++n)
        for (const int8_t code : args.q8_hp1_blocks[n].qs)
            if (code == -128) return Status::invalid_codes;
    validated_weights.insert(key);
    ++weight_validations;
    return Status::success;
}

#if defined(GGML_GEMMINI_PB_Q8_0)
Status validate_q8_0_once(const block_q8_0 * blocks, size_t count, size_t j, size_t k) {
    const WeightKey key{blocks, j, k};
    if (validated_weights.count(key)) return Status::success;
    for (size_t n = 0; n < count; ++n) {
        if (!std::isfinite(ggml_fp16_to_fp32(blocks[n].d))) return Status::invalid_scales;
        for (const int8_t code : blocks[n].qs)
            if (code == -128) return Status::invalid_codes;
    }
    validated_weights.insert(key);
    ++weight_validations;
    return Status::success;
}
#endif
} // namespace

void reset_validated_weights() { validated_weights.clear(); }

Status start_runtime(const RuntimeConfig & config) {
    if (runtime) return Status::invalid_arguments;
    std::vector<int> cpus = config.helper_cpus;
    cpus.push_back(config.issuer_cpu);
    for (const int cpu : cpus)
        if (cpu < 0 || cpu >= CPU_SETSIZE) return Status::invalid_arguments;
    std::sort(cpus.begin(), cpus.end());
    if (std::adjacent_find(cpus.begin(), cpus.end()) != cpus.end()) return Status::invalid_arguments;
    std::unique_ptr<Runtime> r;
    try {
        r = std::make_unique<Runtime>();
        r->issuer_cpu = config.issuer_cpu;
        r->helper_cpus = config.helper_cpus;
        r->lanes = 1 + config.helper_cpus.size();
        r->stats = std::make_unique<ThreadStats[]>(r->lanes + 1);
        r->events = std::make_unique<RuntimeEvent[]>(EVENT_CAP);
        r->threads.reserve(r->lanes);
        r->threads.emplace_back(issuer_main, std::ref(*r));
        for (size_t i = 0; i < config.helper_cpus.size(); ++i)
            r->threads.emplace_back(helper_main, std::ref(*r), i + 1, config.helper_cpus[i]);
    } catch (const std::bad_alloc &) {
        if (r) shutdown_runtime(*r);
        return Status::allocation_failure;
    } catch (const std::system_error &) {
        shutdown_runtime(*r);
        return Status::allocation_failure;
    }
    while (size_t(r->ready.load()) != r->threads.size()) std::this_thread::yield();
    if (r->failed.load()) {
        shutdown_runtime(*r);
        return Status::invalid_arguments;
    }
    runtime = r.release();
    return Status::success;
}

void stop_runtime() {
    if (!runtime) return;
    shutdown_runtime(*runtime);
    delete runtime;
    runtime = nullptr;
}

bool runtime_active() { return runtime != nullptr; }

std::vector<ThreadReport> runtime_report() {
    std::vector<ThreadReport> out;
    Runtime * r = runtime;
    if (!r) return out;
    for (size_t n = 0; n <= r->lanes; ++n) {
        const ThreadStats & st = r->stats[n];
        ThreadReport t{};
        t.role = n == 0 ? "issuer" : n == 1 ? "coordinator" : "helper";
        t.lane = int(n) - 1;
        t.tid = st.tid.load(std::memory_order_relaxed);
        t.cpu_assigned = st.cpu_assigned.load(std::memory_order_relaxed);
        t.cpu_first = st.cpu_first.load(std::memory_order_relaxed);
        t.cpu_last = st.cpu_last.load(std::memory_order_relaxed);
        t.cpu_mismatch = st.cpu_mismatch.get();
        t.sessions = st.sessions.get();
        t.phases = st.phases.get();
        t.busy_prep_ns = st.busy_prep_ns.get();
        t.busy_scale_ns = st.busy_scale_ns.get();
        t.busy_quant_ns = st.busy_quant_ns.get();
        t.busy_gemm_ns = st.busy_gemm_ns.get();
        t.wait_ns = st.wait_ns.get();
        t.gemm_calls = st.gemm_calls.get();
        t.flushes = st.flushes.get();
        t.packed_a_bytes = st.packed_a_bytes.get();
        t.packed_b_bytes = st.packed_b_bytes.get();
        t.scales_b = st.scales_b.get();
        t.zeroed = st.zeroed.get();
        t.scaled_elements = st.scaled_elements.get();
        t.quantized_rows = st.quantized_rows.get();
        pthread_t thread{};
        bool have = true;
        if (n == 1) {
            have = r->have_coordinator.load(std::memory_order_acquire);
            thread = r->coordinator;
        } else {
            thread = r->threads[n == 0 ? 0 : n - 1].native_handle();
        }
        clockid_t clock;
        timespec ts;
        if (have && pthread_getcpuclockid(thread, &clock) == 0 && clock_gettime(clock, &ts) == 0)
            t.thread_cpu_ns = uint64_t(ts.tv_sec) * 1000000000ull + uint64_t(ts.tv_nsec);
        out.push_back(t);
    }
    return out;
}

std::vector<RuntimeEvent> runtime_events() {
    Runtime * r = runtime;
    if (!r) return {};
    const size_t n = std::min(r->event_count.load(), EVENT_CAP);
    return std::vector<RuntimeEvent>(r->events.get(), r->events.get() + n);
}

Session::Session() {
    Runtime * r = runtime;
    if (!r) return;
    if (session_depth++ > 0) return;
    owner_ = true;
    ThreadStats & st = r->stats[1];
    st.tid.store(long(syscall(SYS_gettid)), std::memory_order_relaxed);
    r->coordinator = pthread_self();
    r->have_coordinator.store(true, std::memory_order_release);
    note_session(st, -1, r->issuer_cpu);
    {
        std::lock_guard<std::mutex> lock(r->m);
        r->in_session.store(true, std::memory_order_release);
        r->session_seq.fetch_add(1, std::memory_order_release);
    }
    r->cv.notify_all();
}

Session::~Session() {
    if (!runtime) return;
    --session_depth;
    if (owner_) runtime->in_session.store(false, std::memory_order_release);
}

#if defined(GGML_GEMMINI_TESTING)
uint64_t weight_validation_count() { return weight_validations; }
#endif

Status execute(const int8_t * a, const float * s_a, const int8_t * b, const float * s_b,
               size_t m, size_t j, size_t k, float * output,
               size_t row_stride, size_t column_stride, ggml_gemmini_pb_node_type type) {
    size_t a_count, b_count, c_count, as_count, bs_count, row_end, col_end, end;
    const size_t blocks = k / 32 + (k % 32 != 0);
    if (!a || !b || !s_a || !s_b || !output || !m || !j || !k ||
        !row_stride || !column_stride || type < 0 || type >= GGML_GEMMINI_PB_NODE_TYPE_COUNT ||
        !product(m, k, a_count) || !product(j, k, b_count) || !product(m, j, c_count) ||
        !product(m, blocks, as_count) || !product(blocks, j, bs_count) ||
        !product(m - 1, row_stride, row_end) || !product(j - 1, column_stride, col_end) ||
        __builtin_add_overflow(row_end, col_end, &end) ||
        end >= std::numeric_limits<size_t>::max() / sizeof(float)) {
        return Status::invalid_arguments;
    }
    for (size_t n = 0; n < a_count; ++n) if (a[n] == -128) return Status::invalid_codes;
    for (size_t n = 0; n < b_count; ++n) if (b[n] == -128) return Status::invalid_codes;
    for (size_t n = 0; n < as_count; ++n)
#if defined(GGML_GEMMINI_PB_Q8_0)
        // An all-zero Q8_0 block has scale zero.
        if (!std::isfinite(s_a[n]) || s_a[n] < 0.0f) return Status::invalid_scales;
#else
        if (!std::isfinite(s_a[n]) || s_a[n] <= 0.0f) return Status::invalid_scales;
#endif
    for (size_t n = 0; n < bs_count; ++n)
        if (!std::isfinite(s_b[n])) return Status::invalid_scales;
    return run_blocks(a, s_a, ContiguousWeights{b, s_b, k}, m, j, k, blocks, c_count,
                      output, row_stride, column_stride, type);
}

Status execute_linear(const ggml_gemmini_args_t & args) {
    const auto * meta = std::get_if<quants::act::block::Meta>(&args.act_quant.storage());
    const size_t blocks = args.K / 32;
    size_t a_count, c_count, stored_blocks, s_count;
    if (!args.I || !args.J || !args.K || args.K % 32 || !args.mxint_weight ||
        args.weight_format != ggml_gemmini_args_t::im2p_weight_format_t::q8_hp1 ||
        args.tiled_matmul_type != WS || args.A.bits != 8 || !args.A.valid() ||
        args.A.rows != args.I || args.A.cols != args.K || !args.f_out || !meta ||
        meta->cols != args.K || args.activation_row_offset > meta->rows ||
        args.I > meta->rows - args.activation_row_offset ||
        !product(args.I, args.K, a_count) || !product(args.I, args.J, c_count) ||
        !product(args.J, blocks, stored_blocks) || !args.q8_hp1_blocks ||
        args.q8_hp1_blocks_per_row != blocks || args.q8_hp1_block_count < stored_blocks ||
        !product(meta->rows, blocks, s_count) || meta->scales.size() != s_count) {
        return Status::invalid_arguments;
    }
    const Status weight_status = validate_weights_once(args, stored_blocks);
    if (weight_status != Status::success) return weight_status;
    const float * s_a = meta->scales.data() + args.activation_row_offset * blocks;
    for (size_t n = 0; n < args.I * blocks; ++n)
        if (!std::isfinite(s_a[n]) || s_a[n] <= 0.0f) return Status::invalid_scales;
    try {
        std::vector<int8_t> a(a_count);
        for (size_t i = 0; i < args.I; ++i)
            for (size_t k = 0; k < args.K; ++k) {
                const auto code = args.A.get(i, k);
                if (code < -127 || code > 127) return Status::invalid_codes;
                a[i * args.K + k] = static_cast<int8_t>(code);
            }
        return run_blocks(a.data(), s_a, Hp1Weights{args.q8_hp1_blocks, blocks},
                          args.I, args.J, args.K, blocks, c_count, args.f_out,
                          args.stride_f_out ? args.stride_f_out : args.J,
                          args.col_stride_f_out ? args.col_stride_f_out : 1, node_type(args.matmul_layer));
    } catch (const std::bad_alloc &) {
        return Status::allocation_failure;
    } catch (const std::length_error &) {
        return Status::allocation_failure;
    }
}

#if defined(GGML_GEMMINI_PB_Q8_0)
Status quantize_q8_0_rows(const float * data, size_t rows, size_t k,
                          std::vector<int8_t> & codes, std::vector<float> & scales) {
    const size_t blocks = k / QK8_0;
    size_t count, scale_count;
    if (!data || !rows || !k || k % QK8_0 || !product(rows, k, count) ||
        !product(rows, blocks, scale_count)) return Status::invalid_arguments;
    if (runtime) {
        // Rows are independent, so each lane quantizes whole rows with the same reference quantizer.
        Session session;
        codes.resize(count);
        scales.resize(scale_count);
        std::vector<block_q8_0> scratch(runtime->lanes * blocks);
        QuantJob job{data, rows, k, blocks, codes.data(), scales.data(), scratch.data()};
        run_phase(*runtime, &quant_phase, &job, KIND_QUANT, 0);
        if (job.nonfinite.load()) return Status::invalid_codes;
        if (job.bad_scale.load()) return Status::invalid_scales;
        return Status::success;
    }
    for (size_t n = 0; n < count; ++n)
        if (!std::isfinite(data[n])) return Status::invalid_codes;
    codes.resize(count);
    scales.resize(scale_count);
    std::vector<block_q8_0> row_blocks(blocks);
    for (size_t row = 0; row < rows; ++row) {
        quantize_row_q8_0_ref(data + row * k, row_blocks.data(), int64_t(k));
        for (size_t block = 0; block < blocks; ++block) {
            const float scale = ggml_fp16_to_fp32(row_blocks[block].d);
            if (!std::isfinite(scale)) return Status::invalid_scales;
            scales[row * blocks + block] = scale;
            std::copy_n(row_blocks[block].qs, QK8_0, codes.data() + row * k + block * QK8_0);
        }
    }
    return Status::success;
}

Status execute_linear_q8_0(const ggml_tensor & node, const char * layer) {
    const ggml_tensor * weight = node.src[0];
    const ggml_tensor * activation = node.src[1];
    if (!weight || !activation || weight->type != GGML_TYPE_Q8_0 || activation->type != GGML_TYPE_F32 ||
        node.type != GGML_TYPE_F32 || !weight->data || !activation->data || !node.data)
        return Status::invalid_arguments;
    const size_t k = weight->ne[0], j = weight->ne[1], m = ggml_nrows(activation);
    const size_t blocks = k / QK8_0;
    size_t c_count, stored_blocks;
    if (k % QK8_0 || !product(m, j, c_count) || !product(j, blocks, stored_blocks))
        return Status::invalid_arguments;
    const auto * stored = static_cast<const block_q8_0 *>(weight->data);
    const Status weight_status = validate_q8_0_once(stored, stored_blocks, j, k);
    if (weight_status != Status::success) return weight_status;
    Session session;
    try {
        std::vector<int8_t> a;
        std::vector<float> s_a;
        const Status quantized = quantize_q8_0_rows(static_cast<const float *>(activation->data), m, k, a, s_a);
        if (quantized != Status::success) return quantized;
        return run_blocks(a.data(), s_a.data(), Q8_0Weights{stored, blocks}, m, j, k, blocks, c_count,
                          static_cast<float *>(node.data), j, 1, node_type(layer));
    } catch (const std::bad_alloc &) {
        return Status::allocation_failure;
    } catch (const std::length_error &) {
        return Status::allocation_failure;
    }
}
#endif
} // namespace ggml::gemmini::pb
#endif
