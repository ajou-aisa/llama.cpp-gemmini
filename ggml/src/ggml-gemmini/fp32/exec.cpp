#include "exec.hpp"
#include "dump.hpp"
#include "../ggml-gemmini-buffer.hpp"
#include "ggml-gemmini.h"
#include "ggml-impl.h"
#include "sdk.hpp"

#include <algorithm>
#include <array>
#include <condition_variable>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>

namespace ggml::gemmini::fp32 {
namespace {
Order numeric_order = Order::chunk_fma;
Counters counts;
uint64_t graph_id = 0;
std::string dump_directory;
using Key = std::tuple<const void *, uint64_t, int64_t, int64_t, int>;
std::map<Key, std::vector<float>> weights;

void require(bool condition, const char * why) {
    if (!condition) GGML_ABORT("Gemmini-FP32: %s", why);
}
void pin(int cpu) {
    require(cpu >= 0 && cpu < CPU_SETSIZE, "invalid CPU");
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(cpu, &mask);
    if (sched_setaffinity(0, sizeof(mask), &mask) || sched_getcpu() != cpu)
        throw std::runtime_error("FP32 cannot pin CPU " + std::to_string(cpu));
}
struct Runtime {
    std::mutex mutex;
    std::condition_variable cv;
    bool stop = false, command = false, completed = false;
    size_t sequence = 0, ready = 0, finished = 0, work = 0;
    bool failed = false;
    std::function<void(size_t, size_t)> phase;
    std::function<void()> issue;
    std::vector<std::thread> threads;
    std::array<ThreadReport, 5> report{};
    int main_cpu = -1;
};
std::unique_ptr<Runtime> runtime;

void worker(Runtime & r, size_t lane, int cpu) {
    try { pin(cpu); } catch (...) {
        std::lock_guard<std::mutex> lock(r.mutex);
        r.failed = true;
    }
    std::unique_lock<std::mutex> lock(r.mutex);
    r.report[lane].cpu = cpu;
    ++r.ready;
    r.cv.notify_all();
    size_t seen = 0;
    for (;;) {
        r.cv.wait(lock, [&] { return r.stop || (lane == 4 ? r.command : seen != r.sequence); });
        if (r.stop) return;
        const size_t n = r.work;
        auto phase = r.phase;
        auto issue = r.issue;
        seen = r.sequence;
        lock.unlock();
        r.report[lane].mismatch += sched_getcpu() != cpu;
        if (lane == 4) issue();
        else phase(n * lane / 4, n * (lane + 1) / 4);
        ++r.report[lane].jobs;
        if (lane != 4) r.report[lane].elements += n * (lane + 1) / 4 - n * lane / 4;
        lock.lock();
        if (lane == 4) { r.command = false; r.completed = true; }
        else ++r.finished;
        r.cv.notify_all();
    }
}
void lanes(size_t n, const std::function<void(size_t, size_t)> & fn) {
    if (!runtime) { fn(0, n); return; }
    Runtime & r = *runtime;
    require(sched_getcpu() == r.main_cpu, "coordinator moved off its lane CPU");
    {
        std::lock_guard<std::mutex> lock(r.mutex);
        r.work = n; r.phase = fn; r.finished = 0; ++r.sequence;
    }
    r.cv.notify_all();
    fn(0, n / 4);
    ++r.report[0].jobs;
    r.report[0].elements += n / 4;
    r.report[0].mismatch += sched_getcpu() != r.main_cpu;
    std::unique_lock<std::mutex> lock(r.mutex);
    r.cv.wait(lock, [&] { return r.finished == 3; });
    r.phase = {};
}
float value(const ggml_tensor & t, size_t k, size_t row, size_t head, size_t batch) {
    const char * p = static_cast<const char *>(t.data) + k*t.nb[0] + row*t.nb[1] + head*t.nb[2] + batch*t.nb[3];
    if (t.type == GGML_TYPE_F16) {
        ggml_fp16_t x;
        std::memcpy(&x, p, sizeof(x));
        return ggml_fp16_to_fp32(x);
    }
    float x;
    std::memcpy(&x, p, sizeof(x));
    return x;
}
std::vector<float> pack(const ggml_tensor & t, size_t head, size_t batch, bool transpose) {
    const size_t k = size_t(t.ne[0]), rows = size_t(t.ne[1]);
    std::vector<float> result(rows * k);
    lanes(rows, [&](size_t begin, size_t end) {
        for (size_t row = begin; row < end; ++row)
            for (size_t z = 0; z < k; ++z)
                result[transpose ? z * rows + row : row * k + z] = value(t, z, row, head, batch);
    });
    return result;
}
const std::vector<float> & weight(const ggml_tensor & t) {
    const ggml_tensor * storage = &t;
    while (!storage->buffer && storage->view_src) storage = storage->view_src;
    require(is_gemmini_buffer(storage->buffer) &&
            ggml_backend_buffer_get_usage(storage->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS,
            "linear weight must use an invalidation-aware Gemmini weight buffer");
    const Key key{t.data, gemmini_buffer_generation(storage->buffer), t.ne[0], t.ne[1], int(t.type)};
    auto it = weights.find(key);
    if (it == weights.end()) {
        auto packed = pack(t, 0, 0, true);
        counts.cached_bytes += packed.size() * sizeof(float);
        ++counts.conversions;
        it = weights.emplace(key, std::move(packed)).first;
    }
    return it->second;
}
void write(std::ofstream & file, const float * data, size_t count) {
    file.write(reinterpret_cast<const char *>(data), std::streamsize(count * sizeof(float)));
    require(bool(file), "dump write failed");
}
}

void set_order(Order order) {
    require(order == Order::chunk_fma || order == Order::serial_fma || order == Order::chunk_reverse_fma,
            "unknown numeric order");
    numeric_order = order;
}
Order get_order() { return numeric_order; }
const char * order_name() {
    switch (numeric_order) {
    case Order::chunk_fma: return "chunk_fma";
    case Order::serial_fma: return "serial_fma";
    case Order::chunk_reverse_fma: return "chunk_reverse_fma";
    }
    std::abort();
}
Counters counters() { return counts; }
void reset_counts() {
    const uint64_t conversions = counts.conversions, bytes = counts.cached_bytes;
    counts = {};
    counts.conversions = conversions;
    counts.cached_bytes = bytes;
}
void reset_weights() { weights.clear(); counts.cached_bytes = 0; }
void begin_graph(uint64_t graph, const std::string & directory) {
    graph_id = graph;
    dump_directory = directory;
    reset_counts();
}
void start_runtime(int issuer_cpu, const std::vector<int> & cpus) {
    require(!runtime && cpus.size() == 4, "runtime requires exactly four lane CPUs");
#if defined(__riscv)
    // Gemmini cannot take page faults (its DMA ignores TLB pf/ae): keep every page resident before use.
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) throw std::runtime_error("FP32 mlockall failed");
#endif
    auto all = cpus;
    all.push_back(issuer_cpu);
    std::sort(all.begin(), all.end());
    require(std::adjacent_find(all.begin(), all.end()) == all.end(), "runtime CPUs must be distinct");
    pin(cpus.back());
    auto r = std::make_unique<Runtime>();
    r->main_cpu = cpus.back();
    r->report[0].cpu = r->main_cpu;
    try {
        for (size_t i = 1; i < 4; ++i) r->threads.emplace_back(worker, std::ref(*r), i, cpus[i - 1]);
        r->threads.emplace_back(worker, std::ref(*r), 4, issuer_cpu);
        std::unique_lock<std::mutex> lock(r->mutex);
        r->cv.wait(lock, [&] { return r->ready == 4; });
        if (r->failed) throw std::runtime_error("FP32 worker affinity failed");
    } catch (...) {
        { std::lock_guard<std::mutex> lock(r->mutex); r->stop = true; }
        r->cv.notify_all();
        for (auto & t : r->threads) t.join();
        throw;
    }
    runtime = std::move(r);
}
void stop_runtime() {
    if (!runtime) return;
    { std::lock_guard<std::mutex> lock(runtime->mutex); runtime->stop = true; }
    runtime->cv.notify_all();
    for (auto & t : runtime->threads) t.join();
    runtime.reset();
}
std::vector<ThreadReport> runtime_report() {
    if (!runtime) return {};
    return {runtime->report.begin(), runtime->report.end()};
}

void matmul(const float * a, const float * b, float * c, size_t m, size_t n, size_t k) {
    require(a && b && c && m && n && k, "empty matmul");
    const auto fn = [=] {
        // Linux sfence.vma does not reach Gemmini's TLB. Per-call staging may be remapped.
        gemmini_flush(0);
        asm volatile("" : : "r"(a), "r"(b) : "memory");
        tiled_matmul_auto(m, n, k, a, b, nullptr, c, k, n, 0, n,
                          1, 1, 1, NO_ACTIVATION, 1, 0, false, false, false, true, false, 0, WS);
        asm volatile("" : : "r"(c) : "memory");
    };
    if (!runtime) fn();
    else {
        Runtime & r = *runtime;
        require(sched_getcpu() == r.main_cpu, "GEMM called outside coordinator lane");
        std::unique_lock<std::mutex> lock(r.mutex);
        r.issue = fn; r.completed = false; r.command = true;
        r.cv.notify_all();
        r.cv.wait(lock, [&] { return r.completed; });
        r.issue = {};
    }
    ++counts.gemms;
    ++counts.flushes;
}

bool supported(const ggml_tensor * node) {
    if (!node || node->op != GGML_OP_MUL_MAT || !node->src[0] || !node->src[1] ||
        node->type != GGML_TYPE_F32 || !ggml_is_contiguous(node)) return false;
    const auto & a = *node->src[0];
    const auto & b = *node->src[1];
    if ((a.type != GGML_TYPE_F16 && a.type != GGML_TYPE_F32) || b.type != GGML_TYPE_F32) return false;
    for (int d = 0; d < 4; ++d)
        if (a.ne[d] <= 0 || b.ne[d] <= 0) return false;
    if (a.ne[0] != b.ne[0] || b.ne[2] % a.ne[2] || b.ne[3] % a.ne[3] ||
        node->ne[0] != a.ne[1] || node->ne[1] != b.ne[1] ||
        node->ne[2] != b.ne[2] || node->ne[3] != b.ne[3]) return false;
    const auto role = ggml_gemmini_get_attention_role(node);
    if (role == GGML_GEMMINI_ATTENTION_QK) return a.ne[0] == 64;
    if (role == GGML_GEMMINI_ATTENTION_PV) return a.ne[1] == 64;
    if (a.ne[2] != 1 || a.ne[3] != 1 || b.ne[2] != 1 || b.ne[3] != 1 || !ggml_is_contiguous(&a)) return false;
    return (a.ne[0] == 896 && (a.ne[1] == 896 || a.ne[1] == 128 || a.ne[1] == 4864 || a.ne[1] == 151936)) ||
           (a.ne[0] == 4864 && a.ne[1] == 896);
}
void execute(ggml_tensor * node) {
    if (!supported(node)) GGML_ABORT("Gemmini-FP32 unsupported mul_mat '%s'", node ? node->name : "<null>");
    const auto & a = *node->src[0];
    const auto & b = *node->src[1];
    require(a.data && b.data && node->data, "unallocated tensor");
    const auto role = ggml_gemmini_get_attention_role(node);
    const bool linear = role == GGML_GEMMINI_ATTENTION_NONE;
    const uint64_t layer = role == GGML_GEMMINI_ATTENTION_QK ? counts.qk : counts.pv;
    if (linear) ++counts.linear;
    else if (role == GGML_GEMMINI_ATTENTION_QK) ++counts.qk;
    else ++counts.pv;
    const size_t m = size_t(b.ne[1]), n = size_t(a.ne[1]), k = size_t(a.ne[0]);
    const size_t heads = size_t(b.ne[2]), batches = size_t(b.ne[3]);
    const std::vector<float> * cached = linear ? &weight(a) : nullptr;
    std::ofstream dump;
    if (!dump_directory.empty() && (graph_id == 0 || graph_id == 1 || graph_id == 8)) {
        const std::string name(a.name);
        bool selected = false;
        std::string point;
        if (linear) {
            for (const int l : {0, 23})
                for (const char * op : {"attn_q", "ffn_down"})
                    if (name == "blk." + std::to_string(l) + "." + op + ".weight") {
                        selected = true; point = "blk." + std::to_string(l) + "." + op;
                    }
        } else if (layer == 0 || layer == 23) {
            selected = true;
            point = "blk." + std::to_string(layer) + (role == GGML_GEMMINI_ATTENTION_QK ? ".qk" : ".pv");
        }
        if (selected) {
            std::filesystem::create_directories(dump_directory);
            dump.open(dump_directory + "/graph-" + std::to_string(graph_id) + "-" + point + ".bin", std::ios::binary);
            require(bool(dump), "cannot create dump");
            DumpHeader header;
            header.graph = graph_id; header.order = uint64_t(numeric_order);
            header.role = linear ? 0 : role == GGML_GEMMINI_ATTENTION_QK ? 1 : 2;
            header.m = m; header.n = n; header.k = k; header.heads = heads * batches;
            std::strncpy(header.weight, a.name, sizeof(header.weight) - 1);
            dump.write(reinterpret_cast<const char *>(&header), sizeof(header));
        }
    }
    for (size_t batch = 0; batch < batches; ++batch)
        for (size_t head = 0; head < heads; ++head) {
            auto pa = pack(b, head, batch, false);
            std::vector<float> temporary;
            if (!linear) temporary = pack(a, head / size_t(b.ne[2] / a.ne[2]), batch / size_t(b.ne[3] / a.ne[3]), true);
            const float * pb = linear ? cached->data() : temporary.data();
            float * out = static_cast<float *>(node->data) + (batch * heads + head) * m * n;
            matmul(pa.data(), pb, out, m, n, k);
            if (dump.is_open()) {
                write(dump, pa.data(), pa.size());
                if (!linear) write(dump, pb, k * n);
                write(dump, out, m * n);
            }
        }
}
}
