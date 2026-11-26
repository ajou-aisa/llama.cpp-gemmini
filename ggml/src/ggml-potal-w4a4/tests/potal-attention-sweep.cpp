// Attention-only decode benchmark. Native execution is command-model admission,
// never a substitute for the RISC-V backend's dispatch counters or target time.
#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-gemmini.h"
#include "ggml-potal.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef __riscv
#include "matmul/attention.hpp"
#include "matmul/stripe-api.hpp"
#include "support/potal-emulator.hpp"
#include "trace/timeline.hpp"
#include <sched.h>
#include <thread>
#endif

namespace attnbench {
namespace fs = std::filesystem;
constexpr std::array<int, 5> lengths{{128, 256, 512, 1024, 2048}};

inline void require(bool ok, const std::string & why) {
    if (!ok) throw std::runtime_error(why);
}

// Versioned little-endian tensor container: 88-byte header, followed by logical
// elements (axis 0 fastest), not padding or the backing allocation's byte span.
struct blob {
    uint32_t type = 0, dimensions = 4;
    std::array<int64_t, 4> ne{};
    std::array<uint64_t, 4> nb{};
    std::vector<uint8_t> bytes;
};

inline blob read_blob(const fs::path & path) {
    std::ifstream f(path, std::ios::binary);
    char magic[8]{};
    blob b;
    uint64_t size = 0, elements = 1;
    f.read(magic, 8);
    f.read(reinterpret_cast<char *>(&b.type), 4);
    f.read(reinterpret_cast<char *>(&b.dimensions), 4);
    f.read(reinterpret_cast<char *>(b.ne.data()), 32);
    f.read(reinterpret_cast<char *>(b.nb.data()), 32);
    f.read(reinterpret_cast<char *>(&size), 8);
    require(f && std::memcmp(magic, "PATTN01", 8) == 0 && b.dimensions == 4 &&
            b.type <= 1, "bad tensor header: " + path.string());
    for (auto n : b.ne) {
        require(n > 0 && n <= 65536 && elements <= (1u << 26)/uint64_t(n), "tensor extent");
        elements *= uint64_t(n);
    }
    require(size == elements * (b.type == 1 ? 2 : 4), "tensor payload size");
    b.bytes.resize(size);
    f.read(reinterpret_cast<char *>(b.bytes.data()), size);
    require(bool(f) && f.peek() == EOF, "truncated or trailing tensor: " + path.string());
    return b;
}

inline void write_blob(const fs::path & path, const blob & b) {
    fs::create_directories(path.parent_path());
    std::ofstream f(path, std::ios::binary);
    const uint64_t size = b.bytes.size();
    f.write("PATTN01", 8);
    f.write(reinterpret_cast<const char *>(&b.type), 4);
    f.write(reinterpret_cast<const char *>(&b.dimensions), 4);
    f.write(reinterpret_cast<const char *>(b.ne.data()), 32);
    f.write(reinterpret_cast<const char *>(b.nb.data()), 32);
    f.write(reinterpret_cast<const char *>(&size), 8);
    f.write(reinterpret_cast<const char *>(b.bytes.data()), size);
    f.close();
    require(bool(f), "write tensor: " + path.string());
}

inline blob capture(const ggml_tensor * t) {
    blob b;
    b.type = t->type;
    std::copy(t->ne, t->ne + 4, b.ne.begin());
    std::copy(t->nb, t->nb + 4, b.nb.begin());
    const size_t width = ggml_type_size(t->type);
    b.bytes.resize(ggml_nelements(t) * width);
    size_t dst = 0;
    for (int64_t w = 0; w < t->ne[3]; ++w)
        for (int64_t z = 0; z < t->ne[2]; ++z)
            for (int64_t y = 0; y < t->ne[1]; ++y)
                for (int64_t x = 0; x < t->ne[0]; ++x, dst += width)
                    std::memcpy(b.bytes.data() + dst, static_cast<const char *>(t->data) +
                                x*t->nb[0] + y*t->nb[1] + z*t->nb[2] + w*t->nb[3], width);
    return b;
}

inline void load(ggml_tensor * t, const fs::path & path) {
    const auto b = read_blob(path);
    require(b.type == uint32_t(t->type) && std::equal(b.ne.begin(), b.ne.end(), t->ne) &&
            std::equal(b.nb.begin(), b.nb.end(), t->nb), "input layout: " + path.string());
    const size_t width = ggml_type_size(t->type);
    size_t src = 0;
    for (int64_t h = 0; h < t->ne[2]; ++h)
        for (int64_t y = 0; y < t->ne[1]; ++y)
            for (int64_t x = 0; x < t->ne[0]; ++x, src += width)
                std::memcpy(static_cast<char *>(t->data) + x*t->nb[0] + y*t->nb[1] + h*t->nb[2],
                            b.bytes.data() + src, width);
}

inline uint64_t hash(const blob & b) {
    uint64_t h = 14695981039346656037ull;
    for (uint8_t c : b.bytes) { h ^= c; h *= 1099511628211ull; }
    return h;
}

struct graph {
    ggml_context * ctx;
    ggml_backend_t cpu;
    ggml_tensor * q, * k, * v, * mask, * qk, * probability, * pv;
    ggml_cgraph * full;
    explicit graph(int l) {
        ctx = ggml_init({64*1024*1024, nullptr, false});
        require(ctx != nullptr, "ggml_init");
        cpu = ggml_backend_cpu_init();
        require(cpu != nullptr, "CPU backend init");
        ggml_backend_cpu_set_n_threads(cpu, 1);
        // llama-graph.cpp:1498-1518: n_ctx=2048 (the I25W cache), n_tokens=1.
        q = ggml_permute(ctx, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 64, 14, 1), 0, 2, 1, 3);
        auto * kb = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 128, l);
        k = ggml_view_3d(ctx, kb, 64, l, 2, 256, 128, 0);
        auto * vb = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 2048, 128);
        v = ggml_view_3d(ctx, vb, l, 64, 2, 4096, 262144, 0);
        mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, l, 32);
        ggml_set_name(k, "cache_k_l23");
        ggml_set_name(v, "cache_v_l23");
        qk = ggml_mul_mat(ctx, k, q);
        ggml_mul_mat_set_prec(qk, GGML_PREC_F32);
        ggml_gemmini_set_attention_role(qk, GGML_GEMMINI_ATTENTION_QK);
        qk->src[2] = mask;
        ggml_set_name(qk, "attn_qk-23");
        probability = ggml_soft_max_ext(ctx, qk, mask, 0.125f, 0.0f);
        pv = ggml_mul_mat(ctx, v, probability);
        ggml_gemmini_set_attention_role(pv, GGML_GEMMINI_ATTENTION_PV);
        ggml_set_name(pv, "attn_pv-23");
        full = ggml_new_graph(ctx);
        ggml_build_forward_expand(full, pv);
    }
    ~graph() { ggml_backend_free(cpu); ggml_free(ctx); }
    ggml_status compute(ggml_backend_t backend, ggml_tensor * t) {
        int index = 0;
        while (index < full->n_nodes && full->nodes[index] != t) ++index;
        require(index < full->n_nodes, "missing graph node");
        auto view = ggml_graph_view(full, index, index + 1);
        return ggml_backend_graph_compute(backend, &view);
    }
    void inputs(const fs::path & dir, int l) {
        const auto prefix = std::to_string(l) + "-";
        load(q, dir/(prefix+"Q.bin")); load(k, dir/(prefix+"K.bin"));
        load(v, dir/(prefix+"V.bin")); load(mask, dir/(prefix+"mask.bin"));
    }
    void save(const fs::path & dir) {
        for (auto item : {std::make_pair("Q",q), {"K",k}, {"V",v}, {"mask",mask},
                          {"QK",qk}, {"softmax",probability}, {"PV",pv}})
            write_blob(dir/(std::string(item.first)+".bin"), capture(item.second));
    }
};

inline uint64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
inline uint64_t cycles() {
#ifdef __riscv
    uint64_t n;
    asm volatile("rdcycle %0" : "=r"(n) :: "memory");
    return n;
#else
    return 0; // Native cycles unavailable; ns are not target cycles.
#endif
}

#ifndef __riscv
namespace attention = ggml::potal::attention;
using namespace ggml::potal;
struct model_owner final : stripe_owner {
    potal_test::emulator emu;
    potal_ws_ops ops = emu.ops();
    potal_stripe_session session = {{&ops, 0, {}}};
    std::thread thread;
    int cpu = -1;
    bool start(stripe_consumer consume, std::string &, potal_status &) override {
        thread = std::thread([this, consume] {
            std::string why;
            require(pin_role_cpu(cpu, why), "native owner affinity: " + why);
            owner_runners runners;
            runners.planned = [this](const potal_stripe_gemm_desc & d) {
                return run_planned_stripe_gemm(session, d);
            };
            (void) consume(runners);
        });
        return true;
    }
    void join() override { if (thread.joinable()) thread.join(); }
};
inline attention::result model_run(ggml_tensor * t, int l, const char * op, int rep) {
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    require(sched_getaffinity(0, sizeof(allowed), &allowed) == 0, "native affinity");
    std::array<int, 5> cpus{};
    size_t count = 0;
    for (int c = 0; c < CPU_SETSIZE && count < 5; ++c)
        if (CPU_ISSET(c, &allowed)) cpus[count++] = c;
    require(count == 5, "five native CPUs required");
    model_owner owner;
    owner.cpu = cpus[0];
    const auto r = attention::execute_strict(t, owner, cpus, true);
    require(r.status == POTAL_OK && !r.fatal && r.strict_tasks == 14 &&
            !r.cpu_work_on_issuer_thread && !r.cpu_work_on_issuer_cpu, "strict model admission");
    const auto & e = owner.emu;
    require(!e.in_flight && e.scales.empty() && e.writes == e.releases &&
            !e.scale_miss, "model DMA/scale ownership");
    if (rep == 0)
        std::printf("ATTN_MODEL kv=%d op=%s scale_write=%llu scale_release=%llu index=0 "
                    "loop_ws=%llu fences=%llu tlb=%llu hash=%016llx dispatch=COMMAND_MODEL\n",
                    l, op, (unsigned long long)e.writes, (unsigned long long)e.releases,
                    (unsigned long long)e.loops, (unsigned long long)e.fences,
                    (unsigned long long)e.tlb_flushes, (unsigned long long)hash(capture(t)));
    return r;
}
#endif

inline int sweep(int argc, char ** argv) {
    require(argc == 4, "usage: potal-attention-sweep KV INPUT_DIR OUTPUT_DIR");
    const int l = std::stoi(argv[1]);
    require(std::find(lengths.begin(), lengths.end(), l) != lengths.end(), "unsupported KV length");
    for (const char * name : {"GGML_POTAL_HART_ROLES", "GGML_POTAL_STRICT_ISSUER"})
        require(std::getenv(name) && std::string(std::getenv(name)) == "ON", std::string(name)+" must be ON");
    require(!std::getenv("GGML_POTAL_INDEX_QUEUE"), "Index Queue must be unset");
    graph g(l);
    g.inputs(argv[2], l);
#ifdef __riscv
    auto * backend = ggml_backend_potal_init();
    require(backend != nullptr, "PoTal backend init");
    require(ggml_backend_supports_op(backend, g.qk) && ggml_backend_supports_op(backend, g.pv),
            "backend refused attention graph");
    std::printf("ATTN_MODE MEASURED-FPGA cycles=calling_hart_elapsed not_accelerator_busy\n");
#else
    std::printf("ATTN_MODE MEASURED-NATIVE command_model=1 backend_dispatch_counters=unavailable cycles=unavailable\n");
#endif
    for (int rep = -1; rep < 3; ++rep) {
        uint64_t strict = 0, work = 0;
#ifdef __riscv
        ggml_backend_potal_stats before{}, after{};
        require(ggml_backend_potal_get_stats(backend, &before), "stats before");
#endif
        const auto t0 = now_ns(), c0 = cycles();
#ifdef __riscv
        require(g.compute(backend, g.qk) == GGML_STATUS_SUCCESS, "QK backend failed");
#else
        const auto qr = model_run(g.qk, l, "qk", rep);
#endif
        const auto c1 = cycles(), t1 = now_ns();
        require(g.compute(g.cpu, g.probability) == GGML_STATUS_SUCCESS, "CPU softmax failed");
        const auto c2 = cycles(), t2 = now_ns();
#ifdef __riscv
        require(g.compute(backend, g.pv) == GGML_STATUS_SUCCESS, "PV backend failed");
#else
        const auto pr = model_run(g.pv, l, "pv", rep);
        strict = 2;
        work = qr.cpu_work_on_issuer_thread + qr.cpu_work_on_issuer_cpu +
               pr.cpu_work_on_issuer_thread + pr.cpu_work_on_issuer_cpu;
#endif
        const auto c3 = cycles(), t3 = now_ns();
#ifdef __riscv
        require(ggml_backend_potal_get_stats(backend, &after), "stats after");
        strict = after.attention_strict_nodes - before.attention_strict_nodes;
        work = after.attention_issuer_cpu_work - before.attention_issuer_cpu_work;
        require(strict == 2 && work == 0 && after.attention_failed == before.attention_failed &&
                after.attention_qk_nodes == before.attention_qk_nodes + 1 &&
                after.attention_pv_nodes == before.attention_pv_nodes + 1 &&
                after.attention_completed == before.attention_completed + 2, "backend strict dispatch proof");
#endif
        if (rep < 0) continue;
        std::printf("ATTN_SWEEP kv=%d rep=%d qk_ns=%llu softmax_ns=%llu pv_ns=%llu total_ns=%llu "
                    "qk_cycles=%llu softmax_cycles=%llu pv_cycles=%llu strict_nodes=%llu issuer_cpu_work=%llu\n",
                    l, rep, (unsigned long long)(t1-t0), (unsigned long long)(t2-t1),
                    (unsigned long long)(t3-t2), (unsigned long long)(t3-t0),
                    (unsigned long long)(c1-c0), (unsigned long long)(c2-c1), (unsigned long long)(c3-c2),
                    (unsigned long long)strict, (unsigned long long)work);
        if (rep == 0) g.save(argv[3]);
    }
#ifdef __riscv
    ggml_backend_free(backend);
#endif
    return 0;
}
} // namespace attnbench

#ifndef POTAL_ATTN_SWEEP_LIBRARY
int main(int argc, char ** argv) {
    try { return attnbench::sweep(argc, argv); }
    catch (const std::exception & e) { std::fprintf(stderr, "ATTN_FAIL %s\n", e.what()); return 1; }
}
#endif
