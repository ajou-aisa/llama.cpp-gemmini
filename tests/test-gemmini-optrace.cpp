#include <gemmini/optrace.hpp>

#include <cassert>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <thread>
#include <vector>

namespace trace = ggml::gemmini::optrace;

static trace::RunInfo info(unsigned bits = 8, unsigned dim = 16) {
    trace::RunInfo r;
    r.model = "fixture-model";
    r.profile = "a" + std::to_string(bits) + "w" + std::to_string(bits) +
        "-d" + std::to_string(dim) + "-hp1";
    r.activation_bits = r.weight_bits = bits;
    r.dim = dim;
    r.backend = "IM2P_SIM/GEMMINI_HP1";
    r.mode = "FULL";
    r.prompt_tokens = 256;
    r.requested_generated_tokens = 5;
    for (const char *name : {"IM2P.sim", "llama.cpp-gemmini", "headers"}) {
        r.source_commits[name] = std::string(40, '1');
        r.source_worktree_sha256[name] = std::string(64, '2');
    }
    return r;
}

static trace::Work work(unsigned bits = 8, unsigned dim = 16) {
    trace::Work w;
    w.layer = "blk.0.\"projection\"\n";
    w.activation_bits = w.weight_bits = bits; w.dim = dim;
    w.m = w.geometry_m = w.row_count = 2; w.n = 3; w.k = 3072;
    w.tile_i_count = 1; w.tile_j_count = 2; w.tile_k_count = 3;
    w.activation_stride_bytes = 3072; w.weight_stride_bytes = 3;
    w.output_stride_bytes = 12; w.scale_stride_elements = 3;
    return w;
}

static std::string read(const std::filesystem::path &p) {
    std::ifstream f(p); return {std::istreambuf_iterator<char>(f), {}};
}
static void rejects(const std::function<void()> &f) {
    bool rejected = false;
    try { f(); } catch (const std::exception &) { rejected = true; }
    assert(rejected);
}

int main(int argc, char **argv) {
    assert(argc == 2);
    const std::filesystem::path root = std::filesystem::absolute(argv[1]);
    std::filesystem::create_directories(root);
    assert(!trace::Session::start(nullptr, info()));
    assert(!trace::Session::start("", info()));
    assert(!trace::current_context());
    for (unsigned bits : {4u, 8u}) for (unsigned dim : {16u, 32u, 64u}) {
        const auto tag = std::to_string(bits) + "-" + std::to_string(dim);
        for (unsigned repeat = 0; repeat != 2; ++repeat) {
            const auto path = root / (tag + "-" + std::to_string(repeat) + ".jsonl");
            auto session = trace::Session::start(path.c_str(), info(bits, dim));
            auto context = session->phase("prefill", std::nullopt, 256);
            assert(!trace::current_context());
            {
                trace::ScopedContext bind(context);
                assert(trace::current_context().session == session);
                // Thread-local scope is not implicitly inherited. The owner
                // must carry its immutable context into each worker.
                std::thread worker([context, bits, dim] {
                    assert(!trace::current_context());
                    trace::ScopedContext bind(context);
                    auto w = work(bits, dim);
                    context.session->accepted(trace::current_context(), w);
                    context.session->independent_count(context, w.layer, "dense_main", 1);
                });
                worker.join();
            }
            assert(!trace::current_context());
            auto decode = session->phase("decode", 0, 1);
            auto w = work(bits, dim);
            w.m = w.geometry_m = w.row_count = 1;
            session->accepted(decode, w);
            session->independent_count(decode, w.layer, "dense_main", 1);
            session->finish();
            rejects([&] { session->accepted(decode, w); });
            rejects([&] { trace::Session::start(path.c_str(), info(bits, dim)); });
        }
        assert(read(root/(tag+"-0.jsonl")) == read(root/(tag+"-1.jsonl")));
    }
    const auto bad = [&](const char *name, const std::function<void(trace::Work &)> &mutate) {
        auto s = trace::Session::start((root/(std::string(name)+".jsonl")).c_str(), info());
        auto c = s->phase("prefill", std::nullopt, 256);
        auto w = work(); mutate(w);
        rejects([&] { s->accepted(c, w); });
        s->finish(false, "intentional negative fixture");
    };
    bad("zero-tile", [](auto &w) { w.tile_k_count = 0; });
    bad("profile", [](auto &w) { w.dim = 32; });
    bad("residual-raw", [](auto &w) { w.provenance = "residual"; w.rmd_raw = true; });
    bad("host-multiply", [](auto &w) { w.host_integer_block_multiply = true; });
    bad("stripe-range", [](auto &w) { w.scope = "stripe"; w.row_begin = 3; w.stripe_id = 0; });
    {
        auto s = trace::Session::start((root/"count-mismatch.jsonl").c_str(), info());
        auto c = s->phase("prefill", std::nullopt, 256);
        s->accepted(c, work());
        rejects([&] { s->finish(); });
    }
    {
        auto s = trace::Session::start((root/"wrong-phase.jsonl").c_str(), info());
        rejects([&] { s->phase("decode", 0, 1); });
        auto c = s->phase("prefill", std::nullopt, 256);
        rejects([&] { s->phase("decode", 1, 1); });
        s->phase("decode", 0, 1);
        rejects([&] { s->accepted(c, work()); });
        s->finish(false, "intentional phase negative");
    }
    rejects([&] { trace::Session::start((root/"missing-parent"/"x.jsonl").c_str(), trace::RunInfo{}); });
    std::cout << "OPTRACE_WRITER_PASS six profiles, deterministic serialization, explicit worker context, exclusive output, schema/count/phase negatives\n";
}
