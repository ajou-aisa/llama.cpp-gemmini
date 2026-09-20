#include <gemmini_params.h>
#include "gemmini.h"
#include "ggml-gemmini-args.h"
#include "ggml-gemmini-im2p.hpp"
#include "quants/act/meta.hpp"
#include "residual/rmd/rmd-builder.hpp"
#include "residual/rmd/rmd-executor.hpp"
#include "residual/rmd/rmd-im2p-executor.hpp"
#include <gemmini/optrace.hpp>
#include <im2p_gemmini_frontend.hpp>
#include <im2p_sim.h>

#include <algorithm>
#include <csignal>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <sys/resource.h>

namespace {
namespace trace = ggml::gemmini::optrace;
namespace rmd = ggml::gemmini::rmd;

void require(bool value, const char *message) {
    if (!value) throw std::runtime_error(message);
}

class FileLimit {
public:
    explicit FileLimit(const char *bytes) {
        require(getrlimit(RLIMIT_FSIZE, &previous_) == 0, "read file size limit");
        auto limit = previous_;
        limit.rlim_cur = std::stoull(bytes);
        previous_signal_ = std::signal(SIGXFSZ, SIG_IGN);
        require(previous_signal_ != SIG_ERR && setrlimit(RLIMIT_FSIZE, &limit) == 0,
                "set test-child file size limit");
    }
    ~FileLimit() {
        if (setrlimit(RLIMIT_FSIZE, &previous_) != 0) std::abort();
        std::signal(SIGXFSZ, previous_signal_);
    }
private:
    struct rlimit previous_{};
    void (*previous_signal_)(int) = SIG_DFL;
};

struct Fixture {
    ggml_gemmini_args_t args{};
#if GGML_GEMMINI_WEIGHT_BITS == 4
    std::vector<block_q4_hp1> blocks;
#else
    std::vector<block_q8_hp1> blocks;
#endif
    std::vector<float> output;

    Fixture(size_t m, size_t n, size_t k) : blocks(n * (k / 32)), output(m * n, -1.0f) {
        args.I = m; args.J = n; args.K = k;
        args.block_size_k = 32; args.sA = k; args.sC = n;
        args.f_out = output.data(); args.stride_f_out = n; args.col_stride_f_out = 1;
        args.residual_route = ggml::gemmini::residual::ResidualRoute::ws_packet;
        require(args.A.allocate(m, k, GGML_GEMMINI_ACTIVATION_BITS), "activation allocation");
        std::fill(args.A.bytes->begin(), args.A.bytes->end(), uint8_t{1});
        args.act_quant.storage().emplace<ggml::gemmini::quants::act::tensor::Meta>().scale = 1.0f;
        for (auto &block : blocks) {
            block.channel_scale = 1.0f; block.m = 0;
#if GGML_GEMMINI_WEIGHT_BITS == 4
            std::fill(std::begin(block.qs), std::end(block.qs), uint8_t{0x99});
#else
            std::fill(std::begin(block.qs), std::end(block.qs), int8_t{1});
#endif
        }
#if GGML_GEMMINI_WEIGHT_BITS == 4
        args.weight_format = ggml_gemmini_args_t::im2p_weight_format_t::q4_hp1;
        args.q4_hp1_blocks = blocks.data();
        args.native_block_count = blocks.size(); args.native_blocks_per_row = k / 32;
#else
        args.weight_format = ggml_gemmini_args_t::im2p_weight_format_t::q8_hp1;
        args.q8_hp1_blocks = blocks.data();
        args.q8_hp1_block_count = blocks.size(); args.q8_hp1_blocks_per_row = k / 32;
#endif
        args.native_weight_bytes = blocks.size() * sizeof(blocks.front());
        ggml::gemmini::gemmini_set_tile_ws(&args);
        const auto geometry = args.activation_geometry();
        require(geometry.ok(), "production geometry");
        args.activation_rows_per_stripe = geometry.geometry.stripe_rows;
        args.matmul_layer = "fixture.projection";
    }
};
}

int main(int argc, char **argv) {
    try {
        require(argc >= 2 && argc <= 4,
                "full|frontend-full|pipeline|residual|compact|fail-full|fail-adapter-full|fail-compact [trace-path [byte-limit]] required");
        const std::string mode = argv[1];
        const bool fail = mode == "fail-full" || mode == "fail-adapter-full" || mode == "fail-compact";
        require(mode == "full" || mode == "frontend-full" || mode == "pipeline" || mode == "residual" ||
                mode == "compact" || fail, "invalid mode");
        require(!fail || argc == 4, "failure mode needs trace path and parent_end byte limit");
        Fixture fixture(mode == "pipeline" ? 129 : 1,
                        mode == "pipeline" ? 129 : 3, mode == "pipeline" ? 96 : 32);
        std::shared_ptr<trace::Session> session;
        if (argc >= 3) {
            session = trace::Session::start(argv[2], trace::compiled_run_info(
                "official-host-dispatch-fixture", fixture.args.I, 1));
            fixture.args.optrace_context = std::make_shared<const trace::Context>(
                session->phase("prefill", std::nullopt, fixture.args.I));
        }
        std::unique_ptr<FileLimit> file_limit;
        if (fail) file_limit = std::make_unique<FileLimit>(argv[3]);
        if (mode == "frontend-full" || mode == "fail-full") {
            im2p::gemmini::Options options{};
            options.production_geometry = true;
            auto execution = im2p::gemmini::execute(&fixture.args, im2p::gemmini::Mode::full, options);
            require(execution.status.ok() && bool(execution.run), "FULL frontend start");
            const auto result = im2p::gemmini::fence(*execution.run);
            file_limit.reset();
            require(result.stats.base.work_total_cycles > 0, "FULL numerical completion precedes trace failure");
            require(fail ? !result.status.ok() : result.status.ok(), "FULL frontend status");
            require(std::all_of(fixture.output.begin(), fixture.output.end(), [fail](float value) {
                return value == (fail ? -1.0f : 32.0f);
            }), "FULL provenance failure published caller output");
        } else if (mode == "compact" || mode == "fail-compact") {
            std::vector<int8_t> activations(31, 1);
            std::vector<int32_t> weights(31 * 3, 1);
            std::vector<uint32_t> carriers(3, 0);
            std::vector<rmd::OutputValue> output(3, -777);
            rmd::detail::Im2pCompactDot dot{};
            dot.operand_bits = GGML_GEMMINI_ACTIVATION_BITS;
            dot.activations = activations.data(); dot.rows = 1;
            dot.activation_row_stride_bytes = 31;
            dot.weights = weights.data(); dot.columns = 3; dot.weight_row_stride = 3;
            dot.k = 31; dot.hp1_carriers = carriers.data();
            dot.trace_context = fixture.args.optrace_context;
            dot.trace_layer = fixture.args.matmul_layer;
            dot.source_row_count = 1;
            std::unique_ptr<im2p_sim_t, decltype(&im2p_sim_destroy)> sim(
                im2p_sim_create(), im2p_sim_destroy);
            require(bool(sim), "compact simulator");
            rmd::detail::Im2pProviderStatsAggregate stats{};
            auto status = rmd::RmdStatus::success;
            bool escaped = false;
            try {
                status = rmd::detail::execute_im2p_compact_dot(
                    sim.get(), dot, output.data(), 3, stats);
            } catch (const std::exception &error) {
                escaped = true;
                std::fprintf(stderr, "COMPACT_PROVENANCE_EXCEPTION %s\n", error.what());
            }
            file_limit.reset();
            require(std::all_of(output.begin(), output.end(), [fail](auto value) {
                return value == (fail ? -777 : 31);
            }), "compact provenance failure published caller output");
            require(!escaped, "compact provenance exception escaped status API");
            require(fail ? status == rmd::RmdStatus::execution_failed
                         : status == rmd::RmdStatus::success, "compact status");
            if (session && !fail)
                session->independent_count(*dot.trace_context, dot.trace_layer, "residual", 1);
        } else if (mode == "residual") {
            rmd::RmdStripeBuilder builder;
            builder.reset(0, 0, fixture.args.I, fixture.args.K, fixture.args.J,
                          GGML_GEMMINI_ACTIVATION_BITS);
            for (size_t k = 0; k < 31; ++k) builder.add_residual(0, k, 1);
            const auto packet = builder.finish();
            require(bool(packet), "residual packet");
            std::unique_ptr<im2p_sim_t, decltype(&im2p_sim_destroy)> sim(
                im2p_sim_create(), im2p_sim_destroy);
            require(bool(sim), "residual simulator");
            rmd::CompressedOutput expected, actual;
            rmd::RmdExecutionMetrics metrics{};
            require(rmd::execute_rmd_stripe_reference(fixture.args, *packet, expected) ==
                    rmd::RmdStatus::success, "residual reference");
            require(rmd::execute_rmd_stripe_im2p(sim.get(), fixture.args, *packet,
                    actual, &metrics) == rmd::RmdStatus::success, "residual dispatch");
            require(actual.values == expected.values && actual.domain == expected.domain &&
                    actual.j_padded == expected.j_padded, "residual numerical equality");
            require(metrics.im2p_dot_calls == 1, "K31 must remain one logical NPU GEMM");
        } else {
            const auto result = mode == "full" || mode == "fail-adapter-full"
                ? ggml::gemmini::im2p_adapter::run_full(fixture.args)
                : ggml::gemmini::im2p_adapter::run_stripe_pipeline(fixture.args);
            file_limit.reset();
            require(fail ? !result.result.ok() : result.result.ok(), result.result.message);
            require(std::all_of(fixture.output.begin(), fixture.output.end(),
                [&fixture, fail](float value) { return value == (fail ? -1.0f : float(fixture.args.K)); }),
                "FULL provenance failure published caller output");
        }
        if (session && !fail) session->finish();
        std::printf("OPTRACE_DISPATCH_PASS mode=%s trace=%s\n", mode.c_str(),
                    session ? "ON" : "OFF");
        return 0;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "OPTRACE_DISPATCH_FAIL %s\n", error.what());
        return 1;
    }
}
