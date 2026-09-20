#include <gemmini_params.h>
#include "gemmini.h"
#include "ggml-gemmini-args.h"
#include "ggml-gemmini-im2p.hpp"
#include "quants/act/meta.hpp"
#include "residual/rmd/rmd-builder.hpp"
#include "residual/rmd/rmd-executor.hpp"
#include "residual/rmd/rmd-im2p-executor.hpp"
#include <gemmini/optrace.hpp>
#include <im2p_sim.h>

#include <algorithm>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace trace = ggml::gemmini::optrace;
namespace rmd = ggml::gemmini::rmd;

void require(bool value, const char *message) {
    if (!value) throw std::runtime_error(message);
}

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
        require(argc == 2 || argc == 3, "full|pipeline|residual [trace-path] required");
        const std::string mode = argv[1];
        require(mode == "full" || mode == "pipeline" || mode == "residual", "invalid mode");
        Fixture fixture(mode == "pipeline" ? 129 : 1,
                        mode == "pipeline" ? 129 : 3, mode == "pipeline" ? 96 : 32);
        std::shared_ptr<trace::Session> session;
        if (argc == 3) {
            session = trace::Session::start(argv[2], trace::compiled_run_info(
                "official-host-dispatch-fixture", fixture.args.I, 1));
            fixture.args.optrace_context = std::make_shared<const trace::Context>(
                session->phase("prefill", std::nullopt, fixture.args.I));
        }
        if (mode == "residual") {
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
            const auto result = mode == "full"
                ? ggml::gemmini::im2p_adapter::run_full(fixture.args)
                : ggml::gemmini::im2p_adapter::run_stripe_pipeline(fixture.args);
            require(result.result.ok(), result.result.message);
            require(std::all_of(fixture.output.begin(), fixture.output.end(),
                [&fixture](float value) { return value == float(fixture.args.K); }),
                "dense numerical equality");
        }
        if (session) session->finish();
        std::printf("OPTRACE_DISPATCH_PASS mode=%s trace=%s\n", mode.c_str(),
                    session ? "ON" : "OFF");
        return 0;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "OPTRACE_DISPATCH_FAIL %s\n", error.what());
        return 1;
    }
}
