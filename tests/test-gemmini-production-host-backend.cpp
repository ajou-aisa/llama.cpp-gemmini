#include "ggml-gemmini-args.h"
#include "ggml-gemmini-matmul.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <vector>

#if defined(GGML_GEMMINI_TESTING)
#error "production host backend oracle must not enable GGML_GEMMINI_TESTING"
#endif

int main() {
#if defined(__riscv)
    std::fprintf(stderr, "production host backend oracle requires a non-RISC-V host\n");
    return 2;
#else
    using namespace ggml::gemmini;
    constexpr size_t rows = 3;
    constexpr size_t columns = 2;
    constexpr size_t k = 32;

    ggml_gemmini_args_t args{};
    args.I = rows;
    args.J = columns;
    args.K = k;
    args.tile_I = 1;
    args.tile_J = 1;
    args.tile_K = 1;
    args.activation_rows_per_stripe = DIM;
    args.tiled_matmul_type = static_cast<tiled_matmul_type_t>(1);
    if (!args.A.allocate(rows, k, 8)) return 3;
    for (size_t row = 0; row < rows; ++row)
        for (size_t column = 0; column < k; ++column)
            if (!args.A.set(row, column, 1)) return 4;

    std::array<block_q8_h1, columns> weights{};
    for (auto &weight : weights) {
        weight.s_rf = 0.25f;
        weight.c_b = 1;
        weight.R = 1;
        std::fill(std::begin(weight.qs), std::end(weight.qs), int8_t{1});
    }
    args.weight_format = ggml_gemmini_args_t::im2p_weight_format_t::q8_h1;
    args.q8_h1_blocks = weights.data();
    args.q8_h1_block_count = weights.size();
    args.q8_h1_rows = columns;
    args.blocks_per_row = 1;
    args.native_weight_bytes = weights.size() * sizeof(block_q8_h1);
    std::vector<float> output(rows * columns, 17.0f);
    args.f_out = output.data();
    args.stride_f_out = columns;
    args.col_stride_f_out = 1;
    args.act_quant.storage().emplace<quants::act::exsia::Meta>().theta = {0};

    ResolvedMatmulOptions compact{};
    compact.mode = MatmulInvocationMode::stripe_pipeline;
    compact.job_capacity = 2;
    compact.rmd_backend = RmdBackend::gemmini_ws_compact;
    const MatmulExecution compact_execution = prepare_execution(&args, compact);

    ResolvedMatmulOptions direct = compact;
    direct.rmd_backend = RmdBackend::cpu_direct;
    const MatmulExecution direct_execution = prepare_execution(&args, direct);

    std::printf("PRODUCTION_HOST_BACKEND compact=%u direct=%u\n",
                static_cast<unsigned>(compact_execution.status().code),
                static_cast<unsigned>(direct_execution.status().code));
    return compact_execution.status().code ==
                   MatmulStatusCode::unsupported_backend &&
               direct_execution.status().ok()
               ? 0
               : 1;
#endif
}
