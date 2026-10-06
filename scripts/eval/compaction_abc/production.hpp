#pragma once
#include "types.hpp"
#include "ggml-gemmini-args.h"
#include "residual/rmd/rmd-bitmap-builder.hpp"
#include "residual/rmd/rmd-run-aware.hpp"

namespace abc {
#if GGML_GEMMINI_WEIGHT_BITS == 4
using NativeBlock = block_q4_hp1;
#else
using NativeBlock = block_q8_hp1;
#endif
struct ProductionFixture {
    std::vector<NativeBlock> weights;
    ggml_gemmini_args_t args;
    explicit ProductionFixture(const Weight &source) {
        weights.resize(source.bytes.size() / sizeof(NativeBlock));
        std::memcpy(weights.data(), source.bytes.data(), source.bytes.size());
        args.J = source.n; args.K = source.k; args.block_size_k = 32;
        args.native_blocks_per_row = source.k / 32;
        args.native_block_count = weights.size();
        args.native_weight_bytes = source.bytes.size();
#if GGML_GEMMINI_WEIGHT_BITS == 4
        args.weight_format = ggml_gemmini_args_t::im2p_weight_format_t::q4_hp1;
        args.q4_hp1_blocks = weights.data();
#else
        args.weight_format = ggml_gemmini_args_t::im2p_weight_format_t::q8_hp1;
        args.q8_hp1_blocks = weights.data();
        args.q8_hp1_block_count = weights.size();
        args.q8_hp1_blocks_per_row = source.k / 32;
#endif
    }
    ggml::gemmini::rmd::RunAwareRequest prepare(const Input &input) const {
        namespace rmd = ggml::gemmini::rmd;
        std::vector<uint64_t> selection((input.m * input.k + 63) / 64, 0);
        for (const Event &event : input.events) {
            const size_t cell = event.row * input.k + event.k;
            selection[cell / 64] |= uint64_t{1} << (cell % 64);
        }
        rmd::RmdBitmapBuilder builder;
        builder.reset(input.stripe,input.row_begin,input.m,input.k,input.n,input.bits,selection,input.k);
        for (const Event &event : input.events)
            if (!builder.emit(event.row,event.k,event.residual)) throw std::runtime_error("production emit failed");
        const auto packet = builder.finish();
        rmd::RunAwareRequest result;
        if (!packet || rmd::build_run_aware_request(args, *packet, result) != rmd::RmdStatus::success)
            throw std::runtime_error("production request failed");
        return result;
    }
};
}
