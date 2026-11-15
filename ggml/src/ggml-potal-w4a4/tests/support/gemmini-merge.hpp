#pragma once

// The Gemmini checked merge the stripe pipeline ran before
// residual::merge_prepared_hp1 (host tests and potal-residual-merge-bench):
// rmd::detail::merge_rmd_correction_with_metadata on a merge view of the
// node's arguments, with an ExSIA Meta holding the stripe's theta only. Every
// call resolves the HP1 route plan, which runs wreader::validate() over the
// whole weight, and reads every block's column scale: the bitwise reference
// and the "before" of the prepared merge.

#include "ggml-gemmini-args.h"
#include "quants/act/exsia/exsia-event.hpp"
#include "quants/act/exsia/types.hpp"
#include "quants/act/meta.hpp"
#include "residual/rmd/rmd-compose.hpp"
#include "residual/rmd/rmd-types.hpp"

#include <cstddef>
#include <cstdint>
#include <new>
#include <utility>

namespace potal_test {

// The fields of args a residual merge reads, and nothing the activation
// quantizer writes: I/J/K, the Q8_HP1 weight view, the layer identity, the
// tile geometry and the residual route; f_out = out (row stride out_stride,
// column stride 1). No activation buffer, no metadata, no stripe sink.
inline void init_merge_args(const ggml_gemmini_args_t & args, float * out, size_t out_stride,
                            ggml_gemmini_args_t & merge) {
    merge.I                          = args.I;
    merge.J                          = args.J;
    merge.K                          = args.K;
    merge.sA                         = args.sA;
    merge.sC                         = args.sC;
    merge.residual_route             = args.residual_route;
    merge.weight_format              = args.weight_format;
    merge.q8_hp1_blocks              = args.q8_hp1_blocks;
    merge.q8_hp1_blocks_per_row      = args.q8_hp1_blocks_per_row;
    merge.q8_hp1_block_count         = args.q8_hp1_block_count;
    merge.native_weight_bytes        = args.native_weight_bytes;
    merge.block_size_k               = args.block_size_k;
    merge.f_out                      = out;
    merge.col_stride_f_out           = 1;
    merge.stride_f_out               = out_stride;
    merge.model_arch                 = args.model_arch;
    merge.matmul_layer               = args.matmul_layer;
    merge.tile_I                     = args.tile_I;
    merge.tile_J                     = args.tile_J;
    merge.tile_K                     = args.tile_K;
    merge.activation_rows_per_stripe = args.activation_rows_per_stripe;
}

// Rows [row_begin, row_end) of merge.f_out += correction, stripe theta only;
// the correction is moved in, as the pipeline's merge did.
inline ggml::gemmini::rmd::RmdStatus gemmini_merge(
    const ggml_gemmini_args_t & merge, size_t row_begin, size_t row_end, uint64_t run_id,
    const ggml::gemmini::quants::act::exsia::StripeMetadataSnapshot & snapshot,
    ggml::gemmini::rmd::BlockScaledInt64Correction && correction, size_t * nonzero) try {
    namespace exsia = ggml::gemmini::quants::act::exsia;
    ggml::gemmini::quants::act::Meta meta;
    exsia::Meta &                    e = meta.storage().emplace<exsia::Meta>();
    e.run_id                           = run_id;
    e.e_s                              = snapshot.e_s;
    e.rho                              = snapshot.rho;
    e.sigma                            = snapshot.sigma;
    e.theta.assign(1, snapshot.theta);
    const ggml::gemmini::rmd::Correction tagged = std::move(correction);
    return ggml::gemmini::rmd::detail::merge_rmd_correction_with_metadata(merge, row_begin, row_end, tagged, meta,
                                                                          nonzero, nullptr);
} catch (const std::bad_alloc &) {
    return ggml::gemmini::rmd::RmdStatus::allocation_failure;
}

} // namespace potal_test
