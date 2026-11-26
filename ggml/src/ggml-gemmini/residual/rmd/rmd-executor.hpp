#pragma once

#include "../../../ggml-npu-common/quantization/common/weight_route.hpp"
#include "../../../ggml-npu-common/residual/rmd/rmd-types.hpp"
#include "../../../ggml-npu-common/residual/metrics.hpp"

#include <array>
#include <limits>
#include <mutex>
#include "../../../ggml-npu-common/profiling/include/npu/cpu-timing.h"

struct ggml_gemmini_args_t;
namespace ggml::npu::residual {
struct DirectStripePayload;
}

namespace ggml::gemmini::rmd {
using namespace ggml::npu::rmd;
namespace detail { using namespace ggml::npu::rmd::detail; }
namespace detail {
struct RmdAssemblerAccess;
}

constexpr bool compact_rmd_backend_available(bool hardware_target) {
    return hardware_target;
}

// Final integer tile result (SCU-scaled for production HP1). Physical tile
// order is executor-internal; the assembler validates tags before storing
// compressed output or composing the correction.
struct PhysicalTile {
    uint32_t            packet_block_index = 0;
    uint8_t             lane_position      = 0;
    uint8_t             lane_id            = 0;
    uint32_t            m_tile             = 0; // tile within this lane's packed nonzero rows
    uint32_t            j_tile             = 0;
    uint16_t            valid_rows         = 0;
    uint16_t            valid_cols         = 0;
    const OutputValue * values             = nullptr; // valid_rows rows, row stride kArrayDim
};

// Accepts physical tiles in any order and rejects duplicates, missing tiles and
// out-of-range tags.
class RmdOutputAssembler {
  public:
    RmdStatus begin(const StripePacket & packet, CompressedOutput & output);
    RmdStatus begin(const StripePacket & packet, Correction & correction);
    RmdStatus submit(const PhysicalTile & tile);
    RmdStatus finish();

  private:
    friend struct detail::RmdAssemblerAccess;
    RmdStatus             begin_validated(const StripePacket & packet, CompressedOutput & output);
    RmdStatus             begin_validated(const StripePacket & packet, Correction & correction);
    RmdStatus             begin(const StripePacket & packet);
    const StripePacket *  packet_     = nullptr;
    CompressedOutput *    output_     = nullptr;
    Correction *          correction_ = nullptr;
    std::vector<__int128> correction_values_;
    std::vector<uint8_t>  seen_;
    std::vector<size_t>   tile_offset_; // per block: index of its first tile slot
    size_t                j_tiles_   = 0;
    size_t                expected_  = 0;
    size_t                submitted_ = 0;
};

void collect_packet_metrics(const StripePacket & packet, RmdExecutionMetrics & metrics);
void collect_direct_metrics(const ggml::npu::residual::DirectStripePayload & payload,
                            RmdExecutionMetrics &                 metrics);

namespace detail {
RmdStatus execute_rmd_stripe_ws_with_weights(const ggml_gemmini_args_t & args,
                                             const StripePacket &        packet,
                                             Correction &                correction,
                                             RmdWeightPreparation &      weights,
                                             RmdExecutionMetrics *       metrics = nullptr);

} // namespace detail

// Production HP1 executes a whole compact group K on the NPU, including SCU and
// signed32 fragment accumulation. Host work is radix composition and final
// float reconstruction only. Legacy non-HP1 routes remain explicitly separate;
// the checked-software HP1 oracle is available only through the test entry
// point.
RmdStatus execute_rmd_stripe_ws(const ggml_gemmini_args_t & args,
                                const StripePacket &        packet,
                                CompressedOutput &          output,
                                RmdExecutionMetrics *       metrics = nullptr);

// Composes block-scaled tiles as they arrive and publishes the final INT64
// correction only after all tiles and overflow checks succeed.
RmdStatus execute_rmd_stripe_ws(const ggml_gemmini_args_t & args,
                                const StripePacket &        packet,
                                Correction &                correction,
                                RmdExecutionMetrics *       metrics = nullptr);

#if defined(GGML_GEMMINI_TESTING)
// Scalar packet oracle for tests only. Production callers cannot select or
// invoke it.
RmdStatus execute_rmd_stripe_reference(const ggml_gemmini_args_t & args,
                                       const StripePacket &        packet,
                                       CompressedOutput &          output,
                                       RmdExecutionMetrics *       metrics = nullptr);

RmdStatus execute_rmd_stripe_reference(const ggml_gemmini_args_t & args,
                                       const StripePacket &        packet,
                                       Correction &                correction,
                                       RmdExecutionMetrics *       metrics = nullptr);

// Instantiates the native Gemmini path in host test builds. Widened codes are
// preflighted against npu_element_t and fail before tiled_matmul or metric commit.
RmdStatus execute_rmd_stripe_gemmini_for_test(const ggml_gemmini_args_t & args,
                                              const StripePacket &        packet,
                                              CompressedOutput &          output,
                                              RmdExecutionMetrics *       metrics = nullptr);

RmdStatus gather_weight_tile_for_test(const ggml_gemmini_args_t & args,
                                      uint32_t                    block_id,
                                      const uint16_t *            local_k,
                                      size_t                      valid_k,
                                      size_t                      col_base,
                                      size_t                      valid_cols,
                                      npu_element_t *                    tile,
                                      size_t                      tile_stride,
                                      RmdExecutionMetrics *       metrics = nullptr);

RmdStatus gather_wide_weight_tile_for_test(const ggml_gemmini_args_t & args,
                                           uint32_t                    block_id,
                                           const uint16_t *            local_k,
                                           size_t                      valid_k,
                                           size_t                      col_base,
                                           size_t                      valid_cols,
                                           int32_t *                   tile,
                                           size_t                      tile_stride,
                                           RmdExecutionMetrics *       metrics = nullptr);

RmdStatus repeat_weight_tile_gather_for_test(const ggml_gemmini_args_t & args,
                                             uint32_t                    block_count,
                                             const uint16_t *            local_k,
                                             size_t                      valid_k,
                                             size_t                      col_base,
                                             size_t                      valid_cols,
                                             size_t                      iterations,
                                             uint64_t &                  checksum,
                                             RmdExecutionMetrics &       metrics);

RmdStatus repeat_scalar_weight_tile_gather_for_test(const ggml_gemmini_args_t & args,
                                                    uint32_t                    block_count,
                                                    const uint16_t *            local_k,
                                                    size_t                      valid_k,
                                                    size_t                      col_base,
                                                    size_t                      valid_cols,
                                                    size_t                      iterations,
                                                    uint64_t &                  checksum);
#endif

} // namespace ggml::gemmini::rmd
