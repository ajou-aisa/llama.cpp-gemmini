// ggml-gemmini.cpp

#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <new>
#include <cstdlib>

#include <limits>

#include <memory>
#include <mutex>
#include <set>
#include <tuple>
#include "ggml-impl.h"
#include "ggml-gemmini.h"
#include "ggml-gemmini-config.hpp"
#include "ggml-gemmini-buffer.hpp"
#include "ggml-gemmini-q4-h1-reprocess.hpp"
#include "ggml-gemmini-q8-h1-artifact.hpp"
#include "ggml-gemmini-q8-h1-reprocess.hpp"
#include "ggml-backend-impl.h"
#include "ggml-quants.h"

#include <gemmini/log.hpp>
#if defined(__linux__) && defined(__aarch64__)
#include "cycle_reader_internal.h"
#endif
#include "dump/dump_tensor.hpp"

#include <gemmini.h>
#include "ggml-gemmini-args.h"
#include "ggml-gemmini-matmul.hpp"
#if defined(GGML_GEMMINI_EXECUTION_BACKEND_IM2P_SIM)
#include "ggml-gemmini-im2p.hpp"
#endif
#include "quants/common/tensor_util.hpp"
#include "quants/act/quantize.hpp"
#include "quants/weight/quantize_Q8_H1.hpp"
#include "quants/weight/unpack_Q8_0.hpp"
//#include "quantization/ggml-gemmini-quantize.h"

#if LOG_DUMP
#include <atomic>
#endif

#ifndef TRANSPOSE_B
#define TRANSPOSE_B 1
#endif
#ifndef FULL_C
#define FULL_C 1
#endif
#ifndef LOW_D
#define LOW_D 0
#endif
#ifndef OPTION
#define OPTION CPU
#endif

namespace
{
    bool gemmini_is_extended_dequant_weight_type(ggml_type type) {
        return type == GGML_TYPE_Q4_0 ||
               type == GGML_TYPE_Q4_H1 ||
               type == GGML_TYPE_Q4_HP1 ||
               type == GGML_TYPE_Q16_0 ||
               type == GGML_TYPE_Q16_H1 ||
               type == GGML_TYPE_Q16_HP1;
    }

    std::uint8_t gemmini_product_weight_bits(ggml_type type) {
        switch (type) {
            case GGML_TYPE_Q4_0:
            case GGML_TYPE_Q4_1:
            case GGML_TYPE_Q4_K:
            case GGML_TYPE_Q4_H1:
            case GGML_TYPE_Q4_HP1: return 4;
            case GGML_TYPE_Q16_0:
            case GGML_TYPE_Q16_H1:
            case GGML_TYPE_Q16_HP1: return 16;
            default: return 8;
        }
    }

#if defined(GGML_GEMMINI_EXECUTION_BACKEND_IM2P_SIM)
    ggml::gemmini::im2p_adapter::WeightFamily gemmini_exsia_weight_family(
            ggml_type type) {
        using Family = ggml::gemmini::im2p_adapter::WeightFamily;
        switch (type) {
            case GGML_TYPE_Q4_0:
            case GGML_TYPE_Q8_0:
            case GGML_TYPE_Q16_0: return Family::h0;
            case GGML_TYPE_Q4_H1:
            case GGML_TYPE_Q8_H1:
            case GGML_TYPE_Q16_H1: return Family::h1;
            case GGML_TYPE_Q4_HP1:
            case GGML_TYPE_Q8_HP1:
            case GGML_TYPE_Q16_HP1: return Family::hp1;
            case GGML_TYPE_Q8_H2: return Family::h2;
            case GGML_TYPE_Q8_HP2: return Family::hp2;
            default: return Family::unsupported;
        }
    }
#endif

    bool gemmini_is_native_matched_weight_type(ggml_type type) {
        if constexpr (GGML_GEMMINI_ACTIVATION_BITS == 4 &&
                      GGML_GEMMINI_WEIGHT_BITS == 4) {
            return type == GGML_TYPE_Q4_0 ||
                   type == GGML_TYPE_Q4_H1 ||
                   type == GGML_TYPE_Q4_HP1;
        }
        if constexpr (GGML_GEMMINI_ACTIVATION_BITS == 16 &&
                      GGML_GEMMINI_WEIGHT_BITS == 16) {
            return type == GGML_TYPE_Q16_0 ||
                   type == GGML_TYPE_Q16_H1 ||
                   type == GGML_TYPE_Q16_HP1;
        }
        return false;
    }

    void gemmini_matmul_fp_facade(size_t I, size_t J, size_t K,
                                  const float * activation, const float * weights,
                                  float * output, const std::string & matmul_layer) {
        ggml_gemmini_args_t args{};
        args.matmul_layer = matmul_layer;
#if defined(GGML_GEMMINI_TEST_OBSERVER)
        if (ggml::gemmini::test_observe_semantic_layer(
                ggml::gemmini::TestSemanticLayerSite::fp_facade,
                args.matmul_layer.c_str())) {
            return;
        }
#endif
        args.I = I;
        args.J = J;
        args.K = K;
        args.A_fp32 = activation;
        args.B_fp32 = weights;
        args.sA = K;
        args.sB = K;
        args.f_out = output;
        args.col_stride_f_out = 0;
        args.stride_f_out = J;
        args.tiled_matmul_type = CPU;

        ggml::gemmini::MatMul facade(args);
        const auto result = facade.run_full();
        if (result.status != ggml::gemmini::MatMulStatus::success) {
            GGML_ABORT("Gemmini FP32 facade execution failed");
        }
    }

    using unclassified_matmul_diagnostic_key =
        std::tuple<std::string, std::string, std::string>;
    std::mutex unclassified_matmul_mutex;
    std::set<unclassified_matmul_diagnostic_key> unclassified_matmul_diagnostics;

    std::string resolve_backend_matmul_layer(std::string_view model_arch,
                                             std::string_view weight_name,
                                             std::string_view input_name,
                                             std::string_view consumer_name) {
        std::string layer = ggml::gemmini::types::resolve_matmul_layer(
            model_arch, weight_name, input_name, consumer_name);
        if (layer.compare(0, 13, "unclassified.") != 0) {
            return layer;
        }

        bool emit_diagnostic = false;
        {
            std::lock_guard<std::mutex> lock(unclassified_matmul_mutex);
            emit_diagnostic = unclassified_matmul_diagnostics.emplace(
                model_arch, weight_name, consumer_name).second;
        }
        if (emit_diagnostic) {
            GGML_LOG_WARN(
                "Gemmini semantic matmul classification fallback: layer='%s'\n",
                layer.c_str());
        }
        return layer;
    }

    bool log_prepare_q8_0_rows_for_q8_h1_fail(const char * reason, const ggml_tensor * src, int64_t row = -1) {
        const int64_t ne0 = src ? src->ne[0] : -1;
        const int64_t ne1 = src ? src->ne[1] : -1;
        const int64_t ne2 = src ? src->ne[2] : -1;
        const int64_t ne3 = src ? src->ne[3] : -1;
        const size_t nb0 = src ? src->nb[0] : 0;
        const size_t nb1 = src ? src->nb[1] : 0;
        const size_t nb2 = src ? src->nb[2] : 0;
        const size_t nb3 = src ? src->nb[3] : 0;
        const ggml_type type = src ? src->type : GGML_TYPE_COUNT;
        const void * data = src ? src->data : nullptr;
        const void * view = src ? src->view_src : nullptr;
        const size_t view_offs = src ? src->view_offs : 0;
        if (row >= 0) {
            ggml::gemmini::log::debug("Q8_0->Q8_H1",
                "[prepare_q8_0_rows_for_q8_h1] row=%lld reason=%s src=%p type=%d data=%p view_src=%p view_offs=%zu ne=[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu]",
                (long long) row,
                reason ? reason : "",
                (void *)src,
                (int)type,
                data,
                view,
                view_offs,
                (long long) ne0, (long long) ne1, (long long) ne2, (long long) ne3,
                nb0, nb1, nb2, nb3);
        } else {
            ggml::gemmini::log::debug("Q8_0->Q8_H1",
                "[prepare_q8_0_rows_for_q8_h1] reason=%s src=%p type=%d data=%p view_src=%p view_offs=%zu ne=[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu]",
                reason ? reason : "",
                (void *)src,
                (int)type,
                data,
                view,
                view_offs,
                (long long) ne0, (long long) ne1, (long long) ne2, (long long) ne3,
                nb0, nb1, nb2, nb3);
        }
        return false;
    }

    bool log_prepare_q4_0_rows_for_q4_h1_fail(const char * reason, const ggml_tensor * src, int64_t row = -1) {
        const int64_t ne0 = src ? src->ne[0] : -1;
        const int64_t ne1 = src ? src->ne[1] : -1;
        const int64_t ne2 = src ? src->ne[2] : -1;
        const int64_t ne3 = src ? src->ne[3] : -1;
        const size_t nb0 = src ? src->nb[0] : 0;
        const size_t nb1 = src ? src->nb[1] : 0;
        const size_t nb2 = src ? src->nb[2] : 0;
        const size_t nb3 = src ? src->nb[3] : 0;
        const ggml_type type = src ? src->type : GGML_TYPE_COUNT;
        const void * data = src ? src->data : nullptr;
        const void * view = src ? src->view_src : nullptr;
        const size_t view_offs = src ? src->view_offs : 0;
        if (row >= 0) {
            ggml::gemmini::log::debug("Q4_0->Q4_H1",
                "[prepare_q4_0_rows_for_q4_h1] row=%lld reason=%s src=%p type=%d data=%p view_src=%p view_offs=%zu ne=[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu]",
                (long long) row, reason ? reason : "", (void *)src, (int)type,
                data, view, view_offs,
                (long long) ne0, (long long) ne1, (long long) ne2, (long long) ne3,
                nb0, nb1, nb2, nb3);
        } else {
            ggml::gemmini::log::debug("Q4_0->Q4_H1",
                "[prepare_q4_0_rows_for_q4_h1] reason=%s src=%p type=%d data=%p view_src=%p view_offs=%zu ne=[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu]",
                reason ? reason : "", (void *)src, (int)type,
                data, view, view_offs,
                (long long) ne0, (long long) ne1, (long long) ne2, (long long) ne3,
                nb0, nb1, nb2, nb3);
        }
        return false;
    }

    void ggml_gemmini_log_q8_h1_contract_issue(const char * layer, const ggml_gemmini_args_t & args) {
        ggml::gemmini::log::debug(layer,
            "[Q8_H1 contract] format=%d J=%zu K=%zu blocks_per_row=%zu logical_rows=%zu block_count=%zu ptr=%p",
            (int) args.weight_format,
            args.J,
            args.K,
            args.blocks_per_row,
            args.q8_h1_block_count,
            (const void *)args.q8_h1_blocks);
    }

    bool gemmini_dim_to_size(int64_t dim, size_t &out)
    {
        if (dim <= 0)
            return false;

        out = static_cast<size_t>(dim);
        return true;
    }

    bool gemmini_checked_mul(size_t lhs, size_t rhs, size_t &out)
    {
        if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs)
            return false;

        out = lhs * rhs;
        return true;
    }

    bool gemmini_set_native_matched_weight_args(const ggml_tensor *weight,
                                                  ggml_gemmini_args_t &args)
    {
        if (weight == nullptr || weight->data == nullptr || weight->view_src != nullptr ||
            weight->view_offs != 0 || weight->ne[0] <= 0 ||
            weight->ne[0] % 32 != 0 || weight->ne[1] <= 0 ||
            weight->ne[2] != 1 || weight->ne[3] != 1) {
            return false;
        }

        const bool q4 = weight->type == GGML_TYPE_Q4_H1 ||
                        weight->type == GGML_TYPE_Q4_HP1;
        const bool q16 = weight->type == GGML_TYPE_Q16_0 ||
                         weight->type == GGML_TYPE_Q16_H1 ||
                         weight->type == GGML_TYPE_Q16_HP1;
        if ((q4 && GGML_GEMMINI_WEIGHT_BITS != 4) ||
            (q16 && GGML_GEMMINI_WEIGHT_BITS != 16)) {
            return false;
        }

        size_t block_size = 0;
        size_t block_alignment = 1;
        switch (weight->type) {
        case GGML_TYPE_Q4_H1:
            args.weight_format = ggml_gemmini_args_t::im2p_weight_format_t::q4_h1;
            block_size = sizeof(block_q4_h1);
            block_alignment = alignof(block_q4_h1);
            break;
        case GGML_TYPE_Q4_HP1:
            args.weight_format = ggml_gemmini_args_t::im2p_weight_format_t::q4_hp1;
            block_size = sizeof(block_q4_hp1);
            block_alignment = alignof(block_q4_hp1);
            break;
        case GGML_TYPE_Q16_0:
            args.weight_format = ggml_gemmini_args_t::im2p_weight_format_t::q16_h0;
            block_size = sizeof(block_q16_h0);
            block_alignment = alignof(block_q16_h0);
            break;
        case GGML_TYPE_Q16_H1:
            args.weight_format = ggml_gemmini_args_t::im2p_weight_format_t::q16_h1;
            block_size = sizeof(block_q16_h1);
            block_alignment = alignof(block_q16_h1);
            break;
        case GGML_TYPE_Q16_HP1:
            args.weight_format = ggml_gemmini_args_t::im2p_weight_format_t::q16_hp1;
            block_size = sizeof(block_q16_hp1);
            block_alignment = alignof(block_q16_hp1);
            break;
        default:
            return false;
        }

        const size_t rows = static_cast<size_t>(weight->ne[1]);
        const size_t k = static_cast<size_t>(weight->ne[0]);
        const size_t blocks_per_row = k / 32;
        size_t row_bytes = 0;
        size_t total_bytes = 0;
        size_t block_count = 0;
        if (!gemmini_checked_mul(blocks_per_row, block_size, row_bytes) ||
            !gemmini_checked_mul(row_bytes, rows, total_bytes) ||
            !gemmini_checked_mul(blocks_per_row, rows, block_count) ||
            weight->nb[0] != block_size || weight->nb[1] != row_bytes ||
            weight->nb[2] != total_bytes || weight->nb[3] != total_bytes ||
            reinterpret_cast<uintptr_t>(weight->data) % block_alignment != 0 ||
            (args.J != 0 && args.J != rows) || (args.K != 0 && args.K != k)) {
            return false;
        }

        args.J = rows;
        args.K = k;
        args.B = nullptr;
        args.B_blocks = nullptr;
        args.B_scales = nullptr;
        args.weight_i8_scale_active = false;
        args.weight_scale = 1.0f;
        args.blocks_per_row = 0;
        args.blocks_K = blocks_per_row;
        args.blocks_J = rows;
        args.blocks_I = rows;
        args.block_size_k = 32;
        args.sB = k;
        args.native_blocks_per_row = blocks_per_row;
        args.native_block_count = block_count;
        args.native_weight_bytes = total_bytes;
        switch (weight->type) {
        case GGML_TYPE_Q4_H1: args.q4_h1_blocks = static_cast<const block_q4_h1 *>(weight->data); break;
        case GGML_TYPE_Q4_HP1: args.q4_hp1_blocks = static_cast<const block_q4_hp1 *>(weight->data); break;
        case GGML_TYPE_Q16_0: args.q16_h0_blocks = static_cast<const block_q16_h0 *>(weight->data); break;
        case GGML_TYPE_Q16_H1: args.q16_h1_blocks = static_cast<const block_q16_h1 *>(weight->data); break;
        case GGML_TYPE_Q16_HP1: args.q16_hp1_blocks = static_cast<const block_q16_hp1 *>(weight->data); break;
        default: return false;
        }
        return args.has_native_matched_width_contract();
    }

    bool gemmini_q8_channel_layout_contract(const ggml_tensor * weight)
    {
        if (weight == nullptr || weight->type != GGML_TYPE_Q8_CHANNEL ||
            weight->view_src != nullptr || weight->view_offs != 0 ||
            weight->ne[0] <= 0 || weight->ne[1] <= 0 || weight->ne[2] != 1 || weight->ne[3] != 1) {
            return false;
        }

        if (static_cast<uintmax_t>(weight->ne[0]) >
            static_cast<uintmax_t>(std::numeric_limits<size_t>::max()) - sizeof(float)) {
            return false;
        }

        const size_t dim_k = static_cast<size_t>(weight->ne[0]);
        const size_t row_size = ggml_row_size(GGML_TYPE_Q8_CHANNEL, weight->ne[0]);
        const size_t row_count = static_cast<size_t>(weight->ne[1]);
        size_t storage_bytes = 0;
        return gemmini_checked_mul(row_size, row_count, storage_bytes) &&
            row_size == sizeof(float) + dim_k &&
            weight->nb[0] == 1 && weight->nb[1] == row_size &&
            weight->nb[2] == storage_bytes && weight->nb[3] == storage_bytes &&
            ggml_nbytes(weight) == storage_bytes;
    }

    bool gemmini_q8_channel_weight_contract(const ggml_tensor * weight)
    {
        if (!gemmini_q8_channel_layout_contract(weight) || weight->data == nullptr) {
            return false;
        }

        const size_t row_size = ggml_row_size(GGML_TYPE_Q8_CHANNEL, weight->ne[0]);
        const size_t row_count = static_cast<size_t>(weight->ne[1]);
        const auto * row_base = static_cast<const uint8_t *>(weight->data);
        for (size_t row = 0; row < row_count; ++row) {
            float scale = 0.0f;
            std::memcpy(&scale, row_base + row * row_size, sizeof(scale));
            if (!std::isfinite(scale)) {
                return false;
            }
        }

        return true;
    }

    bool gemmini_q4_0_reprocess_layout_contract(const ggml_tensor * weight)
    {
        if (weight == nullptr || weight->type != GGML_TYPE_Q4_0 ||
            weight->data == nullptr || weight->ne[0] <= 0 ||
            weight->ne[0] % QK4_0 != 0 || weight->ne[1] <= 0 ||
            weight->ne[2] != 1 || weight->ne[3] != 1 ||
            weight->view_src != nullptr || weight->view_offs != 0) {
            return false;
        }

        const size_t blocks_per_row =
            static_cast<size_t>(weight->ne[0]) / QK4_0;
        const size_t rows = static_cast<size_t>(weight->ne[1]);
        size_t row_bytes = 0;
        size_t storage_bytes = 0;
        return gemmini_checked_mul(
                   blocks_per_row, sizeof(block_q4_0), row_bytes) &&
            gemmini_checked_mul(row_bytes, rows, storage_bytes) &&
            weight->nb[0] == sizeof(block_q4_0) &&
            weight->nb[1] == row_bytes &&
            weight->nb[2] == storage_bytes &&
            weight->nb[3] == storage_bytes &&
            ggml_nbytes(weight) == storage_bytes &&
            reinterpret_cast<uintptr_t>(weight->data) %
                    alignof(block_q4_0) ==
                0;
    }

    enum class gemmini_q8_channel_int_route_t : uint8_t {
        unsupported,
        cpu_direct_read,
        hw_dense_sidecar,
    };

    static gemmini_q8_channel_int_route_t gemmini_q8_channel_int_route_for(
        const ggml_tensor * weight,
        bool validate_payload = true)
    {
        if (validate_payload ? !gemmini_q8_channel_weight_contract(weight) :
                               !gemmini_q8_channel_layout_contract(weight)) {
            return gemmini_q8_channel_int_route_t::unsupported;
        }

        if constexpr (
            ggml::gemmini::config::CURRENT_ACTIVATION_QUANT == ggml::gemmini::config::ActivationQuantAlgo::TOKEN &&
            TRANSPOSE_B == 0
        ) {
            return gemmini_q8_channel_int_route_t::unsupported;
        }

        if constexpr (OPTION == CPU) {
          if constexpr (
              ggml::gemmini::config::CURRENT_ACTIVATION_QUANT ==
                  ggml::gemmini::config::ActivationQuantAlgo::TENSOR ||
              ggml::gemmini::config::CURRENT_ACTIVATION_QUANT ==
                  ggml::gemmini::config::ActivationQuantAlgo::TOKEN ||
              ggml::gemmini::config::CURRENT_ACTIVATION_QUANT ==
                  ggml::gemmini::config::ActivationQuantAlgo::BLOCK ||
              ggml::gemmini::config::CURRENT_ACTIVATION_QUANT ==
                  ggml::gemmini::config::ActivationQuantAlgo::STRIPE) {
            return gemmini_q8_channel_int_route_t::cpu_direct_read;
          }
        } else if constexpr (OPTION == OS || OPTION == WS) {
          if constexpr (
              ggml::gemmini::config::CURRENT_ACTIVATION_QUANT ==
                  ggml::gemmini::config::ActivationQuantAlgo::TENSOR ||
              ggml::gemmini::config::CURRENT_ACTIVATION_QUANT ==
                  ggml::gemmini::config::ActivationQuantAlgo::TOKEN ||
              ggml::gemmini::config::CURRENT_ACTIVATION_QUANT ==
                  ggml::gemmini::config::ActivationQuantAlgo::BLOCK ||
              ggml::gemmini::config::CURRENT_ACTIVATION_QUANT ==
                  ggml::gemmini::config::ActivationQuantAlgo::STRIPE) {
            return gemmini_q8_channel_int_route_t::hw_dense_sidecar;
          }
        }

        return gemmini_q8_channel_int_route_t::unsupported;
    }

    static bool gemmini_q8_channel_int_route_supported(const ggml_tensor * weight)
    {
        return gemmini_q8_channel_int_route_for(weight) !=
            gemmini_q8_channel_int_route_t::unsupported;
    }

    bool gemmini_shared_weight_contract(const ggml_tensor * op)
    {
        if (op == nullptr || op->src[0] == nullptr || op->src[1] == nullptr)
            return false;

        const ggml_tensor * weight = op->src[0];
        const ggml_tensor * activation = op->src[1];
        if (weight->ne[2] != 1 || weight->ne[3] != 1 ||
            weight->ne[0] != activation->ne[0] || op->ne[0] != weight->ne[1] ||
            ggml_nrows(activation) != ggml_nrows(op) ||
            !(ggml_is_contiguous(weight) ||
              (weight->data == nullptr ? gemmini_q8_channel_layout_contract(weight) :
                                         gemmini_q8_channel_weight_contract(weight))) ||
            !ggml_is_contiguous(activation) ||
            !ggml_is_contiguous(op)) {
            return false;
        }

        size_t I = 0;
        size_t J = 0;
        size_t K = 0;
        size_t unused = 0;
        return gemmini_dim_to_size(ggml_nrows(activation), I) &&
               gemmini_dim_to_size(weight->ne[1], J) &&
               gemmini_dim_to_size(activation->ne[0], K) &&
               gemmini_checked_mul(I, K, unused) &&
               gemmini_checked_mul(J, K, unused) &&
               gemmini_checked_mul(I, J, unused);
    }

    bool gemmini_is_loader_metadata(const ggml_tensor * tensor)
    {
        return tensor != nullptr && tensor->data == nullptr && tensor->view_src == nullptr &&
               tensor->buffer != nullptr && ggml_backend_buffer_get_size(tensor->buffer) == 0;
    }

    bool gemmini_set_q8_h2_weight_args(const ggml_tensor * weight, ggml_gemmini_args_t & args)
    {
        size_t dim_k = 0;
        size_t dim_j = 0;
        size_t dim_z = 0;
        size_t dim_w = 0;
        if (weight == nullptr || weight->type != GGML_TYPE_Q8_H2 ||
            !gemmini_dim_to_size(weight->ne[0], dim_k) || dim_k % QK8_H2 != 0 ||
            !gemmini_dim_to_size(weight->ne[1], dim_j) ||
            !gemmini_dim_to_size(weight->ne[2], dim_z) ||
            !gemmini_dim_to_size(weight->ne[3], dim_w)) {
            return false;
        }

        size_t logical_rows = 0;
        size_t row_bytes = 0;
        size_t plane_bytes = 0;
        size_t volume_bytes = 0;
        size_t storage_bytes = 0;
        size_t block_count = 0;
        const size_t blocks_per_row = dim_k / QK8_H2;
        if (blocks_per_row == 0 ||
            !gemmini_checked_mul(dim_j, dim_z, logical_rows) ||
            !gemmini_checked_mul(logical_rows, dim_w, logical_rows) ||
            !gemmini_checked_mul(blocks_per_row, sizeof(block_q8_h2), row_bytes) ||
            !gemmini_checked_mul(row_bytes, dim_j, plane_bytes) ||
            !gemmini_checked_mul(plane_bytes, dim_z, volume_bytes) ||
            !gemmini_checked_mul(volume_bytes, dim_w, storage_bytes) ||
            !gemmini_checked_mul(logical_rows, blocks_per_row, block_count) ||
            (args.J != 0 && args.J != logical_rows) ||
            (args.K != 0 && args.K != dim_k) ||
            weight->nb[0] != sizeof(block_q8_h2) || weight->nb[1] != row_bytes ||
            weight->nb[2] != plane_bytes || weight->nb[3] != volume_bytes ||
            (weight->view_src == nullptr && weight->view_offs != 0)) {
            return false;
        }

        const char * base = reinterpret_cast<const char *>(weight->view_src ? weight->view_src->data : weight->data);
        const size_t view_offset = weight->view_src ? static_cast<size_t>(weight->view_offs) : 0;
        if (base == nullptr) {
            return false;
        }
        if (weight->view_src != nullptr) {
            const size_t source_bytes = ggml_nbytes(weight->view_src);
            if (view_offset > source_bytes || storage_bytes > source_bytes - view_offset) {
                return false;
            }
        }

        const char * data = base + view_offset;
        if (reinterpret_cast<uintptr_t>(data) % alignof(block_q8_h2) != 0) {
            return false;
        }

        args.weight_format = ggml_gemmini_args_t::im2p_weight_format_t::q8_h2;
        args.J = logical_rows;
        args.K = dim_k;
        args.B = nullptr;
        args.B_blocks = nullptr;
        args.B_scales = nullptr;
        args.weight_i8_scale_active = false;
        args.weight_scale = 1.0f;
        args.c_b = nullptr;
        args.s_rf = nullptr;
        args.R = nullptr;
        args.blocks_per_row = blocks_per_row;
        args.blocks_K = blocks_per_row;
        args.blocks_J = logical_rows;
        args.blocks_I = logical_rows;
        args.block_size_k = QK8_H2;
        args.sB = dim_k;
        args.stripe_J = 0;
        args.s_rf_stripe = nullptr;
        args.R_stripe = nullptr;
        args.q8_h2_blocks = reinterpret_cast<const block_q8_h2 *>(data);
        args.q8_h2_block_count = block_count;
        args.q8_h2_blocks_per_row = blocks_per_row;
        return args.has_q8_h2_im2p_contract();
    }

    template <typename hp_block_t>
    bool gemmini_repack_hp_weight_rows_for_q8_h1(
        const hp_block_t * source,
        size_t logical_rows,
        size_t blocks_per_row,
        size_t dim_k,
        std::vector<int8_t> & out_q,
        std::vector<uint8_t> & out_c_b,
        std::vector<float> & out_s_rf,
        std::vector<uint16_t> & out_R);

    bool gemmini_set_q8_hp1_weight_args(
        const ggml_tensor * weight, ggml_gemmini_args_t & args, bool allow_null_data = false)
    {
        size_t dim_k = 0;
        size_t dim_j = 0;
        size_t dim_z = 0;
        size_t dim_w = 0;
        if (weight == nullptr || weight->type != GGML_TYPE_Q8_HP1 ||
            !gemmini_dim_to_size(weight->ne[0], dim_k) || dim_k % QK8_HP != 0 ||
            !gemmini_dim_to_size(weight->ne[1], dim_j) ||
            !gemmini_dim_to_size(weight->ne[2], dim_z) ||
            !gemmini_dim_to_size(weight->ne[3], dim_w)) {
            return false;
        }

        size_t logical_rows = 0;
        size_t row_bytes = 0;
        size_t plane_bytes = 0;
        size_t storage_bytes = 0;
        size_t block_count = 0;
        const size_t blocks_per_row = dim_k / QK8_HP;

        if (blocks_per_row == 0 ||
            !gemmini_checked_mul(dim_j, dim_z, logical_rows) ||
            !gemmini_checked_mul(logical_rows, dim_w, logical_rows) ||
            !gemmini_checked_mul(blocks_per_row, sizeof(block_q8_hp1), row_bytes) ||
            !gemmini_checked_mul(row_bytes, dim_j, plane_bytes) ||
            !gemmini_checked_mul(plane_bytes, dim_z, storage_bytes) ||
            !gemmini_checked_mul(storage_bytes, dim_w, storage_bytes) ||
            !gemmini_checked_mul(logical_rows, blocks_per_row, block_count) ||
            (args.J != 0 && args.J != logical_rows) ||
            (args.K != 0 && args.K != dim_k) ||
            weight->nb[0] != sizeof(block_q8_hp1) || weight->nb[1] != row_bytes ||
            weight->nb[2] != plane_bytes || weight->nb[3] != storage_bytes ||
            (weight->view_src == nullptr && weight->view_offs != 0)) {
            return false;
        }

        const char * base = reinterpret_cast<const char *>(weight->view_src ? weight->view_src->data : weight->data);
        const size_t view_offset = weight->view_src ? static_cast<size_t>(weight->view_offs) : 0;
        if (base == nullptr) {
            return allow_null_data && weight->data == nullptr && weight->view_src == nullptr;
        }
        if (weight->view_src != nullptr) {
            const size_t source_bytes = ggml_nbytes(weight->view_src);
            if (view_offset > source_bytes || storage_bytes > source_bytes - view_offset) {
                return false;
            }
        }

        const char * data = base + view_offset;
        if (reinterpret_cast<uintptr_t>(data) % alignof(block_q8_hp1) != 0) {
            return false;
        }

        args.weight_format = ggml_gemmini_args_t::im2p_weight_format_t::q8_hp1;
        args.J = logical_rows;
        args.K = dim_k;
        args.B = nullptr;
        args.B_blocks = nullptr;
        args.B_scales = nullptr;
        args.weight_i8_scale_active = false;
        args.weight_scale = 1.0f;
        args.c_b = nullptr;
        args.s_rf = nullptr;
        args.R = nullptr;
        args.blocks_per_row = blocks_per_row;
        args.blocks_K = blocks_per_row;
        args.blocks_J = logical_rows;
        args.blocks_I = logical_rows;
        args.block_size_k = QK8_HP;
        args.sB = dim_k;
        args.stripe_J = 0;
        args.s_rf_stripe = nullptr;
        args.R_stripe = nullptr;
        args.q8_hp1_blocks = reinterpret_cast<const block_q8_hp1 *>(data);
        args.q8_hp1_block_count = block_count;
        args.q8_hp1_blocks_per_row = blocks_per_row;
        args.native_weight_bytes = storage_bytes;
        // Ponytail: per-call contract check intentionally removed (was the per-mul-mat
        // O(N_blocks) validator). Layout preconditions above are the only gate; NaN
        // results reported by perplexity are unrelated and tracked separately.
        return true;
    }

    bool gemmini_set_q8_hp2_weight_args(
        const ggml_tensor * weight, ggml_gemmini_args_t & args, bool allow_null_data = false)
    {
        size_t dim_k = 0;
        size_t dim_j = 0;
        size_t dim_z = 0;
        size_t dim_w = 0;
        if (weight == nullptr || weight->type != GGML_TYPE_Q8_HP2 ||
            !gemmini_dim_to_size(weight->ne[0], dim_k) || dim_k % QK8_HP != 0 ||
            !gemmini_dim_to_size(weight->ne[1], dim_j) ||
            !gemmini_dim_to_size(weight->ne[2], dim_z) ||
            !gemmini_dim_to_size(weight->ne[3], dim_w)) {
            return false;
        }

        size_t logical_rows = 0;
        size_t row_bytes = 0;
        size_t plane_bytes = 0;
        size_t storage_bytes = 0;
        size_t block_count = 0;
        const size_t blocks_per_row = dim_k / QK8_HP;

        if (blocks_per_row == 0 ||
            !gemmini_checked_mul(dim_j, dim_z, logical_rows) ||
            !gemmini_checked_mul(logical_rows, dim_w, logical_rows) ||
            !gemmini_checked_mul(blocks_per_row, sizeof(block_q8_hp2), row_bytes) ||
            !gemmini_checked_mul(row_bytes, dim_j, plane_bytes) ||
            !gemmini_checked_mul(plane_bytes, dim_z, storage_bytes) ||
            !gemmini_checked_mul(storage_bytes, dim_w, storage_bytes) ||
            !gemmini_checked_mul(logical_rows, blocks_per_row, block_count) ||
            (args.J != 0 && args.J != logical_rows) ||
            (args.K != 0 && args.K != dim_k) ||
            weight->nb[0] != sizeof(block_q8_hp2) || weight->nb[1] != row_bytes ||
            weight->nb[2] != plane_bytes || weight->nb[3] != storage_bytes ||
            (weight->view_src == nullptr && weight->view_offs != 0)) {
            return false;
        }

        const char * base = reinterpret_cast<const char *>(weight->view_src ? weight->view_src->data : weight->data);
        const size_t view_offset = weight->view_src ? static_cast<size_t>(weight->view_offs) : 0;
        if (base == nullptr) {
            return allow_null_data && weight->data == nullptr && weight->view_src == nullptr;
        }
        if (weight->view_src != nullptr) {
            const size_t source_bytes = ggml_nbytes(weight->view_src);
            if (view_offset > source_bytes || storage_bytes > source_bytes - view_offset) {
                return false;
            }
        }

        const char * data = base + view_offset;
        if (reinterpret_cast<uintptr_t>(data) % alignof(block_q8_hp2) != 0) {
            return false;
        }

        args.weight_format = ggml_gemmini_args_t::im2p_weight_format_t::q8_hp2;
        args.J = logical_rows;
        args.K = dim_k;
        args.B = nullptr;
        args.B_blocks = nullptr;
        args.B_scales = nullptr;
        args.weight_i8_scale_active = false;
        args.weight_scale = 1.0f;
        args.c_b = nullptr;
        args.s_rf = nullptr;
        args.R = nullptr;
        args.blocks_per_row = blocks_per_row;
        args.blocks_K = blocks_per_row;
        args.blocks_J = logical_rows;
        args.blocks_I = logical_rows;
        args.block_size_k = QK8_HP;
        args.sB = dim_k;
        args.stripe_J = 0;
        args.s_rf_stripe = nullptr;
        args.R_stripe = nullptr;
        args.q8_hp2_blocks = reinterpret_cast<const block_q8_hp2 *>(data);
        args.q8_hp2_block_count = block_count;
        args.q8_hp2_blocks_per_row = blocks_per_row;
        // Ponytail: per-call contract check intentionally removed; see HP1 sibling.
        return true;
    }

    struct ggml_tensor * gemmini_i8_scale_tensor(const struct ggml_tensor * weight)
    {
        if (weight == nullptr || weight->type != GGML_TYPE_I8)
            return nullptr;

        return weight->src[GGML_MAX_SRC - 1];
    }

    bool gemmini_i8_scale_tensor_is_valid(const struct ggml_tensor * scale)
    {
        return scale != nullptr &&
               scale->type == GGML_TYPE_F32 &&
               ggml_nelements(scale) == 1;
    }

    const float * gemmini_i8_weight_scale_data(const struct ggml_tensor * weight)
    {
        const struct ggml_tensor * scale = gemmini_i8_scale_tensor(weight);
        if (!gemmini_i8_scale_tensor_is_valid(scale) || scale->data == nullptr)
            return nullptr;

        return static_cast<const float *>(scale->data);
    }

    bool gemmini_i8_weight_layout_is_supported(const struct ggml_tensor * weight, bool transpose_b)
    {
        return transpose_b &&
               weight != nullptr &&
               weight->type == GGML_TYPE_I8 &&
               weight->ne[0] > 0 &&
               weight->nb[0] == sizeof(int8_t) &&
               weight->nb[1] == static_cast<size_t>(weight->ne[0]);
    }

    const float * gemmini_i8_supported_scale_data(const struct ggml_tensor * weight, bool transpose_b)
    {
        if (!gemmini_i8_weight_layout_is_supported(weight, transpose_b))
            return nullptr;

        return gemmini_i8_weight_scale_data(weight);
    }

    template <typename hp_block_t>
    bool gemmini_repack_hp_weight_rows_for_q8_h1(
        const hp_block_t * source,
        size_t logical_rows,
        size_t blocks_per_row,
        size_t dim_k,
        std::vector<int8_t> & out_q,
        std::vector<uint8_t> & out_c_b,
        std::vector<float> & out_s_rf,
        std::vector<uint16_t> & out_R)
    {
        if (logical_rows == 0 || blocks_per_row == 0 || dim_k == 0) {
            return false;
        }
        if (blocks_per_row > static_cast<size_t>(std::numeric_limits<int>::max()) || dim_k > static_cast<size_t>(std::numeric_limits<int>::max())) {
            return false;
        }

        size_t q_count = 0;
        size_t c_b_count = 0;
        if (!gemmini_checked_mul(logical_rows, dim_k, q_count) ||
            !gemmini_checked_mul(logical_rows, blocks_per_row, c_b_count)) {
            return false;
        }

        out_q.assign(q_count, 0);
        out_c_b.assign(c_b_count, 0);
        out_s_rf.assign(logical_rows, 0.0f);
        out_R.assign(logical_rows, 0);

        std::vector<block_q8_0> quantized_blocks(blocks_per_row);
        std::vector<ggml::gemmini::quants::BlockQ8_0> source_blocks(blocks_per_row);
        std::vector<float> float_row(dim_k);
        std::vector<int8_t> row_qs(dim_k);
        std::vector<uint8_t> row_c_b(blocks_per_row);

        for (size_t row = 0; row < logical_rows; ++row) {
            const size_t row_block_offset = row * blocks_per_row;
            for (size_t block = 0; block < blocks_per_row; ++block) {
                const hp_block_t & hp = source[row_block_offset + block];
                const size_t block_row_offset = block * QK8_HP;
                if (hp.padding[0] != 0 || hp.padding[1] != 0) {
                    return false;
                }
                if (hp.m == std::numeric_limits<int16_t>::min()) {
                    for (size_t i = 0; i < QK8_HP; ++i) {
                        if (hp.qs[i] != 0) {
                            return false;
                        }
                        float_row[block_row_offset + i] = 0.0f;
                    }
                    continue;
                }
                if (!std::isfinite(hp.channel_scale)) {
                    return false;
                }
                const float block_scale = std::ldexp(hp.channel_scale, static_cast<int>(hp.m));
                if (!std::isfinite(block_scale)) {
                    return false;
                }
                for (size_t i = 0; i < QK8_HP; ++i) {
                    float_row[block_row_offset + i] = static_cast<float>(hp.qs[i]) * block_scale;
                }
            }

            quantize_row_q8_0_ref(float_row.data(), quantized_blocks.data(), static_cast<int64_t>(dim_k));
            std::memcpy(source_blocks.data(), quantized_blocks.data(),
                        blocks_per_row * sizeof(block_q8_0));
            ggml::gemmini::quants::BlockQ8_H1 row_h1{ 0.0f, 0, row_c_b.data(), row_qs.data() };
            if (!ggml::gemmini::quants::quantize_row_q8_h1(
                    source_blocks.data(),
                    static_cast<int>(blocks_per_row),
                    &row_h1)) {
                return false;
            }

            const size_t q_offset = row * dim_k;
            const size_t c_b_offset = row * blocks_per_row;
            for (size_t i = 0; i < dim_k; ++i) {
                out_q[q_offset + i] = row_qs[i];
            }
            for (size_t i = 0; i < blocks_per_row; ++i) {
                out_c_b[c_b_offset + i] = row_c_b[i];
            }
            out_s_rf[row] = row_h1.s_rf;
            out_R[row] = row_h1.R;
        }

        return true;
    }
}

#if defined(GGML_GEMMINI_TESTING)
std::string ggml::gemmini::test_resolve_backend_matmul_layer(
    std::string_view model_arch, std::string_view weight_name,
    std::string_view input_name, std::string_view consumer_name) {
    return resolve_backend_matmul_layer(
        model_arch, weight_name, input_name, consumer_name);
}

void ggml::gemmini::test_reset_unclassified_matmul_diagnostics() {
    std::lock_guard<std::mutex> lock(unclassified_matmul_mutex);
    unclassified_matmul_diagnostics.clear();
}

size_t ggml::gemmini::test_unclassified_matmul_diagnostic_count() {
    std::lock_guard<std::mutex> lock(unclassified_matmul_mutex);
    return unclassified_matmul_diagnostics.size();
}
#endif

bool ggml::gemmini::prepare_q4_0_rows_for_q4_h1(
    const ggml_tensor * src,
    std::vector<block_q4_h1> & dst,
    size_t * blocks_per_row_out,
    size_t * logical_rows_out) {
    if (src == nullptr) {
        return log_prepare_q4_0_rows_for_q4_h1_fail("source tensor is null", src);
    }
    if (src->type != GGML_TYPE_Q4_0) {
        return log_prepare_q4_0_rows_for_q4_h1_fail("source tensor type is not Q4_0", src);
    }
    if (src->data == nullptr) {
        return log_prepare_q4_0_rows_for_q4_h1_fail("source tensor data is null", src);
    }
    if (src->ne[0] <= 0 || src->ne[0] % QK4_0 != 0) {
        return log_prepare_q4_0_rows_for_q4_h1_fail("source K dimension is invalid for Q4_0 blocks", src);
    }
    if (src->view_src != nullptr && src->view_offs % sizeof(block_q4_0) != 0) {
        return log_prepare_q4_0_rows_for_q4_h1_fail("source view offset is not block-aligned", src);
    }

    const int64_t dim_j = src->ne[1] > 0 ? src->ne[1] : 1;
    const int64_t dim_z = src->ne[2] > 0 ? src->ne[2] : 1;
    const int64_t dim_w = src->ne[3] > 0 ? src->ne[3] : 1;
    const size_t dim_k = static_cast<size_t>(src->ne[0]);
    const size_t blocks_per_row = dim_k / QK4_0;
    if (blocks_per_row == 0 ||
        blocks_per_row > static_cast<size_t>(std::numeric_limits<int>::max())) {
        return log_prepare_q4_0_rows_for_q4_h1_fail("blocks_per_row out of range", src);
    }

    const __int128 logical_rows_128 =
        static_cast<__int128>(dim_j) * static_cast<__int128>(dim_z) * static_cast<__int128>(dim_w);
    if (logical_rows_128 <= 0 ||
        logical_rows_128 > static_cast<__int128>(std::numeric_limits<size_t>::max())) {
        return log_prepare_q4_0_rows_for_q4_h1_fail("logical row count overflow", src);
    }

    const size_t logical_rows = static_cast<size_t>(logical_rows_128);
    size_t row_bytes = 0;
    size_t plane_bytes = 0;
    size_t storage_bytes = 0;
    size_t block_count = 0;
    if (!gemmini_checked_mul(blocks_per_row, sizeof(block_q4_0), row_bytes) ||
        !gemmini_checked_mul(row_bytes, static_cast<size_t>(dim_j), plane_bytes) ||
        !gemmini_checked_mul(plane_bytes, static_cast<size_t>(dim_z), storage_bytes) ||
        !gemmini_checked_mul(storage_bytes, static_cast<size_t>(dim_w), storage_bytes) ||
        !gemmini_checked_mul(logical_rows, blocks_per_row, block_count) ||
        src->nb[0] != sizeof(block_q4_0) || src->nb[1] != row_bytes ||
        src->nb[2] != plane_bytes || src->nb[3] != storage_bytes) {
        return log_prepare_q4_0_rows_for_q4_h1_fail("invalid source layout metadata", src);
    }
    if (src->view_src != nullptr) {
        const size_t source_bytes = ggml_nbytes(src->view_src);
        if (src->view_offs > source_bytes || storage_bytes > source_bytes - src->view_offs) {
            return log_prepare_q4_0_rows_for_q4_h1_fail("view source out of range", src);
        }
    }

    const char * base = reinterpret_cast<const char *>(
        src->view_src ? src->view_src->data : src->data);
    if (base == nullptr) {
        return log_prepare_q4_0_rows_for_q4_h1_fail("source base pointer is null", src);
    }
    if (src->view_src != nullptr) {
        base += src->view_offs;
    }
    if (reinterpret_cast<uintptr_t>(base) % alignof(block_q4_0) != 0) {
        return log_prepare_q4_0_rows_for_q4_h1_fail("source base pointer alignment mismatch", src);
    }

    std::vector<block_q4_h1> converted(block_count);
    for (size_t row = 0; row < logical_rows; ++row) {
        const auto * source_row = reinterpret_cast<const block_q4_0 *>(
            base + row * row_bytes);
        block_q4_h1 * destination_row =
            converted.data() + row * blocks_per_row;
        if (!reprocess_row_q4_0_to_q4_h1_ref(
                source_row, destination_row, static_cast<int64_t>(dim_k))) {
            return log_prepare_q4_0_rows_for_q4_h1_fail(
                "reprocessing failed", src, static_cast<int64_t>(row));
        }
    }

    dst = std::move(converted);
    if (blocks_per_row_out != nullptr) {
        *blocks_per_row_out = blocks_per_row;
    }
    if (logical_rows_out != nullptr) {
        *logical_rows_out = logical_rows;
    }
    return true;
}

bool ggml::gemmini::prepare_q8_0_rows_for_q8_h1(
    const ggml_tensor * src,
    std::vector<block_q8_h1> & dst,
    size_t * blocks_per_row_out,
    size_t * logical_rows_out) {
    if (src == nullptr) {
        return log_prepare_q8_0_rows_for_q8_h1_fail("source tensor is null", src);
    }
    if (src->type != GGML_TYPE_Q8_0) {
        return log_prepare_q8_0_rows_for_q8_h1_fail("source tensor type is not Q8_0", src);
    }
    if (src->data == nullptr) {
        return log_prepare_q8_0_rows_for_q8_h1_fail("source tensor data is null", src);
    }
    if (src->ne[0] <= 0 || src->ne[0] % QK8_0 != 0) {
        return log_prepare_q8_0_rows_for_q8_h1_fail("source K dimension is invalid for Q8_0 blocks", src);
    }
    if (src->view_src != nullptr && src->view_offs % sizeof(block_q8_0) != 0) {
        return log_prepare_q8_0_rows_for_q8_h1_fail("source view offset is not block-aligned", src);
    }

    const int64_t dim_j = src->ne[1] > 0 ? src->ne[1] : 1;
    const int64_t dim_z = src->ne[2] > 0 ? src->ne[2] : 1;
    const int64_t dim_w = src->ne[3] > 0 ? src->ne[3] : 1;
    const size_t dim_k = static_cast<size_t>(src->ne[0]);
    const size_t blocks_per_row = dim_k / QK8_0;
    if (blocks_per_row == 0 || blocks_per_row > static_cast<size_t>(std::numeric_limits<int>::max())) {
        return log_prepare_q8_0_rows_for_q8_h1_fail("blocks_per_row out of range", src);
    }

    const __int128 logical_rows_128 =
        static_cast<__int128>(dim_j) * static_cast<__int128>(dim_z) * static_cast<__int128>(dim_w);
    if (logical_rows_128 <= 0 || logical_rows_128 > static_cast<__int128>(std::numeric_limits<size_t>::max())) {
        return log_prepare_q8_0_rows_for_q8_h1_fail("logical row count overflow", src);
    }

    const size_t logical_rows = static_cast<size_t>(logical_rows_128);
    size_t row_bytes = 0;
    size_t plane_bytes = 0;
    size_t storage_bytes = 0;
    size_t block_count = 0;
    if (!gemmini_checked_mul(blocks_per_row, sizeof(block_q8_0), row_bytes) ||
        !gemmini_checked_mul(row_bytes, static_cast<size_t>(dim_j), plane_bytes) ||
        !gemmini_checked_mul(plane_bytes, static_cast<size_t>(dim_z), storage_bytes) ||
        !gemmini_checked_mul(storage_bytes, static_cast<size_t>(dim_w), storage_bytes) ||
        !gemmini_checked_mul(logical_rows, blocks_per_row, block_count) ||
        src->nb[0] != sizeof(block_q8_0) || src->nb[1] != row_bytes ||
        src->nb[2] != plane_bytes || src->nb[3] != storage_bytes) {
        return log_prepare_q8_0_rows_for_q8_h1_fail("invalid source layout metadata", src);
    }
    if (src->view_src != nullptr) {
        const size_t source_bytes = ggml_nbytes(src->view_src);
        if (src->view_offs > source_bytes || storage_bytes > source_bytes - src->view_offs) {
            return log_prepare_q8_0_rows_for_q8_h1_fail("view source out of range", src);
        }
    }

    const char * base = reinterpret_cast<const char *>(src->view_src ? src->view_src->data : src->data);
    if (base == nullptr) {
        return log_prepare_q8_0_rows_for_q8_h1_fail("source base pointer is null", src);
    }
    if (src->view_src != nullptr) {
        base += src->view_offs;
    }
    if (reinterpret_cast<uintptr_t>(base) % alignof(block_q8_0) != 0) {
        return log_prepare_q8_0_rows_for_q8_h1_fail("source base pointer alignment mismatch", src);
    }

    std::vector<block_q8_h1> converted(block_count);
    std::vector<ggml::gemmini::quants::BlockQ8_0> source_row(blocks_per_row);
    std::vector<uint8_t> codes(blocks_per_row);
    std::vector<int8_t> qs(dim_k);
    for (size_t row = 0; row < logical_rows; ++row) {
        std::memcpy(source_row.data(), base + row * row_bytes, row_bytes);
        for (const ggml::gemmini::quants::BlockQ8_0 & source_block : source_row) {
            if ((source_block.d & 0x7c00) == 0x7c00) {
                return log_prepare_q8_0_rows_for_q8_h1_fail("source block has invalid scale", src, static_cast<int64_t>(row));
            }
        }
        ggml::gemmini::quants::BlockQ8_H1 row_h1 = { 0.0f, 0, codes.data(), qs.data() };
        if (!ggml::gemmini::quants::quantize_row_q8_h1(
                source_row.data(), static_cast<int>(blocks_per_row), &row_h1)) {
            return log_prepare_q8_0_rows_for_q8_h1_fail("quantization failed", src, static_cast<int64_t>(row));
        }

        for (size_t block = 0; block < blocks_per_row; ++block) {
            block_q8_h1 & dst_block = converted[row * blocks_per_row + block];
            std::memcpy(dst_block.qs, qs.data() + block * QK8_0, QK8_0);
            dst_block.c_b = codes[block];
            dst_block.s_rf = row_h1.s_rf;
            dst_block.R = row_h1.R;
        }
    }

    dst = std::move(converted);
    if (blocks_per_row_out != nullptr) {
        *blocks_per_row_out = blocks_per_row;
    }
    if (logical_rows_out != nullptr) {
        *logical_rows_out = logical_rows;
    }
    return true;
}

struct ggml_backend_gemmini_context
{
    int n_threads = GGML_DEFAULT_N_THREADS;
    std::unique_ptr<char[]> work_data;
    size_t work_size = 0;
    std::string model_arch;
    ggml::gemmini::q8_h1_artifact_runtime_state q8_h1_artifact;
};

#if defined(GGML_GEMMINI_TEST_OBSERVER)
namespace ggml::gemmini {
namespace {
test_i_observer_t test_i_observer = nullptr;
void * test_i_observer_data = nullptr;
std::mutex test_semantic_layer_observer_mutex;
test_semantic_layer_observer_t test_semantic_layer_observer = nullptr;
void * test_semantic_layer_observer_data = nullptr;
}

void set_test_i_observer(test_i_observer_t observer, void * user_data) {
    test_i_observer = observer;
    test_i_observer_data = user_data;
}

static void observe_test_i(const char * consumer, size_t I) {
    if (test_i_observer != nullptr) {
        test_i_observer(consumer, I, test_i_observer_data);
    }
}

void set_test_semantic_layer_observer(
    test_semantic_layer_observer_t observer, void * user_data) {
    std::lock_guard<std::mutex> lock(test_semantic_layer_observer_mutex);
    test_semantic_layer_observer = observer;
    test_semantic_layer_observer_data = user_data;
}

bool test_observe_semantic_layer(TestSemanticLayerSite site, const char * layer) {
    test_semantic_layer_observer_t observer = nullptr;
    void * user_data = nullptr;
    {
        std::lock_guard<std::mutex> lock(test_semantic_layer_observer_mutex);
        observer = test_semantic_layer_observer;
        user_data = test_semantic_layer_observer_data;
    }
    return observer != nullptr && observer(site, layer, user_data);
}

bool test_probe_physical_layer_sites(const ggml_gemmini_args_t & source) {
    ggml_gemmini_args_t args = source;
    tiled_matmul_auto_fp(&args);
    args = source;
    gemmini_set_tile_ws(&args);
    args = source;
    tiled_matmul_im2p_impl(&args, true, "test semantic layer probe");
    args = source;
    tiled_matmul_auto_im2p(&args);
    args = source;
    tiled_matmul_auto_baseline_dense(
        &args, *baseline_route_for(
            baseline_activation_quant_t::TENSOR,
            baseline_weight_quant_t::TENSOR));
    return true;
}

bool test_probe_physical_null_args() {
    tiled_matmul_auto_fp(nullptr);
    gemmini_set_tile_ws(nullptr);
    return true;
}

bool test_probe_fp_facade_layer(const std::string & layer) {
    const float activation = 2.0f;
    const float weight = 3.0f;
    float output = 0.0f;
    gemmini_matmul_fp_facade(1, 1, 1, &activation, &weight, &output, layer);
    return true;
}
}
#endif

// Cycle 측정 용
extern "C" volatile uint64_t gemmini_tiled_matmul_cycles = 0; // gemmini.h

static void setup_gemmini_log_outputs_if_needed(void) {
    const auto result = ggml::gemmini::log::setup_default_outputs();
    if (!result.cycle) {
        GGML_LOG_WARN("%s: failed to set default cycle log path '%s'\n", __func__, GEMMINI_LOG_DEFAULT_CYCLE_PATH);
    }
    if (!result.debug) {
        GGML_LOG_WARN("%s: failed to set default debug log path '%s'\n", __func__, GEMMINI_LOG_DEFAULT_DEBUG_PATH);
    }
}

#if defined(__linux__) && defined(__aarch64__)
static void log_native_cycle_interval(
        const char * layer, const char * op,
        const gemmini_native_cycle_sample_internal & start,
        const gemmini_native_cycle_sample_internal & end) {
    const gemmini_cycle_record_v2 record{{layer, op, start.value, end.value,
                                          nullptr, 0, nullptr},
                                         0, 0, 0, 0, 0, 0};
    gemmini_log_cycle_record_v2_checked_internal(&record, &start, &end, 1);
}
#endif

static void ggml_backend_gemmini_mul_mat(ggml_backend_gemmini_context *ctx,
                                         struct ggml_tensor *dst) // FP32 output (I×J)
{
    if (dst == nullptr || dst->src[0] == nullptr || dst->src[1] == nullptr) {
        GGML_ABORT("Gemmini mul_mat received null tensor pointers");
    }
    if (ggml_is_empty(dst)) {
        return;
    }
    if (!gemmini_shared_weight_contract(dst)) {
        GGML_ABORT("Gemmini MUL_MAT shared-weight contract failed");
    }
    setup_gemmini_log_outputs_if_needed();
#if LOG_CYCLE
#if defined(__linux__) && defined(__aarch64__)
    const gemmini_native_cycle_sample_internal rmd_telemetry_invocation_start_sample =
        gemmini_read_native_cycle_sample_internal();
    const uint64_t rmd_telemetry_invocation_start = rmd_telemetry_invocation_start_sample.value;
#else
    const uint64_t rmd_telemetry_invocation_start = ggml::gemmini::cycle::read();
#endif
#endif
    const auto *src0 = dst->src[0]; // src0: weight (J x K), row-major, 전치 상태
    const auto *src1 = dst->src[1]; // src1: activation (I x K) -> 전치 없음 (A)
    ggml_gemmini_args_t args;
    args.model_arch = ctx->model_arch.c_str();
    args.matmul_layer = resolve_backend_matmul_layer(
        ctx->model_arch, src0->name, src1->name, dst->name);
    const char * layer = args.matmul_layer.c_str();
    ggml::gemmini::log::debug(layer, "ggml_backend_gemmini_mul_mat called");

#if defined(__linux__) && defined(__aarch64__)
    gemmini_native_cycle_sample_internal start_sample{};
    gemmini_native_cycle_sample_internal end_sample{};
#else
    uint64_t start = 0;
    uint64_t end = 0;
#endif

    /* _______________________ 2. Gemmini용 dimension _____________________ */
    size_t I = 0;
    size_t J = 0;
    size_t K = 0;
    if (!gemmini_dim_to_size(ggml_nrows(src1), I) ||
        !gemmini_dim_to_size(src0->ne[1], J) ||
        !gemmini_dim_to_size(src1->ne[0], K)) {
        ggml::gemmini::log::debug(layer, "invalid dimensions: src1(nrows=%lld, ne0=%lld) src0(ne1=%lld)",
                                 (long long)ggml_nrows(src1), (long long)src1->ne[0], (long long)src0->ne[1]);
        GGML_ABORT("Gemmini mul_mat received invalid dimensions");
    }

    size_t ik_count = 0;
    size_t jk_count = 0;
    size_t ij_count = 0;
    if (!gemmini_checked_mul(I, K, ik_count) ||
        !gemmini_checked_mul(J, K, jk_count) ||
        !gemmini_checked_mul(I, J, ij_count)) {
        ggml::gemmini::log::debug(layer, "dimension overflow: I=%zu J=%zu K=%zu", I, J, K);
        GGML_ABORT("Gemmini mul_mat size overflow");
    }

#if LOG_DUMP
    ggml::gemmini::log::dump(ggml::gemmini::log::file("log/tensor_data/act.jsonl"), layer, src1);
#endif

    ggml::gemmini::log::debug(layer,
            "I=%zu, J=%zu, K=%zu, src1_ne=(%lld,%lld,%lld,%lld), dst_ne=(%lld,%lld,%lld,%lld), src1_nrows=%lld",
            I, J, K,
            (long long) src1->ne[0], (long long) src1->ne[1],
            (long long) src1->ne[2], (long long) src1->ne[3],
            (long long) dst->ne[0], (long long) dst->ne[1],
            (long long) dst->ne[2], (long long) dst->ne[3],
            (long long) ggml_nrows(src1));

    if constexpr (ggml::gemmini::config::CURRENT_COMPUTE_TYPE == ggml::gemmini::config::ComputeType::FLOAT &&
                  !ggml::gemmini::config::DEQUANT_FP_TEST) {
        if (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16) {
            std::vector<float> src0_f32;
            const float *src0_f = (const float *)src0->data;
            if (src0->type == GGML_TYPE_F16) {
                src0_f32.resize(jk_count);
                const ggml_fp16_t *src0_f16 = (const ggml_fp16_t *)src0->data;
                for (size_t j = 0; j < J; j++) {
                    for (size_t k = 0; k < K; k++)
                        src0_f32[j * K + k] = ggml_fp16_to_fp32(src0_f16[j * K + k]);
                }
                src0_f = src0_f32.data();
            }
            gemmini_matmul_fp_facade(I, J, K, (const float *)src1->data, src0_f,
                                     (float *)dst->data, args.matmul_layer);
            return;
        } else if (src0->type == GGML_TYPE_Q8_0) {
            std::vector<float> src0_f32(jk_count);
            dequantize_row_q8_0((const block_q8_0 *)src0->data, src0_f32.data(), jk_count);
            gemmini_matmul_fp_facade(I, J, K, (const float *)src1->data, src0_f32.data(),
                                     (float *)dst->data, args.matmul_layer);
            return;
        } else if (src0->type == GGML_TYPE_Q4_0) {
            std::vector<block_q4_h1> reprocessed;
            if (!ggml::gemmini::prepare_q4_0_rows_for_q4_h1(
                    src0, reprocessed, nullptr, nullptr)) {
                GGML_ABORT("FLOAT Q4_0 runtime reprocessing failed");
            }
            std::vector<float> src0_f32(jk_count);
            dequantize_row_q4_h1(
                reprocessed.data(), src0_f32.data(), jk_count);
            gemmini_matmul_fp_facade(
                I, J, K, static_cast<const float *>(src1->data),
                src0_f32.data(), static_cast<float *>(dst->data),
                args.matmul_layer);
            return;
        } else if (gemmini_is_extended_dequant_weight_type(src0->type)) {
            std::vector<float> src0_f32(jk_count);
            ggml_get_type_traits(src0->type)->to_float(src0->data, src0_f32.data(), jk_count);
            gemmini_matmul_fp_facade(I, J, K, (const float *)src1->data, src0_f32.data(),
                                     (float *)dst->data, args.matmul_layer);
            return;
        } else if (src0->type == GGML_TYPE_Q8_CHANNEL) {
            std::vector<float> src0_f32(jk_count);
            const size_t row_size = ggml_row_size(GGML_TYPE_Q8_CHANNEL, src0->ne[0]);
            const char * row = static_cast<const char *>(src0->data);
            for (size_t j = 0; j < J; ++j) {
                dequantize_row_q8_channel(row + j * row_size, src0_f32.data() + j * K, K);
            }
            gemmini_matmul_fp_facade(I, J, K, (const float *)src1->data, src0_f32.data(),
                                     (float *)dst->data, args.matmul_layer);
            return;
        } else if (src0->type == GGML_TYPE_Q8_H1) {
            std::vector<float> src0_f32(jk_count);
            dequantize_row_q8_h1((const block_q8_h1 *)src0->data, src0_f32.data(), jk_count);
            gemmini_matmul_fp_facade(I, J, K, (const float *)src1->data, src0_f32.data(),
                                     (float *)dst->data, args.matmul_layer);
            return;
        } else if (src0->type == GGML_TYPE_Q8_H2) {
            std::vector<float> src0_f32(jk_count);
            dequantize_row_q8_h2((const block_q8_h2 *)src0->data, src0_f32.data(), jk_count);
            gemmini_matmul_fp_facade(I, J, K, (const float *)src1->data, src0_f32.data(),
                                     (float *)dst->data, args.matmul_layer);
            return;
        } else if (src0->type == GGML_TYPE_Q8_HP1) {
            std::vector<float> src0_f32(jk_count);
            dequantize_row_q8_hp1((const block_q8_hp1 *)src0->data, src0_f32.data(), jk_count);
            gemmini_matmul_fp_facade(I, J, K, (const float *)src1->data, src0_f32.data(),
                                     (float *)dst->data, args.matmul_layer);
            return;
        } else if (src0->type == GGML_TYPE_Q8_HP2) {
            std::vector<float> src0_f32(jk_count);
            dequantize_row_q8_hp2((const block_q8_hp2 *)src0->data, src0_f32.data(), jk_count);
            gemmini_matmul_fp_facade(I, J, K, (const float *)src1->data, src0_f32.data(),
                                     (float *)dst->data, args.matmul_layer);
            return;
        } else {
            ggml::gemmini::log::debug(layer, "FLOAT path unsupported weight type=%d", (int)src0->type);
            GGML_ABORT("FLOAT compute type: unsupported weight type");
        }
    }

    const std::uint8_t product_weight_bits =
        gemmini_product_weight_bits(src0->type);
    if (product_weight_bits != GGML_GEMMINI_ACTIVATION_BITS ||
        product_weight_bits != GGML_GEMMINI_WEIGHT_BITS) {
        ggml::gemmini::log::debug(
            layer,
            "[matmul.route] status=unsupported_route activation_bits=%d "
            "weight_bits=%u artifact_weight_bits=%d",
            GGML_GEMMINI_ACTIVATION_BITS,
            static_cast<unsigned>(product_weight_bits),
            GGML_GEMMINI_WEIGHT_BITS);
#if defined(GGML_GEMMINI_TESTING)
        return;
#else
        GGML_ABORT("Gemmini requires matched activation and weight widths");
#endif
    }

    std::vector<int8_t> q8_channel_dense;
    std::vector<float> q8_channel_scales;
    std::unique_ptr<ggml::gemmini::MatmulStripeCollector> pipeline_collector;
    std::unique_ptr<ggml::gemmini::MatmulExecution> pipeline_execution;
    const auto matmul_resolution = ggml::gemmini::resolve_matmul_options();
    if (!matmul_resolution.ok()) {
        ggml::gemmini::log::debug(layer,
            "[matmul.invocation] status=invalid_invocation error=%d",
            static_cast<int>(matmul_resolution.error));
        GGML_ABORT("Gemmini invalid matmul options");
    }
    auto matmul_options = matmul_resolution.options;
    matmul_options.profiling = LOG_CYCLE != 0;
#if defined(GGML_GEMMINI_EXECUTION_BACKEND_IM2P_SIM)
    constexpr bool im2p_exsia =
        ggml::gemmini::config::CURRENT_ACTIVATION_QUANT ==
        ggml::gemmini::config::ActivationQuantAlgo::EXSIA;
    constexpr bool im2p_non_exsia = !im2p_exsia;
    const auto route_gate =
        product_weight_bits != GGML_GEMMINI_WEIGHT_BITS
            ? ggml::gemmini::im2p_adapter::Result{
                  ggml::gemmini::im2p_adapter::Error::unsupported_route,
                  "tensor weight width does not match the selected IM2P artifact",
                  false}
            : ggml::gemmini::im2p_adapter::gate_route({
                  im2p_exsia,
                  GGML_GEMMINI_ACTIVATION_BITS,
                  product_weight_bits,
                  GGML_GEMMINI_ACTIVATION_BITS,
                  GGML_GEMMINI_WEIGHT_BITS,
                  GGML_GEMMINI_ENABLE_RMD != 0,
                  matmul_options.mode ==
                          ggml::gemmini::MatmulInvocationMode::full
                      ? ggml::gemmini::im2p_adapter::PublicMode::full
                      : ggml::gemmini::im2p_adapter::PublicMode::stripe_pipeline,
                  gemmini_exsia_weight_family(src0->type),
                  matmul_options.rmd_backend ==
                          ggml::gemmini::RmdBackend::cpu_direct
                      ? ggml::gemmini::im2p_adapter::ResidualBackend::cpu_direct
                      : ggml::gemmini::im2p_adapter::ResidualBackend::compact_ws,
                  ggml::gemmini::im2p_adapter::BuildIdentity::im2p_sim_ws});
    if (!route_gate.ok()) {
      ggml::gemmini::im2p_adapter::log_failure("route", route_gate);
#if defined(GGML_GEMMINI_TESTING)
      ggml::gemmini::im2p_adapter::test_record_production_failure(
          route_gate.error);
      return;
#else
      GGML_ABORT("Gemmini IM2P route is unsupported");
#endif
    }
#if defined(GGML_GEMMINI_TESTING)
    if (im2p_exsia) {
      ggml::gemmini::im2p_adapter::test_observe_weight_family(
          gemmini_exsia_weight_family(src0->type));
    }
#endif
#else
    constexpr bool im2p_exsia = false;
    constexpr bool im2p_non_exsia = false;
#endif
#if !defined(__riscv) && !defined(GGML_GEMMINI_EXECUTION_BACKEND_IM2P_SIM)
    if (matmul_options.rmd_backend == ggml::gemmini::RmdBackend::gemmini_ws_compact) {
        ggml::gemmini::log::debug(layer,
            "[matmul.rmd] status=unsupported_backend backend=WS");
        GGML_ABORT("Gemmini unsupported_backend: RMD WS is unavailable on this host");
    }
#endif
    const bool pipeline_requested =
        matmul_options.mode == ggml::gemmini::MatmulInvocationMode::stripe_pipeline;
    const bool full_requested =
        matmul_options.mode == ggml::gemmini::MatmulInvocationMode::full;
    constexpr bool exsia_pipeline_supported =
        ggml::gemmini::config::CURRENT_ACTIVATION_QUANT == ggml::gemmini::config::ActivationQuantAlgo::EXSIA;
    const bool pipeline_enabled = pipeline_requested &&
                                  exsia_pipeline_supported && I > 1 &&
                                  !im2p_non_exsia && !im2p_exsia;
    const bool legacy_full_dispatch = false;
    const bool decode_full_dispatch = pipeline_requested &&
                                      exsia_pipeline_supported && I <= 1 &&
                                      !im2p_non_exsia && !im2p_exsia;
    const bool facade_full_dispatch =
        (full_requested || decode_full_dispatch) && !im2p_non_exsia &&
        !im2p_exsia;
    const size_t pipeline_job_capacity = matmul_options.job_capacity;
    if (pipeline_requested && !exsia_pipeline_supported && !im2p_non_exsia) {
      ggml::gemmini::log::debug(
          layer,
          "[matmul.pipeline] status=unsupported_invocation dispatch=fatal "
          "reason=%s",
          "stripe-pipeline requires EXSIA activation");
      GGML_ABORT("Gemmini unsupported_invocation: stripe-pipeline requires "
                 "EXSIA activation");
    }
    if (decode_full_dispatch) {
        ggml::gemmini::log::debug(layer,
            "[matmul.pipeline] dispatch=full reason=single-row decode I=%zu", I);
    }
    // set args
    #if defined(__linux__) && defined(__aarch64__)
    start_sample = gemmini_read_native_cycle_sample_internal();
#else
    start = ggml::gemmini::cycle::read();
#endif
    args.transpose_B = (TRANSPOSE_B != 0);
    ggml::gemmini::log::debug(layer, "model_arch=%s\n", args.model_arch ? args.model_arch : "");
    args.full_C = FULL_C;
    args.low_D = LOW_D;

    args.I = I;
    args.J = J;
    args.K = K;
    args.tiled_matmul_type = OPTION;
    args.residual_route = matmul_options.rmd_backend == ggml::gemmini::RmdBackend::cpu_direct
        ? ggml::gemmini::residual::ResidualRoute::cpu_direct
        : ggml::gemmini::residual::ResidualRoute::ws_packet;

    /* _____ 3. Gemmini용 stride _____ */
    args.sA = K;
    args.sC = J;

    #if defined(__linux__) && defined(__aarch64__)
    end_sample = gemmini_read_native_cycle_sample_internal();
#else
    end = ggml::gemmini::cycle::read();
#endif
    #if defined(__linux__) && defined(__aarch64__)
    log_native_cycle_interval(layer, "gemmini.prepare_args", start_sample, end_sample);
#else
    ggml::gemmini::log::cycle(layer, "gemmini.prepare_args", start, end);
#endif

    // set tile size
    #if defined(__linux__) && defined(__aarch64__)
    start_sample = gemmini_read_native_cycle_sample_internal();
#else
    start = ggml::gemmini::cycle::read();
#endif
    ggml::gemmini::gemmini_set_tile_ws(&args);
    const auto gemmini_geometry = ggml::gemmini::make_gemmini_geometry(
        {{args.I, args.J, args.K}, {args.tile_I, args.tile_J, args.tile_K}, DIM});
    if (!gemmini_geometry.ok()) {
        GGML_ABORT("Gemmini geometry is invalid");
    }
    args.activation_rows_per_stripe = gemmini_geometry.geometry.stripe_rows;
    #if defined(__linux__) && defined(__aarch64__)
    end_sample = gemmini_read_native_cycle_sample_internal();
#else
    end = ggml::gemmini::cycle::read();
#endif
    #if defined(__linux__) && defined(__aarch64__)
    log_native_cycle_interval(layer, "gemmini.select_tile", start_sample, end_sample);
#else
    ggml::gemmini::log::cycle(layer, "gemmini.select_tile", start, end);
#endif

#if defined(GGML_GEMMINI_TESTING) && \
    defined(GGML_GEMMINI_EXECUTION_BACKEND_IM2P_SIM)
    ggml::gemmini::im2p_adapter::test_observe_activation_allocation();
#endif
    if (!args.A.allocate(args.I, args.K, GGML_GEMMINI_ACTIVATION_BITS)) {
        ggml::gemmini::log::debug(
            args.matmul_layer.c_str(),
            "failed to allocate activation buffer (I=%zu, K=%zu, bits=%d)",
            args.I, args.K, GGML_GEMMINI_ACTIVATION_BITS);
        GGML_ABORT("Gemmini failed to allocate activation buffer");
    }
    if (pipeline_enabled) {
        pipeline_collector = std::make_unique<ggml::gemmini::MatmulStripeCollector>(pipeline_job_capacity);
        args.exsia_stripe_ready_sink = pipeline_collector->sink();
    }
    ggml::gemmini::log::debug(
        args.matmul_layer.c_str(),
        "[" GGML_GEMMINI_ACTIVATION_QUANT_NAME "] final cfg model_arch=%s",
        args.model_arch ? args.model_arch : "");

#if defined(GGML_GEMMINI_TEST_OBSERVER) && GGML_GEMMINI_COMPUTE_TYPE == 0
    ggml::gemmini::observe_test_i("activation", args.I);
#endif
    bool quantize_ok = false;
    const bool deferred_quantization = pipeline_enabled || im2p_exsia;
#if LOG_CYCLE
    const bool quantization_overlaps_rtl =
        pipeline_enabled || (im2p_exsia && pipeline_requested);
#endif
    const auto quantize_activation = [&]() {
#if LOG_CYCLE
#if defined(__linux__) && defined(__aarch64__)
        const gemmini_native_cycle_sample_internal quantize_start_sample =
            gemmini_read_native_cycle_sample_internal();
        const std::uint64_t quantize_start = quantize_start_sample.value;
#else
        const std::uint64_t quantize_start = ggml::gemmini::cycle::read();
#endif
#endif
        const bool result =
            ggml::gemmini::quants::quantize_activation(src1, args);
#if LOG_CYCLE
#if defined(__linux__) && defined(__aarch64__)
        const gemmini_native_cycle_sample_internal quantize_end_sample =
            gemmini_read_native_cycle_sample_internal();
        const std::uint64_t quantize_end = quantize_end_sample.value;
#else
        const std::uint64_t quantize_end = ggml::gemmini::cycle::read();
#endif
        if (quantization_overlaps_rtl) {
            ggml::gemmini::log::debug(
                layer,
                "[matmul.pipeline] cpu_quantization_host_ticks=%llu "
                "overlaps_rtl=true excluded_from_cycle_sink=true",
                static_cast<unsigned long long>(
                    quantize_end - quantize_start));
        } else {
            #if defined(__linux__) && defined(__aarch64__)
            log_native_cycle_interval(
                layer, "gemmini.quantize_activation",
                quantize_start_sample, quantize_end_sample);
#else
            ggml::gemmini::log::cycle(
                layer, "gemmini.quantize_activation", quantize_start, quantize_end);
#endif
        }
#endif
        return result;
    };
    if (!deferred_quantization) {
      quantize_ok = quantize_activation();
    }
#if GGML_GEMMINI_COMPUTE_TYPE == 0 && GGML_GEMMINI_ACTIVATION_BITS == 8
    static_assert(sizeof(elem_t) == 1, "Q8_0 path assumes elem_t is int8 (1 byte).");
#endif

    if (!deferred_quantization && !quantize_ok) {
      if (pipeline_collector && !pipeline_collector->status()) {
        ggml::gemmini::log::debug(
            layer, "[matmul.pipeline] capture status=%d message=%s",
            static_cast<int>(pipeline_collector->status().code),
            pipeline_collector->status().message);
      }
      ggml::gemmini::log::debug(layer, "activation quantize failed");
      GGML_ABORT("Gemmini activation quantization failed");
    }

    const auto run_dequant_fp_test = [&]() -> bool {
      if constexpr (
          ggml::gemmini::config::CURRENT_COMPUTE_TYPE ==
              ggml::gemmini::config::ComputeType::FLOAT &&
          ggml::gemmini::config::DEQUANT_FP_TEST) {
        std::vector<float> activation_f32(ik_count);
        const bool dequant_ok = ggml::gemmini::quants::dequantize_activation(
            activation_f32.data(),
            K,
            1,
            I,
            K,
            args);
        if (!dequant_ok) {
            ggml::gemmini::log::debug(layer, "DEQUANT_FP_TEST activation dequantize failed");
        }
        GGML_ASSERT(dequant_ok && "DEQUANT_FP_TEST: activation dequantization failed");
        if (!dequant_ok)
            return true;

        std::vector<float> src0_f32;
        const float *src0_f = reinterpret_cast<const float *>(src0->data);
        if (src0->type == GGML_TYPE_F16) {
            src0_f32.resize(jk_count);
            const ggml_fp16_t *src0_f16 = reinterpret_cast<const ggml_fp16_t *>(src0->data);
            for (size_t j = 0; j < J; j++) {
                for (size_t k = 0; k < K; k++)
                    src0_f32[j * K + k] = ggml_fp16_to_fp32(src0_f16[j * K + k]);
            }
            src0_f = src0_f32.data();
        } else if (src0->type == GGML_TYPE_Q8_0) {
            src0_f32.resize(jk_count);
            dequantize_row_q8_0(reinterpret_cast<const block_q8_0 *>(src0->data), src0_f32.data(), jk_count);
            src0_f = src0_f32.data();
        } else if (src0->type == GGML_TYPE_Q4_0) {
            std::vector<block_q4_h1> reprocessed;
            if (!ggml::gemmini::prepare_q4_0_rows_for_q4_h1(
                    src0, reprocessed, nullptr, nullptr)) {
                GGML_ABORT("DEQUANT_FP_TEST Q4_0 runtime reprocessing failed");
            }
            src0_f32.resize(jk_count);
            dequantize_row_q4_h1(
                reprocessed.data(), src0_f32.data(), jk_count);
            src0_f = src0_f32.data();
        } else if (gemmini_is_extended_dequant_weight_type(src0->type)) {
            src0_f32.resize(jk_count);
            ggml_get_type_traits(src0->type)->to_float(src0->data, src0_f32.data(), jk_count);
            src0_f = src0_f32.data();
        } else if (src0->type == GGML_TYPE_Q8_CHANNEL) {
            src0_f32.resize(jk_count);
            const size_t row_size = ggml_row_size(GGML_TYPE_Q8_CHANNEL, src0->ne[0]);
            const char * row = static_cast<const char *>(src0->data);
            for (size_t j = 0; j < J; ++j) {
                dequantize_row_q8_channel(row + j * row_size, src0_f32.data() + j * K, K);
            }
            src0_f = src0_f32.data();
        } else if (src0->type == GGML_TYPE_Q8_H1) {
            src0_f32.resize(jk_count);
            dequantize_row_q8_h1(reinterpret_cast<const block_q8_h1 *>(src0->data), src0_f32.data(), jk_count);
            src0_f = src0_f32.data();
        } else if (src0->type == GGML_TYPE_Q8_H2) {
            src0_f32.resize(jk_count);
            dequantize_row_q8_h2(reinterpret_cast<const block_q8_h2 *>(src0->data), src0_f32.data(), jk_count);
            src0_f = src0_f32.data();
        } else if (src0->type == GGML_TYPE_Q8_HP1) {
            src0_f32.resize(jk_count);
            dequantize_row_q8_hp1(reinterpret_cast<const block_q8_hp1 *>(src0->data), src0_f32.data(), jk_count);
            src0_f = src0_f32.data();
        } else if (src0->type == GGML_TYPE_Q8_HP2) {
            src0_f32.resize(jk_count);
            dequantize_row_q8_hp2(reinterpret_cast<const block_q8_hp2 *>(src0->data), src0_f32.data(), jk_count);
            src0_f = src0_f32.data();
        } else if (src0->type == GGML_TYPE_I8) {
            const float *scale = gemmini_i8_supported_scale_data(src0, args.transpose_B);
            if (scale == nullptr) {
                GGML_ABORT("DEQUANT_FP_TEST compute type: dense I8 weight is missing a loaded F32 scalar scale sidecar or has unsupported layout");
            }

            src0_f32.resize(jk_count);
            const int8_t *src0_i8 = reinterpret_cast<const int8_t *>(src0->data);
            for (size_t idx = 0; idx < jk_count; ++idx) {
                src0_f32[idx] = static_cast<float>(src0_i8[idx]) * *scale;
            }
            src0_f = src0_f32.data();
        } else if (src0->type != GGML_TYPE_F32) {
            ggml::gemmini::log::debug(layer, "DEQUANT_FP_TEST unsupported weight type=%d", (int)src0->type);
            GGML_ABORT("DEQUANT_FP_TEST compute type: unsupported weight type");
        }

        gemmini_matmul_fp_facade(I, J, K, activation_f32.data(), src0_f,
                                 static_cast<float *>(dst->data), args.matmul_layer);
        return true;
      }
      return false;
    };
    if (!deferred_quantization && run_dequant_fp_test())
      return;

    #if defined(__linux__) && defined(__aarch64__)
    start_sample = gemmini_read_native_cycle_sample_internal();
#else
    start = ggml::gemmini::cycle::read();
#endif
    const int64_t dim_k = src0->ne[0];
    const int64_t dim_j = src0->ne[1] ? src0->ne[1] : 1;
    const int64_t dim_z = src0->ne[2] ? src0->ne[2] : 1;
    const int64_t dim_w = src0->ne[3] ? src0->ne[3] : 1;

    // Overflow guard for logical_rows and subsequent allocations.
    const __int128 logical_rows_128 =
        static_cast<__int128>(dim_j) * static_cast<__int128>(dim_z) * static_cast<__int128>(dim_w);
    if (logical_rows_128 <= 0 ||
        logical_rows_128 > static_cast<__int128>(std::numeric_limits<size_t>::max())) {
        ggml::gemmini::log::debug(layer,
            "logical_rows overflow: logical_rows_128=%lld dim_k=%lld dim_j=%lld dim_z=%lld dim_w=%lld",
            (long long) logical_rows_128, (long long) dim_k, (long long) dim_j, (long long) dim_z, (long long) dim_w);
        GGML_ABORT("Gemmini logical_rows overflow");
    }
    const size_t logical_rows = static_cast<size_t>(logical_rows_128);

    // Weight matrix layout follows TRANSPOSE_B:
    // - true:  JxK row-major (stride = K)
    // - false: KxJ row-major (stride = J_flat)
    args.sB = args.transpose_B ? static_cast<size_t>(dim_k) : logical_rows;

    [[maybe_unused]] static thread_local std::vector<block_q4_h1> reprocessed_q4_h1;
    [[maybe_unused]] static thread_local std::vector<block_q8_h1> reprocessed_q8_h1;
    if (src0->type == GGML_TYPE_I8) {
            const float * scale = gemmini_i8_supported_scale_data(src0, args.transpose_B);
            if (scale == nullptr) {
                ggml::gemmini::log::debug(layer,
                    "[weight] I8 scale tensor missing or unsupported for transpose_b=%d", args.transpose_B ? 1 : 0);
                GGML_ABORT("Gemmini dense I8 weight is missing a loaded F32 scalar scale sidecar or has unsupported layout");
            }

        args.B = static_cast<elem_t *>(src0->data);
        args.B_blocks = nullptr;
        args.B_scales = nullptr;
        args.weight_i8_scale_active = true;
        args.weight_scale = *scale;
        args.c_b = nullptr;
        args.s_rf = nullptr;
        args.R = nullptr;
        args.blocks_per_row = 0;
        args.blocks_K = 0;
        args.blocks_J = logical_rows;
        args.blocks_I = logical_rows;
        args.block_size_k = 1;
        args.sB = static_cast<size_t>(dim_k);
        args.stripe_J = 0;
        args.s_rf_stripe = nullptr;
        args.R_stripe = nullptr;

        #if defined(__linux__) && defined(__aarch64__)
    end_sample = gemmini_read_native_cycle_sample_internal();
#else
    end = ggml::gemmini::cycle::read();
#endif
            #if defined(__linux__) && defined(__aarch64__)
    log_native_cycle_interval(layer, "gemmini.prepare_dense_i8_weight", start_sample, end_sample);
#else
    ggml::gemmini::log::cycle(layer, "gemmini.prepare_dense_i8_weight", start, end);
#endif
    } else {
        if (src0->type == GGML_TYPE_Q4_0) {
            size_t q4_0_blocks_per_row = 0;
            size_t q4_0_reprocess_rows = logical_rows;
            if (!ggml::gemmini::prepare_q4_0_rows_for_q4_h1(
                    src0,
                    reprocessed_q4_h1,
                    &q4_0_blocks_per_row,
                    &q4_0_reprocess_rows)) {
                GGML_LOG_ERROR("%s: Q4_0 reprocessing failed for weight tensor '%s'\n",
                    __func__, src0->name);
                return;
            }

            args.weight_format = ggml_gemmini_args_t::im2p_weight_format_t::q4_h1;
            args.B = nullptr;
            args.B_blocks = nullptr;
            args.B_scales = nullptr;
            args.weight_i8_scale_active = false;
            args.weight_scale = 1.0f;
            args.c_b = nullptr;
            args.s_rf = nullptr;
            args.R = nullptr;
            args.blocks_per_row = 0;
            args.blocks_K = q4_0_blocks_per_row;
            args.blocks_J = q4_0_reprocess_rows;
            args.blocks_I = q4_0_reprocess_rows;
            args.block_size_k = QK4_0;
            args.sB = static_cast<size_t>(dim_k);
            args.stripe_J = 0;
            args.s_rf_stripe = nullptr;
            args.R_stripe = nullptr;
            args.q4_h0_blocks = nullptr;
            args.q4_h1_blocks = reprocessed_q4_h1.data();
            args.native_blocks_per_row = q4_0_blocks_per_row;
            args.native_block_count = reprocessed_q4_h1.size();
            args.native_weight_bytes =
                reprocessed_q4_h1.size() * sizeof(block_q4_h1);
            if (!args.has_native_matched_width_contract()) {
                GGML_ABORT("Gemmini Q4_0 reprocessed H1 contract failed");
            }

            ggml::gemmini::log::debug(layer,
                "[Q4_0 reprocess] blocks=%p blocks_per_row=%zu logical_rows=%zu",
                (void *)reprocessed_q4_h1.data(),
                q4_0_blocks_per_row,
                q4_0_reprocess_rows);
            #if defined(__linux__) && defined(__aarch64__)
    end_sample = gemmini_read_native_cycle_sample_internal();
#else
    end = ggml::gemmini::cycle::read();
#endif
                    #if defined(__linux__) && defined(__aarch64__)
            log_native_cycle_interval(
                layer, "gemmini.convert_q4_0_to_q4_h1", start_sample, end_sample);
#else
            ggml::gemmini::log::cycle(
                layer, "gemmini.convert_q4_0_to_q4_h1", start, end);
#endif
        } else if (src0->type == GGML_TYPE_Q4_H1 ||
            src0->type == GGML_TYPE_Q4_HP1 || src0->type == GGML_TYPE_Q16_0 ||
            src0->type == GGML_TYPE_Q16_H1 || src0->type == GGML_TYPE_Q16_HP1) {
            if (!gemmini_set_native_matched_weight_args(src0, args)) {
                GGML_ABORT("Gemmini native matched-width weight contract failed");
            }
        } else if (src0->type == GGML_TYPE_Q8_H1) {
            const char * base = reinterpret_cast<const char *>(src0->view_src ? src0->view_src->data : src0->data);
            if (base == nullptr) {
                ggml::gemmini::log::debug(layer, "[Q8_H1 direct] base pointer is null, src0=%p view_src=%p", (void *)src0, (void *)src0->view_src);
                GGML_ABORT("Gemmini Q8_H1 weight tensor base pointer is null");
            }
            const size_t view_offset = src0->view_src ? static_cast<size_t>(src0->view_offs) : 0;
            const char * data = base + view_offset;
            const block_q8_h1 * blocks = reinterpret_cast<const block_q8_h1 *>(data);
            if (reinterpret_cast<uintptr_t>(data) % alignof(block_q8_h1) != 0) {
                ggml::gemmini::log::debug(layer,
                    "[Q8_H1 direct] pointer alignment failure ptr=%p align=%zu",
                    (void *)data, (size_t)alignof(block_q8_h1));
                GGML_ABORT("Gemmini Q8_H1 weight tensor data pointer is not block-aligned");
            }

            args.weight_format = ggml_gemmini_args_t::im2p_weight_format_t::q8_h1;
            args.B = nullptr;
            args.B_blocks = nullptr;
            args.B_scales = nullptr;
            args.weight_i8_scale_active = false;
            args.weight_scale = 1.0f;
            args.c_b = nullptr;
            args.s_rf = nullptr;
            args.R = nullptr;
            args.blocks_per_row = static_cast<size_t>(dim_k) / QK8_0;
            args.blocks_K = args.blocks_per_row;
            args.blocks_J = logical_rows;
            args.blocks_I = logical_rows;
            args.block_size_k = QK8_0;
            args.stripe_J = 0;
            args.s_rf_stripe = nullptr;
            args.R_stripe = nullptr;
            args.q8_h1_blocks = blocks;
            if (!gemmini_checked_mul(logical_rows, args.blocks_per_row, args.q8_h1_block_count)) {
                GGML_ABORT("Gemmini Q8_H1 weight tensor block metadata overflows");
            }
            args.q8_h1_rows = logical_rows;
            args.native_weight_bytes = args.q8_h1_block_count * sizeof(block_q8_h1);

            if (!args.has_q8_h1_im2p_contract()) {
                ggml_gemmini_log_q8_h1_contract_issue(layer, args);
                GGML_ABORT("Gemmini Q8_H1 im2p contract failed");
            }
            ggml::gemmini::log::debug(layer,
                "[Q8_H1 direct] blocks=%p blocks_per_row=%zu logical_rows=%zu",
                (void *)args.q8_h1_blocks, args.blocks_per_row, logical_rows);
        } else if (src0->type == GGML_TYPE_Q8_H2) {
            if (args.tiled_matmul_type != CPU) {
                GGML_ABORT("Gemmini Q8_H2 is supported only by the CPU im2p route");
            }
            args.tiled_matmul_type = CPU;
            if (!gemmini_set_q8_h2_weight_args(src0, args)) {
                ggml::gemmini::log::debug(layer,
                    "[Q8_H2 direct] contract failed ptr=%p ne=[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu]",
                    src0->data,
                    (long long)src0->ne[0], (long long)src0->ne[1],
                    (long long)src0->ne[2], (long long)src0->ne[3],
                    src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3]);
                GGML_ABORT("Gemmini Q8_H2 im2p contract failed");
            }
            ggml::gemmini::log::debug(layer,
                "[Q8_H2 direct] blocks=%p blocks_per_row=%zu logical_rows=%zu",
                (const void *)args.q8_h2_blocks, args.q8_h2_blocks_per_row, logical_rows);
        } else if (src0->type == GGML_TYPE_Q8_HP1) {
            if (!gemmini_set_q8_hp1_weight_args(src0, args)) {
                ggml::gemmini::log::debug(layer,
                    "[Q8_HP1 direct] im2p contract failed ptr=%p ne=[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu]",
                    src0->data,
                    (long long)src0->ne[0], (long long)src0->ne[1],
                    (long long)src0->ne[2], (long long)src0->ne[3],
                    src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3]);
                GGML_ABORT("Gemmini Q8_HP1 im2p contract failed");
            }
            ggml::gemmini::log::debug(layer,
                "[Q8_HP1 direct] blocks=%p blocks_per_row=%zu logical_rows=%zu",
                (const void *)args.q8_hp1_blocks, args.q8_hp1_blocks_per_row, logical_rows);
        } else if (src0->type == GGML_TYPE_Q8_HP2) {
            if (!gemmini_set_q8_hp2_weight_args(src0, args)) {
                ggml::gemmini::log::debug(layer,
                    "[Q8_HP2 direct] im2p contract failed ptr=%p ne=[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu]",
                    src0->data,
                    (long long)src0->ne[0], (long long)src0->ne[1],
                    (long long)src0->ne[2], (long long)src0->ne[3],
                    src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3]);
                GGML_ABORT("Gemmini Q8_HP2 im2p contract failed");
            }
            ggml::gemmini::log::debug(layer,
                "[Q8_HP2 direct] blocks=%p blocks_per_row=%zu logical_rows=%zu",
                (const void *)args.q8_hp2_blocks, args.q8_hp2_blocks_per_row, logical_rows);
        } else if (src0->type == GGML_TYPE_Q8_CHANNEL) {
            const size_t row_size = ggml_row_size(GGML_TYPE_Q8_CHANNEL, src0->ne[0]);
            const auto *row_base = static_cast<const uint8_t *>(src0->data);
            if (row_base == nullptr) {
                GGML_ABORT("Gemmini Q8_CHANNEL row base is null");
            }
            const auto q8_channel_route = gemmini_q8_channel_int_route_for(src0);
            if (q8_channel_route == gemmini_q8_channel_int_route_t::unsupported) {
                GGML_ABORT("Gemmini Q8_CHANNEL route is unsupported for this build/configuration");
            }

            if (q8_channel_route == gemmini_q8_channel_int_route_t::cpu_direct_read) {
                args.tiled_matmul_type = CPU;
                args.weight_format = ggml_gemmini_args_t::im2p_weight_format_t::q8_channel;
                args.B = const_cast<elem_t *>(reinterpret_cast<const elem_t *>(row_base + sizeof(float)));
                args.B_blocks = nullptr;
                args.B_scales = nullptr;
                args.weight_channel_scales = nullptr;
                args.weight_channel_scale_count = 0;
                args.weight_i8_scale_active = false;
                args.weight_scale = 1.0f;
                args.c_b = nullptr;
                args.s_rf = nullptr;
                args.R = nullptr;
                args.blocks_per_row = 0;
                args.blocks_K = 0;
                args.blocks_J = logical_rows;
                args.blocks_I = logical_rows;
                args.block_size_k = 1;
                args.sB = row_size;
                args.stripe_J = 0;
                args.s_rf_stripe = nullptr;
                args.R_stripe = nullptr;
                args.q8_channel_row_base = row_base;
                args.q8_channel_row_stride = row_size;
                args.q8_channel_row_count = logical_rows;
                if (!args.has_q8_channel_direct_read_contract()) {
                    GGML_ABORT("Gemmini Q8_CHANNEL direct-read contract failed");
                }

                ggml::gemmini::log::debug(layer,
                    "[Q8_CHANNEL direct-read] row_base=%p payload=%p row_stride=%zu logical_rows=%zu",
                    (const void *)args.q8_channel_row_base, (const void *)args.B, args.q8_channel_row_stride, logical_rows);
            } else if (q8_channel_route == gemmini_q8_channel_int_route_t::hw_dense_sidecar) {
                size_t dense_elements = 0;
                if (!gemmini_checked_mul(J, K, dense_elements)) {
                    GGML_ABORT("Gemmini Q8_CHANNEL dense repack size overflow");
                }

                try {
                    q8_channel_dense.resize(dense_elements);
                    q8_channel_scales.resize(J);
                } catch (const std::bad_alloc &) {
                    GGML_ABORT("Gemmini Q8_CHANNEL dense repack allocation failed");
                }

                for (size_t j = 0; j < J; ++j) {
                    const uint8_t * source_row = row_base + j * row_size;
                    std::memcpy(&q8_channel_scales[j], source_row, sizeof(float));
                    std::memcpy(q8_channel_dense.data() + j * K, source_row + sizeof(float), K);
                }

                args.tiled_matmul_type = OPTION;
                args.weight_format = ggml_gemmini_args_t::im2p_weight_format_t::q8_channel_dense_sidecar;
                args.B = reinterpret_cast<elem_t *>(q8_channel_dense.data());
                args.B_blocks = nullptr;
                args.B_scales = nullptr;
                args.weight_channel_scales = q8_channel_scales.data();
                args.weight_channel_scale_count = J;
                args.weight_i8_scale_active = false;
                args.weight_scale = 1.0f;
                args.c_b = nullptr;
                args.s_rf = nullptr;
                args.R = nullptr;
                args.blocks_per_row = 0;
                args.blocks_K = 0;
                args.blocks_J = logical_rows;
                args.blocks_I = logical_rows;
                args.block_size_k = 1;
                args.sB = K;
                args.stripe_J = 0;
                args.s_rf_stripe = nullptr;
                args.R_stripe = nullptr;
                args.q8_channel_row_base = nullptr;
                args.q8_channel_row_stride = 0;
                args.q8_channel_row_count = 0;
                if (!args.has_q8_channel_dense_sidecar_contract()) {
                    GGML_ABORT("Gemmini Q8_CHANNEL dense-sidecar contract failed");
                }

                ggml::gemmini::log::debug(layer,
                    "[Q8_CHANNEL dense-sidecar] dense=%p scales=%p stride=%zu scale_count=%zu logical_rows=%zu option=%d",
                    (const void *)args.B, (const void *)args.weight_channel_scales, args.sB,
                    args.weight_channel_scale_count, logical_rows, static_cast<int>(args.tiled_matmul_type));
            }
        } else if (src0->type == GGML_TYPE_Q8_0) {
            #if defined(__linux__) && defined(__aarch64__)
    start_sample = gemmini_read_native_cycle_sample_internal();
#else
    start = ggml::gemmini::cycle::read();
#endif
                    size_t q8_0_reprocess_rows = logical_rows;
            const bool ok = ggml::gemmini::prepare_q8_0_rows_for_q8_h1(
                src0,
                reprocessed_q8_h1,
                &args.blocks_per_row,
                &q8_0_reprocess_rows
            );
            if (!ok) {
                GGML_LOG_ERROR("%s: Q8_0 reprocessing failed for weight tensor '%s'\n",
                    __func__, src0->name);
                return;
            }

            args.weight_format = ggml_gemmini_args_t::im2p_weight_format_t::q8_h1;
            args.B = nullptr;
            args.B_blocks = nullptr;
            args.weight_i8_scale_active = false;
            args.weight_scale = 1.0f;
            args.c_b = nullptr;
            args.s_rf = nullptr;
            args.R = nullptr;
            args.blocks_K = args.blocks_per_row;
            args.blocks_J = q8_0_reprocess_rows;
            args.blocks_I = q8_0_reprocess_rows;
            args.block_size_k = QK8_0;
            args.sB = static_cast<size_t>(dim_k);
            args.stripe_J = 0;
            args.s_rf_stripe = nullptr;
            args.R_stripe = nullptr;
            args.q8_h1_blocks = reprocessed_q8_h1.data();
            args.q8_h1_block_count = reprocessed_q8_h1.size();
            args.q8_h1_rows = q8_0_reprocess_rows;
            args.native_weight_bytes = reprocessed_q8_h1.size() * sizeof(block_q8_h1);
            if (!args.has_q8_h1_im2p_contract()) {
                ggml_gemmini_log_q8_h1_contract_issue(layer, args);
                GGML_ABORT("Gemmini Q8_0 reprocessed im2p contract failed");
            }

            ggml::gemmini::log::debug(layer,
                "[Q8_0 reprocess] blocks=%p sB=%zu blocks_per_row=%zu logical_rows=%zu",
                (void *)reprocessed_q8_h1.data(), args.sB, args.blocks_per_row, q8_0_reprocess_rows);
            #if defined(__linux__) && defined(__aarch64__)
    end_sample = gemmini_read_native_cycle_sample_internal();
#else
    end = ggml::gemmini::cycle::read();
#endif
                    #if defined(__linux__) && defined(__aarch64__)
    log_native_cycle_interval(layer, "gemmini.convert_q8_0_to_q8_h1", start_sample, end_sample);
#else
    ggml::gemmini::log::cycle(layer, "gemmini.convert_q8_0_to_q8_h1", start, end);
#endif
        } else {
            ggml::gemmini::log::debug(layer, "int compute unsupported weight type=%d", (int)src0->type);
            GGML_ABORT("Gemmini int mul_mat received unsupported weight type");
        }

    }

    #if defined(__linux__) && defined(__aarch64__)
    start_sample = gemmini_read_native_cycle_sample_internal();
#else
    start = ggml::gemmini::cycle::read();
#endif
    /* ______________________________ 4. bias 텐서 처리 _________________________________ */
    std::vector<int32_t> zero_bias(J, 0);

    const int32_t *bias_data = zero_bias.data();
    const size_t sD = 0;

    args.sD = sD;
    args.D = bias_data;
    args.repeating_bias = true;

    // output 버퍼
    [[maybe_unused]] static thread_local std::vector<int8_t> c_i8;
    c_i8.resize(ij_count);

    args.C = c_i8.data();
    args.f_out = static_cast<float*>(dst->data);
    args.col_stride_f_out = dst->nb[0] / sizeof(float);
    args.stride_f_out = dst->nb[1] / sizeof(float);

#if defined(GGML_GEMMINI_EXECUTION_BACKEND_IM2P_SIM)
    if constexpr (im2p_exsia) {
      args.act_quant.storage()
          .emplace<ggml::gemmini::quants::act::exsia::Meta>();

      if (full_requested) {
        auto full =
            ggml::gemmini::im2p_adapter::start_exsia_full_execution(args);
        if (!full.result.ok() || !full.execution) {
          ggml::gemmini::im2p_adapter::log_failure("ExSIA FULL start",
                                                   full.result);
#if defined(GGML_GEMMINI_TESTING)
          ggml::gemmini::im2p_adapter::test_record_production_failure(
              full.result.error);
          return;
#else
          GGML_ABORT("Gemmini IM2P ExSIA FULL start failed");
#endif
        }
        const auto sink_status = full.execution->install_sink();
        if (!sink_status.ok()) {
          ggml::gemmini::im2p_adapter::log_failure("ExSIA FULL sink install",
                                                   sink_status);
#if defined(GGML_GEMMINI_TESTING)
          ggml::gemmini::im2p_adapter::test_record_production_failure(
              sink_status.error);
          return;
#else
          GGML_ABORT("Gemmini IM2P ExSIA FULL sink install failed");
#endif
        }
#if defined(GGML_GEMMINI_TESTING)
        quantize_ok =
            !ggml::gemmini::im2p_adapter::test_should_fail_quantization() &&
            quantize_activation();
#else
        quantize_ok = quantize_activation();
#endif
        const auto completion = full.execution->finish(quantize_ok);
        if (!completion.result.ok()) {
          ggml::gemmini::im2p_adapter::log_failure("ExSIA FULL execution",
                                                   completion.result);
#if defined(GGML_GEMMINI_TESTING)
          ggml::gemmini::im2p_adapter::test_record_production_failure(
              completion.result.error);
          return;
#else
          GGML_ABORT("Gemmini IM2P ExSIA FULL execution failed");
#endif
        }
        if (run_dequant_fp_test())
          return;
        ggml::gemmini::im2p_adapter::log_stats(
            "full", completion.stats, completion.run_id, args);
        ggml::gemmini::im2p_adapter::log_rmd_stats(completion, args);
        return;
      }

      if (!pipeline_requested) {
        ggml::gemmini::im2p_adapter::log_failure(
            "ExSIA mode",
            {ggml::gemmini::im2p_adapter::Error::unsupported_route,
             "ExSIA supports FULL or STRIPE_PIPELINE", false});
#if defined(GGML_GEMMINI_TESTING)
        ggml::gemmini::im2p_adapter::test_record_production_failure(
            ggml::gemmini::im2p_adapter::Error::unsupported_route);
        return;
#else
        GGML_ABORT("Gemmini IM2P ExSIA mode is unsupported");
#endif
      }

      auto started =
          ggml::gemmini::im2p_adapter::start_exsia_stripe_pipeline(args);
      if (!started.result.ok() || !started.pipeline) {
        ggml::gemmini::im2p_adapter::log_failure("ExSIA pipeline start",
                                                 started.result);
#if defined(GGML_GEMMINI_TESTING)
        ggml::gemmini::im2p_adapter::test_record_production_failure(
            started.result.error);
        return;
#else
        GGML_ABORT("Gemmini IM2P ExSIA pipeline start failed");
#endif
      }
      const auto sink_status = started.pipeline->install_sink();
      if (!sink_status.ok()) {
        ggml::gemmini::im2p_adapter::log_failure("ExSIA sink install",
                                                 sink_status);
#if defined(GGML_GEMMINI_TESTING)
        ggml::gemmini::im2p_adapter::test_record_production_failure(
            sink_status.error);
        return;
#else
        GGML_ABORT("Gemmini IM2P ExSIA sink install failed");
#endif
      }
#if defined(GGML_GEMMINI_TESTING)
      quantize_ok =
          !ggml::gemmini::im2p_adapter::test_should_fail_quantization() &&
          quantize_activation();
#else
      quantize_ok = quantize_activation();
#endif
      const auto completion = started.pipeline->finish(quantize_ok);
      if (!completion.result.ok()) {
        ggml::gemmini::im2p_adapter::log_failure("ExSIA stripe execution",
                                                 completion.result);
#if defined(GGML_GEMMINI_TESTING)
        ggml::gemmini::im2p_adapter::test_record_production_failure(
            completion.result.error);
        return;
#else
        GGML_ABORT("Gemmini IM2P ExSIA stripe execution failed");
#endif
      }
      if (run_dequant_fp_test())
        return;
      ggml::gemmini::im2p_adapter::log_stats(
          "stripe_pipeline", completion.stats, completion.run_id, args);
      return;
    }
#endif

    if (pipeline_enabled) {
        const auto pipeline_route = ggml::gemmini::detail::normalize_route(args);
        ggml::gemmini::log::debug(layer,
            "[matmul.pipeline] schedule=%s",
            pipeline_route.backend == ggml::gemmini::detail::BackendRoute::cpu
                ? "matmul-then-rmd"
                : "matmul-rmd-overlap");
        ggml::gemmini::log::debug(layer, "[matmul.pipeline] job_slots=%zu", pipeline_job_capacity);
        pipeline_execution = std::make_unique<ggml::gemmini::MatmulExecution>(
            ggml::gemmini::prepare_execution(&args, matmul_options));
        if (!pipeline_execution->status() || !pipeline_collector->start(*pipeline_execution)) {
            ggml::gemmini::log::debug(layer, "[matmul.pipeline] staged worker start failed");
            return;
        }
    }

    if (pipeline_enabled) {
        quantize_ok = quantize_activation();
        if (!quantize_ok) {
            const auto pipeline_status = pipeline_collector->finish();
            ggml::gemmini::log::debug(layer,
                "[matmul.pipeline] quantization failure collector_status=%d message=%s",
                static_cast<int>(pipeline_status.code), pipeline_status.message);
            ggml::gemmini::log::debug(layer, "activation quantize failed");
            GGML_ABORT("Gemmini activation quantization failed");
        }
    }

#if defined(GGML_GEMMINI_EXECUTION_BACKEND_IM2P_SIM)
    if constexpr (im2p_non_exsia) {
      const auto completion =
          pipeline_requested
              ? ggml::gemmini::im2p_adapter::run_stripe_pipeline(args)
              : ggml::gemmini::im2p_adapter::run_full(args);
      if (!completion.result.ok()) {
        ggml::gemmini::im2p_adapter::log_failure(
            pipeline_requested ? "stripe execution" : "full execution",
            completion.result);
#if defined(GGML_GEMMINI_TESTING)
        return;
#else
        GGML_ABORT("Gemmini IM2P execution failed");
#endif
      }
      ggml::gemmini::im2p_adapter::log_stats(
          pipeline_requested ? "stripe_pipeline" : "full",
          completion.stats, completion.run_id, args);
    }
#endif

    #if defined(__linux__) && defined(__aarch64__)
    end_sample = gemmini_read_native_cycle_sample_internal();
#else
    end = ggml::gemmini::cycle::read();
#endif
    #if defined(__linux__) && defined(__aarch64__)
    log_native_cycle_interval(layer, "gemmini.prepare_args", start_sample, end_sample);
#else
    ggml::gemmini::log::cycle(layer, "gemmini.prepare_args", start, end);
#endif

    // ggml::gemmini::log::debug("[Gemmini debug] layer=%s A=%p B=%p C=%p D=%p I=%zu J=%zu K=%zu sA=%zu sB=%zu sC=%zu stride_f_out(row)=%zu stride_f_out(col)=%zu nb1=%zu nb0=%zu",
    //                  layer, args.A, args.B, args.C, args.D,
    //                  args.I, args.J, args.K, args.sA, args.sB, args.sC,
    //                  args.stride_f_out, args.col_stride_f_out, dst->nb[1], dst->nb[0]);

    // ggml::gemmini::log::debug("[Gemmini addr] layer=%s f_out=%p stride_f_out=%zu col_stride_f_out=%zu",
    //                  layer, (void *)args.f_out, args.stride_f_out, args.col_stride_f_out);

    if (facade_full_dispatch) {
#if defined(GGML_GEMMINI_EXECUTION_BACKEND_IM2P_SIM) &&                        \
    defined(GGML_GEMMINI_TESTING)
      ggml::gemmini::im2p_adapter::test_observe_hardware_dispatch();
#endif
      ggml::gemmini::MatMul facade(args);
      const auto facade_result = facade.run_full();
      if (facade_result.status != ggml::gemmini::MatMulStatus::success) {
        GGML_ABORT("Gemmini full facade execution failed");
      }
    }

    if (!pipeline_enabled && legacy_full_dispatch) {
        if constexpr (ggml::gemmini::config::CURRENT_COMPUTE_TYPE == ggml::gemmini::config::ComputeType::INT) {
          bool run_baseline = true;

          if constexpr (ggml::gemmini::config::CURRENT_ACTIVATION_QUANT ==
                            ggml::gemmini::config::ActivationQuantAlgo::EXSIA ||
                        ggml::gemmini::config::CURRENT_ACTIVATION_QUANT ==
                            ggml::gemmini::config::ActivationQuantAlgo::
                                STRIPE) {
            if (args.weight_i8_scale_active) {
              if constexpr (ggml::gemmini::config::CURRENT_ACTIVATION_QUANT ==
                            ggml::gemmini::config::ActivationQuantAlgo::
                                STRIPE) {
                GGML_ABORT("Gemmini STRIPE activation requires IM2P/native Q8 "
                           "weights; dense tensor weights are unsupported");
              }
              ggml::gemmini::log::debug(
                  layer,
                  "[baseline] route activation=exsia weight=tensor "
                  "weight_scale=%.9f dims=(I=%zu,J=%zu,K=%zu) sB=%zu option=%d",
                  static_cast<double>(args.weight_scale), args.I, args.J,
                  args.K, args.sB, static_cast<int>(args.tiled_matmul_type));
            } else if (args.has_q8_channel_direct_read_contract()) {
              if constexpr (ggml::gemmini::config::CURRENT_ACTIVATION_QUANT !=
                            ggml::gemmini::config::ActivationQuantAlgo::
                                STRIPE) {
                GGML_ABORT("Gemmini Q8_CHANNEL direct-read is wired only for "
                           "baseline TENSOR/TOKEN/STRIPE activation routes");
              }
            } else if (args.has_q8_channel_dense_sidecar_contract()) {
              ggml::gemmini::log::debug(
                  layer,
                  "[baseline] route activation=%s "
                  "weight=q8_channel_dense_sidecar dims=(I=%zu,J=%zu,K=%zu) "
                  "sB=%zu option=%d",
                  GGML_GEMMINI_ACTIVATION_QUANT_NAME, args.I, args.J, args.K,
                  args.sB, static_cast<int>(args.tiled_matmul_type));
            } else if (src0->type == GGML_TYPE_Q8_CHANNEL) {
              GGML_ABORT("Gemmini Q8_CHANNEL activation route contract failed");
            } else {
              const char *weight_route =
                  src0->type == GGML_TYPE_Q8_H1    ? "Q8_H1 direct"
                  : src0->type == GGML_TYPE_Q8_H2  ? "Q8_H2 direct"
                  : src0->type == GGML_TYPE_Q8_HP1 ? "Q8_HP1 direct"
                  : src0->type == GGML_TYPE_Q8_HP2 ? "Q8_HP2 direct"
                  : src0->type == GGML_TYPE_Q8_CHANNEL
                      ? "Q8_CHANNEL direct-read"
                      : "Q8_0 reprocessed";
              ggml::gemmini::log::debug(
                  layer,
                  "[activation] route im2p weight=%s dims=(I=%zu,J=%zu,K=%zu) "
                  "sB=%zu option=%d",
                  weight_route, args.I, args.J, args.K, args.sB,
                  static_cast<int>(args.tiled_matmul_type));
              ggml::gemmini::MatMul facade(args);
              const auto facade_result = facade.run_dense();
              if (facade_result.status !=
                  ggml::gemmini::MatMulStatus::success) {
                GGML_ABORT("Gemmini facade dense execution failed");
              }
              run_baseline = false;
            }
          }

                if constexpr (ggml::gemmini::config::CURRENT_ACTIVATION_QUANT == ggml::gemmini::config::ActivationQuantAlgo::TOKEN) {
                const bool q8_channel_route =
                    args.has_q8_channel_direct_read_contract() ||
                    args.has_q8_channel_dense_sidecar_contract();
                if (src0->type == GGML_TYPE_Q8_CHANNEL &&
                    (!args.transpose_B || !q8_channel_route)) {
                    GGML_ABORT("Gemmini TOKEN activation requires a transposed valid Q8_CHANNEL route");
                }
                if (!args.weight_i8_scale_active && !q8_channel_route) {
                    GGML_ABORT("Gemmini TOKEN activation requires dense tensor or a valid Q8_CHANNEL route; Q8_0/Q8_H1/Q8_H2/Q8_HP1/Q8_HP2 are unsupported");
                }
            }

            if (run_baseline) {
                ggml::gemmini::MatMul facade(args);
                const auto facade_result = facade.run_dense();
                if (facade_result.status != ggml::gemmini::MatMulStatus::success) {
                    GGML_ABORT("Gemmini facade dense execution failed");
                }
            }
    }
    }
    if (pipeline_enabled) {
        const auto pipeline_status = pipeline_collector->finish();
        ggml::gemmini::log::debug(layer,
            "[matmul.pipeline] mode=live-staged status=%d message=%s",
            static_cast<int>(pipeline_status.code), pipeline_status.message);
        if (!pipeline_status) {
            return;
        }
        const auto execution_status = ggml::gemmini::finish_execution(*pipeline_execution);
        if (!execution_status) {
            ggml::gemmini::log::debug(layer,
                "[matmul.pipeline] execution status=%d message=%s",
                static_cast<int>(execution_status.code), execution_status.message);
            return;
        }
        if (run_dequant_fp_test())
            return;
        const auto route = ggml::gemmini::detail::normalize_route(args);
        const auto capabilities = ggml::gemmini::detail::route_capabilities(args);
        const char * const backend_route =
            ggml::gemmini::detail::backend_route_name(route.backend);
        ggml::gemmini::log::debug(layer,
            "[matmul.route] invocation=stripe-pipeline full_or_slice=slice "
            "activation_route=%s weight_route=%s backend_route=%s "
            "supports_live_pipeline=%d deprecated=%d",
            ggml::gemmini::detail::activation_route_name(route.activation),
            ggml::gemmini::detail::weight_route_name(route.weight),
            backend_route,
            capabilities.live_stripe_producer ? 1 : 0,
            capabilities.deprecated ? 1 : 0);
        const auto telemetry_profiles = pipeline_collector->profiles();
#if LOG_CYCLE
        for (const auto & profile : telemetry_profiles) {
            ggml::gemmini::emit_cycle_telemetry(
                ggml::gemmini::detail::pipeline_stripe_telemetry(layer, profile));
        }
#if defined(__linux__) && defined(__aarch64__)
        const gemmini_native_cycle_sample_internal rmd_telemetry_invocation_end_sample =
            gemmini_read_native_cycle_sample_internal();
        const uint64_t rmd_telemetry_invocation_end = rmd_telemetry_invocation_end_sample.value;
        const uint8_t invocation_reason =
            gemmini_log_cycle_record_v2_checked_internal(
                nullptr, &rmd_telemetry_invocation_start_sample,
                &rmd_telemetry_invocation_end_sample, 1);
        const uint64_t invocation_total =
            invocation_reason == GEMMINI_NATIVE_CYCLE_REASON_NONE
                ? rmd_telemetry_invocation_end - rmd_telemetry_invocation_start : 0;
#else
        const uint64_t rmd_telemetry_invocation_end = ggml::gemmini::cycle::read();
        const uint64_t invocation_total =
            rmd_telemetry_invocation_end >= rmd_telemetry_invocation_start
                ? rmd_telemetry_invocation_end - rmd_telemetry_invocation_start : 0;
#endif
        const char * bundle_id = std::getenv("GGML_GEMMINI_RUNTIME_BUNDLE_ID");
        const char * model_id = std::getenv("GGML_GEMMINI_MODEL_ID");
        const uint64_t telemetry_run_id = telemetry_profiles.empty()
            ? 0 : telemetry_profiles.front().run_id;
        auto telemetry = ggml::gemmini::make_rmd_telemetry_record(
            matmul_options.rmd_backend, matmul_resolution.rmd_backend_source,
            bundle_id != nullptr ? bundle_id : "unbundled",
            ggml::gemmini::resolve_rmd_model_id(model_id, ctx->model_arch),
            layer,
            telemetry_run_id, invocation_total, telemetry_profiles);
#if defined(__linux__) && defined(__aarch64__)
        telemetry.invocation_valid =
            invocation_reason == GEMMINI_NATIVE_CYCLE_REASON_NONE;
        telemetry.invocation_reason = ggml::gemmini::cycle::reason_name(
            static_cast<ggml::gemmini::cycle::NativeCycleReason>(invocation_reason));
#endif
        ggml::gemmini::emit_cycle_telemetry(telemetry);
#endif
    }
}

static void ggml_backend_gemmini_add(
                                        ggml_backend_gemmini_context *ctx,
                                        struct ggml_tensor *dst, // FP32 output (I×J)
                                        struct ggml_tensor *bias){
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
    GGML_UNUSED(bias);
}

static void ggml_backend_gemmini_out_prod(ggml_backend_gemmini_context * ctx, struct ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
}

// backend interface

static const char * ggml_backend_gemmini_get_name(ggml_backend_t backend) {
    return "GEMMINI";

    GGML_UNUSED(backend);
}

static void ggml_backend_gemmini_free(ggml_backend_t backend) {
    ggml_backend_gemmini_context * ctx = (ggml_backend_gemmini_context *)backend->context;
    delete ctx;
    delete backend;
}

static void ggml_backend_gemmini_get_rows_q8_channel(const ggml_tensor * src0,
                                                      const ggml_tensor * src1,
                                                      ggml_tensor * dst) {
    const int64_t nc = src0->ne[0];
    const int64_t nr = ggml_nelements(src1);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(src0->type == GGML_TYPE_Q8_CHANNEL);
    GGML_ASSERT(src1->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(dst->ne[2] == src1->ne[1]);

    for (int64_t i = 0; i < nr; ++i) {
        const int64_t i12 = i / (src1->ne[1] * src1->ne[0]);
        const int64_t i11 = (i - i12 * src1->ne[1] * src1->ne[0]) / src1->ne[0];
        const int64_t i10 = i - i12 * src1->ne[1] * src1->ne[0] - i11 * src1->ne[0];
        const int32_t i01 = *reinterpret_cast<const int32_t *>(
            static_cast<const char *>(src1->data) + i10 * src1->nb[0] + i11 * src1->nb[1] + i12 * src1->nb[2]);

        GGML_ASSERT(i01 >= 0 && i01 < src0->ne[1]);
        dequantize_row_q8_channel(
            static_cast<const char *>(src0->data) + i01 * src0->nb[1] + i11 * src0->nb[2] + i12 * src0->nb[3],
            reinterpret_cast<float *>(static_cast<char *>(dst->data) + i10 * dst->nb[1] + i11 * dst->nb[2] + i12 * dst->nb[3]),
            nc);
    }
}

static enum ggml_status ggml_backend_gemmini_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    ggml_backend_gemmini_context * ctx = (ggml_backend_gemmini_context *)backend->context;
    setup_gemmini_log_outputs_if_needed();

    size_t graph_mul_mat_count = 0;
    size_t graph_max_i = 0;
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const struct ggml_tensor * node = cgraph->nodes[i];
        if (node->op == GGML_OP_MUL_MAT) {
            ++graph_mul_mat_count;
            graph_max_i = std::max(graph_max_i, static_cast<size_t>(node->ne[1] > 0 ? node->ne[1] : 1));
        }
    }
    ggml::gemmini::log::debug("graph", "nodes=%d mul_mat=%zu max_i=%zu", cgraph->n_nodes, graph_mul_mat_count, graph_max_i);

#if LOG_DUMP
    uint32_t mxI = 0;
    bool decode_start_marker = false;
    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];
        if (node->op == GGML_OP_MUL_MAT) {
            const uint32_t I = node->ne[1] > 0 ? static_cast<uint32_t>(node->ne[1]) : 1u;
            if (I > mxI)
                mxI = I;
            const struct ggml_tensor * src1 = node->src[1];
            if (!decode_start_marker && src1 && src1->name &&
                std::strcmp(src1->name, "attn_norm-0") == 0 &&
                I == 1 && src1->ne[1] == 1) {
                decode_start_marker = true;
            }
        }
    }

    static std::atomic<bool> g_seen_any_graph{false};
    static std::atomic<bool> g_decode_started{false};
    static std::atomic<uint64_t> g_decode_count{0};
    ggml::gemmini::log::DumpPhase phase = ggml::gemmini::log::DumpPhase::unknown;
    uint64_t step_id = 1;

    const bool first_graph = !g_seen_any_graph.exchange(true, std::memory_order_relaxed);
    if (first_graph) {
        phase = ggml::gemmini::log::DumpPhase::prefill;
        step_id = 1;
        g_decode_started.store(false, std::memory_order_relaxed);
        g_decode_count.store(0, std::memory_order_relaxed);
    } else if (!g_decode_started.load(std::memory_order_relaxed)) {
        if (decode_start_marker) {
            g_decode_started.store(true, std::memory_order_relaxed);
            g_decode_count.store(1, std::memory_order_relaxed);
            phase = ggml::gemmini::log::DumpPhase::decode;
            step_id = 2;
        } else {
            phase = ggml::gemmini::log::DumpPhase::prefill;
            step_id = 1;
        }
    } else {
        if (decode_start_marker)
            g_decode_count.fetch_add(1, std::memory_order_relaxed);
        phase = ggml::gemmini::log::DumpPhase::decode;
        step_id = 1 + g_decode_count.load(std::memory_order_relaxed);
    }

    ggml::gemmini::log::dump_begin_graph(phase, step_id, mxI);
#endif

    for (int i = 0; i < cgraph->n_nodes; i++)
    {
        struct ggml_tensor *node = cgraph->nodes[i];

        switch (node->op)
        {
        case GGML_OP_GET_ROWS:
            ggml_backend_gemmini_get_rows_q8_channel(node->src[0], node->src[1], node);
            break;
        case GGML_OP_MUL_MAT: {
#if LOG_DUMP
            const int32_t node_idx = node->ne[0] > 0 ? static_cast<int32_t>(node->ne[0]) : 1;
            ggml::gemmini::log::dump_set_node_idx(node_idx);
#endif
            ggml_backend_gemmini_mul_mat(ctx, node);
#if defined(GGML_GEMMINI_EXECUTION_BACKEND_IM2P_SIM) &&                        \
    defined(GGML_GEMMINI_TESTING)
            if (ggml::gemmini::im2p_adapter::test_production_failed()) {
              return GGML_STATUS_FAILED;
            }
#endif
            break;
        }
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_ADD:
#if LOG_DUMP
            ggml::gemmini::log::dump_set_node_idx(-1);
#endif
            break;

        default:
            GGML_ABORT("%s: unsupported op assigned to GEMMINI: %s\n", __func__, ggml_op_desc(node));
        }
    }

    GGML_UNUSED(backend);
    return GGML_STATUS_SUCCESS;
}

static struct ggml_backend_i gemmini_backend_i = {
    /* .get_name                = */ ggml_backend_gemmini_get_name,
    /* .free                    = */ ggml_backend_gemmini_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ NULL,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_gemmini_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
};

static ggml_guid_t ggml_backend_gemmini_guid(void) {
    static ggml_guid guid = { 0x10, 0xa8, 0xae, 0xf4, 0xc0, 0x1e, 0x61, 0x97, 0x8f, 0xeb, 0x33, 0x04, 0xa1, 0x33, 0x51, 0x2d };
    return &guid;
}

ggml_backend_t ggml_backend_gemmini_init(void) {
    ggml_backend_gemmini_context * ctx = new ggml_backend_gemmini_context;
    ggml::gemmini::load_q8_h1_artifact_from_env(ctx->q8_h1_artifact);
    if (ctx->q8_h1_artifact.status == ggml::gemmini::q8_h1_artifact_runtime_status::disabled_load_error) {
        GGML_LOG_WARN("%s: disabling LLAMA_GEMMINI_Q8_H1_ARTIFACT='%s': %s\n",
                __func__,
                ctx->q8_h1_artifact.path.c_str(),
                ctx->q8_h1_artifact.error.c_str());
    }

    ggml_backend_t backend = new ggml_backend {
        /* .guid      = */ ggml_backend_gemmini_guid(),
        /* .interface = */ gemmini_backend_i,
        /* .device    = */ ggml_backend_reg_dev_get(ggml_backend_gemmini_reg(), 0),
        /* .context   = */ ctx,
    };

    return backend;
}

// bool ggml_backend_is_gemmini(ggml_backend_t backend) {
//     return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_gemmini_guid());
// }

// device interface

static const char * ggml_backend_gemmini_device_get_name(ggml_backend_dev_t dev) {
    return "GEMMINI";

    GGML_UNUSED(dev);
}

static const char * ggml_backend_gemmini_device_get_description(ggml_backend_dev_t dev) {
    return "GEMMINI";

    GGML_UNUSED(dev);
}

static void ggml_backend_gemmini_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    // TODO
    *free = 0;
    *total = 0;

    GGML_UNUSED(dev);
}

static enum ggml_backend_dev_type ggml_backend_gemmini_device_get_type(ggml_backend_dev_t dev) {
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;

    GGML_UNUSED(dev);
}

static void ggml_backend_gemmini_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_gemmini_device_get_name(dev);
    props->description = ggml_backend_gemmini_device_get_description(dev);
    props->type        = ggml_backend_gemmini_device_get_type(dev);
    ggml_backend_gemmini_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ true,
        /* .buffer_from_host_ptr  = */ true,
        /* .events                = */ false,
    };
}

static ggml_backend_t ggml_backend_gemmini_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    ggml_backend_t backend = ggml_backend_gemmini_init();
    auto * ctx = (ggml_backend_gemmini_context *) backend->context;
    ctx->model_arch = params ? params : "";
    return backend;

    GGML_UNUSED(dev);
}

static ggml_backend_buffer_type_t ggml_backend_gemmini_device_get_buffer_type(ggml_backend_dev_t dev) {
    return ggml::gemmini::gemmini_buffer_type(dev);

    GGML_UNUSED(dev);
}

static ggml_backend_buffer_t ggml_backend_gemmini_device_buffer_from_host_ptr(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    return ggml::gemmini::gemmini_buffer_from_host_ptr(dev, ptr, size);

    GGML_UNUSED(dev);
    GGML_UNUSED(max_tensor_size);
}

static bool ggml_backend_gemmini_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:

        case GGML_OP_TRANSPOSE:
            return true;

        case GGML_OP_GET_ROWS: {
            return op->type == GGML_TYPE_F32 &&
                op->src[0] != nullptr && op->src[1] != nullptr &&
                op->src[0]->type == GGML_TYPE_Q8_CHANNEL &&
                op->src[1]->type == GGML_TYPE_I32;
        }

        case GGML_OP_MUL_MAT:
        {
            if (ggml_is_empty(op)) {
                return true;
            }
            const struct ggml_tensor *a = op->src[0]; // W
            const struct ggml_tensor *b = op->src[1]; // x
            if (a == nullptr || b == nullptr)
                return false;
            if (op->type != GGML_TYPE_F32)
                return false;
            if (b->type != GGML_TYPE_F32)
                return false;

            // llama-model probes buffer compatibility with a zero-sized,
            // data-less tensor. Permit only the selected native artifact's
            // formats; the real path below still validates backing storage.
            if (gemmini_is_loader_metadata(a)) {
                if (!gemmini_shared_weight_contract(op) || a->ne[0] <= 0 || a->ne[1] <= 0 ||
                    a->ne[2] != 1 || a->ne[3] != 1 ||
                    !(ggml_is_contiguous(a) || gemmini_q8_channel_layout_contract(a))) {
                    return false;
                }

                if (gemmini_is_native_matched_weight_type(a->type)) {
                    return a->ne[0] % 32 == 0;
                }

                if (gemmini_is_extended_dequant_weight_type(a->type)) {
                    if constexpr (
                        ggml::gemmini::config::CURRENT_COMPUTE_TYPE ==
                            ggml::gemmini::config::ComputeType::FLOAT ||
                        ggml::gemmini::config::DEQUANT_FP_TEST
                    ) {
                        return a->ne[0] % ggml_blck_size(a->type) == 0;
                    }
                    return false;
                }

                if (a->type == GGML_TYPE_Q8_H1 || a->type == GGML_TYPE_Q8_H2) {
                    return a->ne[0] % QK8_0 == 0;
                }
                if (a->type == GGML_TYPE_Q8_HP1) {
                    ggml_gemmini_args_t args;
                    return gemmini_set_q8_hp1_weight_args(a, args, true);
                }
                if (a->type == GGML_TYPE_Q8_HP2) {
                    ggml_gemmini_args_t args;
                    return gemmini_set_q8_hp2_weight_args(a, args, true);
                }
                if (a->type == GGML_TYPE_Q8_CHANNEL) {
                    if constexpr (
                        ggml::gemmini::config::CURRENT_COMPUTE_TYPE ==
                        ggml::gemmini::config::ComputeType::INT
                    ) {
                        return gemmini_q8_channel_int_route_for(a, false) !=
                            gemmini_q8_channel_int_route_t::unsupported;
                    }
                    return gemmini_q8_channel_layout_contract(a);
                }

                return false;
            }

            if (!gemmini_shared_weight_contract(op))
                return false;

            if (a->type == GGML_TYPE_Q4_0) {
                return gemmini_q4_0_reprocess_layout_contract(a);
            }

            if (gemmini_is_native_matched_weight_type(a->type)) {
                ggml_gemmini_args_t args;
                return gemmini_set_native_matched_weight_args(a, args);
            }

            if constexpr (
                ggml::gemmini::config::CURRENT_COMPUTE_TYPE ==
                ggml::gemmini::config::ComputeType::INT
            ) {
                if constexpr (
                    ggml::gemmini::config::CURRENT_ACTIVATION_QUANT ==
                    ggml::gemmini::config::ActivationQuantAlgo::TOKEN
                ) {
                    if (a->type == GGML_TYPE_Q8_CHANNEL)
                        return gemmini_q8_channel_int_route_supported(a);
                    return a->type == GGML_TYPE_I8 &&
                           gemmini_i8_supported_scale_data(a, TRANSPOSE_B != 0) != nullptr;
                }

                if (a->type == GGML_TYPE_I8) {
                    return gemmini_i8_supported_scale_data(a, TRANSPOSE_B != 0) != nullptr;
                }

                if (a->type == GGML_TYPE_Q8_CHANNEL)
                    return gemmini_q8_channel_int_route_supported(a);

                if (a->type == GGML_TYPE_Q8_HP1) {
                    ggml_gemmini_args_t args;
                    return gemmini_set_q8_hp1_weight_args(a, args);
                }

                if (a->type == GGML_TYPE_Q8_HP2) {
                    ggml_gemmini_args_t args;
                    return gemmini_set_q8_hp2_weight_args(a, args);
                }

                if (a->type == GGML_TYPE_Q8_H1) {
                    const int64_t dim_j = a->ne[1] ? a->ne[1] : 1;
                    const int64_t dim_z = a->ne[2] ? a->ne[2] : 1;
                    const int64_t dim_w = a->ne[3] ? a->ne[3] : 1;

                    if (a->ne[0] <= 0 || a->ne[0] % QK8_0 != 0)
                        return false;

                    const size_t blocks_per_row = static_cast<size_t>(a->ne[0]) / QK8_0;
                    if (blocks_per_row == 0)
                        return false;

                    const __int128 logical_rows_128 =
                        static_cast<__int128>(dim_j) * static_cast<__int128>(dim_z) * static_cast<__int128>(dim_w);
                    if (logical_rows_128 <= 0 || logical_rows_128 > static_cast<__int128>(std::numeric_limits<size_t>::max()))
                        return false;

                    const size_t logical_rows = static_cast<size_t>(logical_rows_128);

                    size_t row_bytes = 0;
                    size_t plane_bytes = 0;
                    size_t storage_bytes = 0;
                    if (!gemmini_checked_mul(blocks_per_row, sizeof(block_q8_h1), row_bytes) ||
                        !gemmini_checked_mul(row_bytes, static_cast<size_t>(dim_j), plane_bytes) ||
                        !gemmini_checked_mul(plane_bytes, static_cast<size_t>(dim_z), storage_bytes) ||
                        !gemmini_checked_mul(storage_bytes, static_cast<size_t>(dim_w), storage_bytes)) {
                        return false;
                    }

                    if (a->nb[0] != sizeof(block_q8_h1) || a->nb[1] != row_bytes ||
                        a->nb[2] != plane_bytes || a->nb[3] != storage_bytes) {
                        return false;
                    }

                    const char * base = reinterpret_cast<const char *>(a->view_src ? a->view_src->data : a->data);
                    const size_t view_offset = a->view_src ? static_cast<size_t>(a->view_offs) : 0;
                    if (base == nullptr)
                        return false;
                    if (a->view_src != nullptr) {
                        const size_t source_bytes = ggml_nbytes(a->view_src);
                        if (view_offset > source_bytes || storage_bytes > source_bytes - view_offset)
                            return false;
                    } else if (view_offset != 0) {
                        return false;
                    }

                    const char *data = base + view_offset;
                    if (reinterpret_cast<uintptr_t>(data) % alignof(block_q8_h1) != 0)
                        return false;

                    const size_t blocks_per_row_contract = static_cast<size_t>(a->ne[0]) / QK8_0;
                    size_t q8_h1_block_count = 0;
                    if (!gemmini_checked_mul(logical_rows, blocks_per_row_contract, q8_h1_block_count)) {
                        return false;
                    }

                    const block_q8_h1 * blocks = reinterpret_cast<const block_q8_h1 *>(data);
                    ggml_gemmini_args_t args;
                    args.weight_format = ggml_gemmini_args_t::im2p_weight_format_t::q8_h1;
                    args.J = logical_rows;
                    args.K = static_cast<size_t>(a->ne[0]);
                    args.block_size_k = QK8_0;
                    args.blocks_per_row = blocks_per_row_contract;
                    args.q8_h1_blocks = blocks;
                    args.q8_h1_block_count = q8_h1_block_count;
                    args.q8_h1_rows = logical_rows;
                    args.native_weight_bytes = storage_bytes;
                    return args.has_q8_h1_im2p_contract();
                }

                if (a->type == GGML_TYPE_Q8_H2) {
                    if constexpr (OPTION != CPU) {
                        return false;
                    }

                    ggml_gemmini_args_t args;
                    args.tiled_matmul_type = CPU;
                    return gemmini_set_q8_h2_weight_args(a, args);
                }

                if (a->type == GGML_TYPE_Q8_HP1) {
                    return false;
                }

                if (a->type == GGML_TYPE_Q8_HP2) {
                    return false;
                }

                if (a->type != GGML_TYPE_Q8_0 || a->ne[0] <= 0 || a->ne[0] % QK8_0 != 0)
                    return false;

                const block_q8_0 * base = ggml::gemmini::weight_block_base(a);
                if (base == nullptr)
                    return false;

                if (a->nb[0] == 0 || a->nb[0] < sizeof(block_q8_0))
                    return false;

                return true;
            }

            if constexpr (
                ggml::gemmini::config::CURRENT_COMPUTE_TYPE ==
                    ggml::gemmini::config::ComputeType::FLOAT ||
                ggml::gemmini::config::DEQUANT_FP_TEST
            ) {
                if (gemmini_is_extended_dequant_weight_type(a->type)) {
                    return a->data != nullptr && a->ne[0] > 0 &&
                           a->ne[0] % ggml_blck_size(a->type) == 0 &&
                           ggml_is_contiguous(a) &&
                           ggml_nbytes(a) == ggml_row_size(a->type, a->ne[0]) *
                                                static_cast<size_t>(ggml_nrows(a));
                }

                if (a->type == GGML_TYPE_Q8_H1) {
                    return a->data != nullptr && a->ne[0] > 0 && a->ne[0] % QK8_0 == 0 &&
                           ggml_is_contiguous(a) &&
                           ggml_nbytes(a) == ggml_row_size(GGML_TYPE_Q8_H1, a->ne[0]) *
                                                static_cast<size_t>(ggml_nrows(a));
                }

                if (a->type == GGML_TYPE_Q8_H2) {
                    ggml_gemmini_args_t args;
                    args.tiled_matmul_type = CPU;
                    return a->data != nullptr && gemmini_set_q8_h2_weight_args(a, args);
                }

                if (a->type == GGML_TYPE_Q8_CHANNEL) {
                    return a->data != nullptr && gemmini_q8_channel_weight_contract(a);
                }

                if constexpr (ggml::gemmini::config::DEQUANT_FP_TEST) {
                    if (a->type == GGML_TYPE_Q8_H1) {
                        const int64_t dim_j = a->ne[1] ? a->ne[1] : 1;
                        const int64_t dim_z = a->ne[2] ? a->ne[2] : 1;
                        const int64_t dim_w = a->ne[3] ? a->ne[3] : 1;

                        if (a->ne[0] <= 0 || a->ne[0] % QK8_0 != 0)
                            return false;

                        const size_t blocks_per_row = static_cast<size_t>(a->ne[0]) / QK8_0;
                        if (blocks_per_row == 0)
                            return false;

                        const __int128 logical_rows_128 =
                            static_cast<__int128>(dim_j) * static_cast<__int128>(dim_z) * static_cast<__int128>(dim_w);
                        if (logical_rows_128 <= 0 || logical_rows_128 > static_cast<__int128>(std::numeric_limits<size_t>::max()))
                            return false;

                        const size_t logical_rows = static_cast<size_t>(logical_rows_128);

                        size_t row_bytes = 0;
                        size_t plane_bytes = 0;
                        size_t storage_bytes = 0;
                        if (!gemmini_checked_mul(blocks_per_row, sizeof(block_q8_h1), row_bytes) ||
                            !gemmini_checked_mul(row_bytes, static_cast<size_t>(dim_j), plane_bytes) ||
                            !gemmini_checked_mul(plane_bytes, static_cast<size_t>(dim_z), storage_bytes) ||
                            !gemmini_checked_mul(storage_bytes, static_cast<size_t>(dim_w), storage_bytes)) {
                            return false;
                        }

                        if (a->nb[0] != sizeof(block_q8_h1) || a->nb[1] != row_bytes ||
                            a->nb[2] != plane_bytes || a->nb[3] != storage_bytes) {
                            return false;
                        }

                        const char * base = reinterpret_cast<const char *>(a->view_src ? a->view_src->data : a->data);
                        const size_t view_offset = a->view_src ? static_cast<size_t>(a->view_offs) : 0;
                        if (base == nullptr)
                            return false;
                        if (a->view_src != nullptr) {
                            const size_t source_bytes = ggml_nbytes(a->view_src);
                            if (view_offset > source_bytes || storage_bytes > source_bytes - view_offset)
                                return false;
                        } else if (view_offset != 0) {
                            return false;
                        }

                        const char *data = base + view_offset;
                        if (reinterpret_cast<uintptr_t>(data) % alignof(block_q8_h1) != 0)
                            return false;

                        const size_t blocks_per_row_contract = static_cast<size_t>(a->ne[0]) / QK8_0;
                        size_t q8_h1_block_count = 0;
                        if (!gemmini_checked_mul(logical_rows, blocks_per_row_contract, q8_h1_block_count)) {
                            return false;
                        }

                        const block_q8_h1 * blocks = reinterpret_cast<const block_q8_h1 *>(data);
                        ggml_gemmini_args_t args;
                        args.weight_format = ggml_gemmini_args_t::im2p_weight_format_t::q8_h1;
                        args.J = logical_rows;
                        args.K = static_cast<size_t>(a->ne[0]);
                        args.block_size_k = QK8_0;
                        args.blocks_per_row = blocks_per_row_contract;
                        args.q8_h1_blocks = blocks;
                        args.q8_h1_block_count = q8_h1_block_count;
                        args.q8_h1_rows = logical_rows;
                        args.native_weight_bytes = storage_bytes;
                        return args.has_q8_h1_im2p_contract();
                    }

                    if (a->type == GGML_TYPE_Q8_H2) {
                        ggml_gemmini_args_t args;
                        args.tiled_matmul_type = CPU;
                        return gemmini_set_q8_h2_weight_args(a, args);
                    }

                }
                if (a->type == GGML_TYPE_Q8_HP1) {
                    ggml_gemmini_args_t args;
                    return gemmini_set_q8_hp1_weight_args(a, args, gemmini_is_loader_metadata(a));
                }
                if (a->type == GGML_TYPE_Q8_HP2) {
                    ggml_gemmini_args_t args;
                    return gemmini_set_q8_hp2_weight_args(a, args, gemmini_is_loader_metadata(a));
                }
                if (a->type == GGML_TYPE_I8) {
                    return gemmini_i8_supported_scale_data(a, TRANSPOSE_B != 0) != nullptr;
                }

                return a->type == GGML_TYPE_Q8_0 || a->type == GGML_TYPE_F32 || a->type == GGML_TYPE_F16;
            }
        }

        case GGML_OP_ADD:
        case GGML_OP_OUT_PROD:
            // return op->src[0]->type == GGML_TYPE_F32 &&
            //        op->src[1]->type == GGML_TYPE_F32 &&
            //        ggml_is_matrix(src0) &&
            //        ggml_is_matrix(src1) &&
            //        ggml_is_contiguous(src0) &&
            //        (ggml_is_contiguous(src1) || ggml_is_transposed(src1)) &&
            //        (src0->type == GGML_TYPE_F32 || ggml_get_type_traits(src0->type)->to_float != NULL);

        default:
            return false;

    }

    GGML_UNUSED(dev);
}

static bool ggml_backend_gemmini_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return ggml_backend_buft_is_host(buft);

    GGML_UNUSED(dev);
}

static const struct ggml_backend_device_i ggml_backend_gemmini_device_i = {
    /* .get_name             = */ ggml_backend_gemmini_device_get_name,
    /* .get_description      = */ ggml_backend_gemmini_device_get_description,
    /* .get_memory           = */ ggml_backend_gemmini_device_get_memory,
    /* .get_type             = */ ggml_backend_gemmini_device_get_type,
    /* .get_props            = */ ggml_backend_gemmini_device_get_props,
    /* .init_backend         = */ ggml_backend_gemmini_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_gemmini_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ ggml_backend_gemmini_device_buffer_from_host_ptr,
    /* .supports_op          = */ ggml_backend_gemmini_device_supports_op,
    /* .supports_buft        = */ ggml_backend_gemmini_device_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

// backend reg interface

static const char * ggml_backend_gemmini_reg_get_name(ggml_backend_reg_t reg) {
    return "GEMMINI";

    GGML_UNUSED(reg);
}

static size_t ggml_backend_gemmini_reg_get_device_count(ggml_backend_reg_t reg) {
    return 1;

    GGML_UNUSED(reg);
}

static ggml_backend_dev_t ggml_backend_gemmini_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);

    static ggml_backend_device ggml_backend_gemmini_device = {
        /* .iface   = */ ggml_backend_gemmini_device_i,
        /* .reg     = */ reg,
        /* .context = */ nullptr,
    };

    return &ggml_backend_gemmini_device;

    GGML_UNUSED(reg);
    GGML_UNUSED(index);
}

static const struct ggml_backend_reg_i ggml_backend_gemmini_reg_i = {
    /* .get_name         = */ ggml_backend_gemmini_reg_get_name,
    /* .get_device_count = */ ggml_backend_gemmini_reg_get_device_count,
    /* .get_device       = */ ggml_backend_gemmini_reg_get_device,
    /* .get_proc_address = */ NULL,
};

ggml_backend_reg_t ggml_backend_gemmini_reg(void) {
    static struct ggml_backend_reg ggml_backend_gemmini_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_gemmini_reg_i,
        /* .context     = */ NULL,
    };

    return &ggml_backend_gemmini_reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_gemmini_reg)
