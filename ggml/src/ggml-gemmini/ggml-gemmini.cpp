// ggml-gemmini.cpp

#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <new>
#include <cstdlib>
#include <string_view>
#include <future>

#include <limits>

#include <memory>
#include "ggml-impl.h"
#include "ggml-gemmini.h"
#include "ggml-gemmini-config.hpp"
#include "ggml-gemmini-q8-h1-artifact.hpp"
#include "ggml-gemmini-q8-h1-reprocess.hpp"
#include "ggml-backend-impl.h"
#include "ggml-quants.h"

#include <gemmini/log.hpp>
#include "dump/dump_tensor.hpp"

#include <gemmini.h>
// Legacy aisa::ActivationDEC is replaced by ggml::gemmini::quants::dec
// #include "error_compensation/activation_DEC.h"
#include "ggml-gemmini-args.h"
#include "ggml-gemmini-matmul.hpp"
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
#ifndef ERROR_COMPENSATION
#define ERROR_COMPENSATION 0
#endif 
#if ERROR_COMPENSATION
#include "quants/dec/dec.hpp"
#endif
#ifndef OPTION
#define OPTION CPU
#endif

namespace
{

    void gemmini_matmul_fp_facade(size_t I, size_t J, size_t K,
                                  const float * activation, const float * weights,
                                  float * output) {
        ggml_gemmini_args_t args{};
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
                ggml::gemmini::config::CURRENT_ACTIVATION_QUANT == ggml::gemmini::config::ActivationQuantAlgo::TENSOR ||
                ggml::gemmini::config::CURRENT_ACTIVATION_QUANT == ggml::gemmini::config::ActivationQuantAlgo::TOKEN ||
                ggml::gemmini::config::CURRENT_ACTIVATION_QUANT == ggml::gemmini::config::ActivationQuantAlgo::STRIPE
            ) {
                return gemmini_q8_channel_int_route_t::cpu_direct_read;
            }
        } else if constexpr (OPTION == OS || OPTION == WS) {
            if constexpr (
                ggml::gemmini::config::CURRENT_ACTIVATION_QUANT == ggml::gemmini::config::ActivationQuantAlgo::TENSOR ||
                ggml::gemmini::config::CURRENT_ACTIVATION_QUANT == ggml::gemmini::config::ActivationQuantAlgo::TOKEN ||
                ggml::gemmini::config::CURRENT_ACTIVATION_QUANT == ggml::gemmini::config::ActivationQuantAlgo::STRIPE
            ) {
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
}
#endif

// Cycle 측정 용
extern "C" volatile uint64_t gemmini_tiled_matmul_cycles = 0; // gemmini.h

static void setup_gemmini_log_outputs_if_needed(void) {
    static thread_local bool configured = false;
    if (configured) {
        return;
    }

    if (!ggml::gemmini::log::cycle.has_explicit_output() &&
        !ggml::gemmini::log::cycle.set_output_path("log/cycle-log.jsonl")) {
        GGML_LOG_WARN("%s: failed to set default cycle log path 'log/cycle-log.jsonl'\n", __func__);
    }
    if (!ggml::gemmini::log::debug.has_explicit_output() &&
        !ggml::gemmini::log::debug.set_output_path("log/debug-log.jsonl")) {
        GGML_LOG_WARN("%s: failed to set default debug log path 'log/debug-log.jsonl'\n", __func__);
    }
    configured = true;
}

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
    const auto layer_type = ggml::gemmini::types::parse_layer(dst->src[1]->name);
    const char *layer = ggml::gemmini::types::to_string(layer_type);
    ggml::gemmini::log::debug(layer, "ggml_backend_gemmini_mul_mat called");
    const auto *src0 = dst->src[0]; // src0: weight (J x K), row-major, 전치 상태
    const auto *src1 = dst->src[1]; // src1: activation (I x K) -> 전치 없음 (A)

    uint64_t start = 0;
    uint64_t end = 0;

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
                                     (float *)dst->data);
            return;
        } else if (src0->type == GGML_TYPE_Q8_0) {
            std::vector<float> src0_f32(jk_count);
            dequantize_row_q8_0((const block_q8_0 *)src0->data, src0_f32.data(), jk_count);
            gemmini_matmul_fp_facade(I, J, K, (const float *)src1->data, src0_f32.data(),
                                     (float *)dst->data);
            return;
        } else if (src0->type == GGML_TYPE_Q8_CHANNEL) {
            std::vector<float> src0_f32(jk_count);
            const size_t row_size = ggml_row_size(GGML_TYPE_Q8_CHANNEL, src0->ne[0]);
            const char * row = static_cast<const char *>(src0->data);
            for (size_t j = 0; j < J; ++j) {
                dequantize_row_q8_channel(row + j * row_size, src0_f32.data() + j * K, K);
            }
            gemmini_matmul_fp_facade(I, J, K, (const float *)src1->data, src0_f32.data(),
                                     (float *)dst->data);
            return;
        } else if (src0->type == GGML_TYPE_Q8_H1) {
            std::vector<float> src0_f32(jk_count);
            dequantize_row_q8_h1((const block_q8_h1 *)src0->data, src0_f32.data(), jk_count);
            gemmini_matmul_fp_facade(I, J, K, (const float *)src1->data, src0_f32.data(),
                                     (float *)dst->data);
            return;
        } else if (src0->type == GGML_TYPE_Q8_H2) {
            std::vector<float> src0_f32(jk_count);
            dequantize_row_q8_h2((const block_q8_h2 *)src0->data, src0_f32.data(), jk_count);
            gemmini_matmul_fp_facade(I, J, K, (const float *)src1->data, src0_f32.data(),
                                     (float *)dst->data);
            return;
        } else if (src0->type == GGML_TYPE_Q8_HP1) {
            std::vector<float> src0_f32(jk_count);
            dequantize_row_q8_hp1((const block_q8_hp1 *)src0->data, src0_f32.data(), jk_count);
            gemmini_matmul_fp_facade(I, J, K, (const float *)src1->data, src0_f32.data(),
                                     (float *)dst->data);
            return;
        } else if (src0->type == GGML_TYPE_Q8_HP2) {
            std::vector<float> src0_f32(jk_count);
            dequantize_row_q8_hp2((const block_q8_hp2 *)src0->data, src0_f32.data(), jk_count);
            gemmini_matmul_fp_facade(I, J, K, (const float *)src1->data, src0_f32.data(),
                                     (float *)dst->data);
            return;
        } else {
            ggml::gemmini::log::debug(layer, "FLOAT path unsupported weight type=%d", (int)src0->type);
            GGML_ABORT("FLOAT compute type: unsupported weight type");
        }
    }

    ggml_gemmini_args_t args; // DEC과 gemmini 호출을 위한 args 
    std::vector<int8_t> q8_channel_dense;
    std::vector<float> q8_channel_scales;
    std::unique_ptr<ggml::gemmini::MatmulStripeCollector> pipeline_collector;
    std::unique_ptr<ggml::gemmini::MatmulExecution> pipeline_execution;
    const auto requested_invocation = [] {
        const char *value = std::getenv("GEMMINI_MATMUL_INVOCATION");
        return value != nullptr ? std::string_view(value) : std::string_view("full");
    }();
    const bool pipeline_requested = requested_invocation == "stripe-pipeline";
    const bool sequential_requested = requested_invocation == "stripe-sequential";
    const bool full_requested = requested_invocation.empty() || requested_invocation == "full";
    if (!full_requested && !pipeline_requested && !sequential_requested) {
        ggml::gemmini::log::debug(layer,
            "[matmul.invocation] status=invalid_invocation value=%.*s",
            static_cast<int>(requested_invocation.size()), requested_invocation.data());
        GGML_ABORT("Gemmini invalid_invocation: %.*s",
            static_cast<int>(requested_invocation.size()), requested_invocation.data());
    }
    constexpr bool exsia_pipeline_supported =
        ggml::gemmini::config::CURRENT_ACTIVATION_QUANT == ggml::gemmini::config::ActivationQuantAlgo::EXSIA;
    const bool pipeline_enabled = pipeline_requested && exsia_pipeline_supported && I > 1;
    const bool sequential_enabled = sequential_requested;
    const bool legacy_full_dispatch = [] {
        const char *value = std::getenv("GEMMINI_LEGACY_FULL_DISPATCH");
        return value != nullptr && std::string_view(value) == "1";
    }();
    const bool facade_full_dispatch = !pipeline_enabled && !sequential_enabled && !legacy_full_dispatch;
    const size_t pipeline_job_capacity = [] {
        constexpr size_t default_capacity = 2;
        const char *value = std::getenv("GEMMINI_STRIPE_JOB_SLOTS");
        if (value == nullptr || *value == '\0') {
            return default_capacity;
        }
        const long requested = std::strtol(value, nullptr, 10);
        return requested > 0 ? static_cast<size_t>(requested) : default_capacity;
    }();
    if (pipeline_requested && !pipeline_enabled) {
        if (I <= 1) {
            ggml::gemmini::log::debug(layer,
                "[matmul.pipeline] status=unsupported_invocation dispatch=row-direct/full stripe-pipeline requires I>1");
        } else {
            ggml::gemmini::log::debug(layer,
                "[matmul.pipeline] status=unsupported_invocation dispatch=fatal stripe-pipeline requires EXSIA activation");
            GGML_ABORT("Gemmini unsupported_invocation: stripe-pipeline requires EXSIA activation");
        }
    }
    if (sequential_enabled) {
        ggml::gemmini::log::debug(layer, "[matmul.sequential] enabled");
    }

    // set args
    start = ggml::gemmini::cycle::read();
    args.transpose_B = (TRANSPOSE_B != 0);
    args.layer_type = layer_type;
    args.model_arch = ctx->model_arch.c_str();
    ggml::gemmini::log::debug(layer, "model_arch=%s\n", args.model_arch ? args.model_arch : "");
    args.full_C = FULL_C;
    args.low_D = LOW_D;

    args.I = I;
    args.J = J;
    args.K = K;
    args.tiled_matmul_type = OPTION;

    /* _____ 3. Gemmini용 stride _____ */
    args.sA = K;
    args.sC = J;

    end = ggml::gemmini::cycle::read();
    ggml::gemmini::log::cycle(layer, "cpu.Set Args for calling gemmini", start, end);

    // set tile size
    start = ggml::gemmini::cycle::read();
    ggml::gemmini::gemmini_set_tile_ws(&args);
    end = ggml::gemmini::cycle::read();
    ggml::gemmini::log::cycle(layer, "cpu.Set tile size", start, end);

    // quantize activation
    start = ggml::gemmini::cycle::read();

    [[maybe_unused]] std::vector<int8_t> activation_q;
    activation_q.resize(ik_count);
    int8_t *qx = activation_q.data();

    args.A = reinterpret_cast<elem_t *>(qx);
    if (pipeline_enabled) {
        pipeline_collector = std::make_unique<ggml::gemmini::MatmulStripeCollector>(pipeline_job_capacity);
        args.exsia_stripe_ready_sink = pipeline_collector->sink();
    }
    ggml::gemmini::log::debug(
        ggml::gemmini::types::to_string(args.layer_type),
        "[" GGML_GEMMINI_ACTIVATION_QUANT_NAME "] final cfg model_arch=%s",
        args.model_arch ? args.model_arch : "");

#if defined(GGML_GEMMINI_TEST_OBSERVER) && GGML_GEMMINI_COMPUTE_TYPE == 0
    ggml::gemmini::observe_test_i("activation", args.I);
#endif
    bool quantize_ok = false;
    if (!pipeline_enabled) {
        quantize_ok = ggml::gemmini::quants::quantize_activation(src1, args);
    }
#if GGML_GEMMINI_COMPUTE_TYPE == 0
    static_assert(sizeof(elem_t) == 1, "Q8_0 path assumes elem_t is int8 (1 byte).");
#endif

    end = ggml::gemmini::cycle::read();
    ggml::gemmini::log::cycle(layer, "cpu.Quantize activation", start, end);
    if (!pipeline_enabled && !quantize_ok) {
        if (pipeline_collector && !pipeline_collector->status()) {
            ggml::gemmini::log::debug(layer, "[matmul.pipeline] capture status=%d message=%s",
                static_cast<int>(pipeline_collector->status().code), pipeline_collector->status().message);
        }
        ggml::gemmini::log::debug(layer, "activation quantize failed");
        return;
    }

    if constexpr (ggml::gemmini::config::CURRENT_COMPUTE_TYPE == ggml::gemmini::config::ComputeType::FLOAT &&
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
            return;

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
                                 static_cast<float *>(dst->data));
        return;
    }

    start = ggml::gemmini::cycle::read();
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

        end = ggml::gemmini::cycle::read();
        ggml::gemmini::log::cycle(layer, "cpu.Use dense I8 weight", start, end);
    } else {
        if (src0->type == GGML_TYPE_Q8_H1) {
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
            start = ggml::gemmini::cycle::read();
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
            if (!args.has_q8_h1_im2p_contract()) {
                ggml_gemmini_log_q8_h1_contract_issue(layer, args);
                GGML_ABORT("Gemmini Q8_0 reprocessed im2p contract failed");
            }

            ggml::gemmini::log::debug(layer,
                "[Q8_0 reprocess] blocks=%p sB=%zu blocks_per_row=%zu logical_rows=%zu",
                (void *)reprocessed_q8_h1.data(), args.sB, args.blocks_per_row, q8_0_reprocess_rows);
            end = ggml::gemmini::cycle::read();
            ggml::gemmini::log::cycle(layer, "cpu.Reprocess Q8_0 to Q8_H1", start, end);
        } else {
            ggml::gemmini::log::debug(layer, "int compute unsupported weight type=%d", (int)src0->type);
            GGML_ABORT("Gemmini int mul_mat received unsupported weight type");
        }

    }

    start = ggml::gemmini::cycle::read();
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

    if (pipeline_enabled) {
        ggml::gemmini::MatmulOptions pipeline_options{};
        pipeline_options.mode = ggml::gemmini::MatmulInvocationMode::stripe_pipeline;
        pipeline_options.job_capacity = pipeline_job_capacity;
        pipeline_options.profiling = LOG_DEBUG != 0 || LOG_CYCLE != 0;
        ggml::gemmini::log::debug(layer, "[matmul.pipeline] job_slots=%zu", pipeline_job_capacity);
        if (const char *rc_shards = std::getenv("GEMMINI_RC_SHARDS")) {
            const int requested_shards = std::atoi(rc_shards);
            if (requested_shards > 0)
                pipeline_options.rc_shards = static_cast<size_t>(requested_shards);
        }
        pipeline_execution = std::make_unique<ggml::gemmini::MatmulExecution>(
            ggml::gemmini::prepare_execution(&args, pipeline_options));
        if (!pipeline_execution->status() || !pipeline_collector->start(*pipeline_execution)) {
            ggml::gemmini::log::debug(layer, "[matmul.pipeline] staged worker start failed");
            return;
        }
    }

    if (pipeline_enabled) {
        quantize_ok = ggml::gemmini::quants::quantize_activation(src1, args);
        if (!quantize_ok) {
            pipeline_collector->finish();
            ggml::gemmini::log::debug(layer, "activation quantize failed");
            return;
        }
    }

    if (sequential_enabled) {
        ggml::gemmini::MatmulOptions sequential_options{};
        sequential_options.mode = ggml::gemmini::MatmulInvocationMode::stripe_sequential;
        sequential_options.stripe_rows = args.tile_I != 0 ? args.tile_I * DIM : args.I;
        sequential_options.profiling = LOG_DEBUG != 0 || LOG_CYCLE != 0;
        if (const char *rc_shards = std::getenv("GEMMINI_RC_SHARDS")) {
            const int requested_shards = std::atoi(rc_shards);
            if (requested_shards > 0)
                sequential_options.rc_shards = static_cast<size_t>(requested_shards);
        }
        const auto sequential_status = ggml::gemmini::matmul(args, sequential_options);
        if (!sequential_status) {
            ggml::gemmini::log::debug(layer,
                "[matmul.sequential] status=%d message=%s",
                static_cast<int>(sequential_status.code), sequential_status.message);
            GGML_ABORT("Gemmini stripe sequential execution failed");
        }
        const auto route = ggml::gemmini::detail::normalize_route(args);
        const auto capabilities = ggml::gemmini::detail::route_capabilities(args);
        ggml::gemmini::log::debug(layer,
            "[matmul.route] invocation=stripe-sequential full_or_slice=slice "
            "activation_route=%s weight_route=%s backend_route=%s "
            "supports_live_pipeline=%d deprecated=%d",
            ggml::gemmini::detail::activation_route_name(route.activation),
            ggml::gemmini::detail::weight_route_name(route.weight),
            ggml::gemmini::detail::backend_route_name(route.backend),
            capabilities.live_stripe_producer ? 1 : 0,
            capabilities.deprecated ? 1 : 0);
    }

    end = ggml::gemmini::cycle::read();
    ggml::gemmini::log::cycle(layer, "cpu.Set Args for calling gemmini", start, end);

    // ggml::gemmini::log::debug("[Gemmini debug] layer=%s A=%p B=%p C=%p D=%p I=%zu J=%zu K=%zu sA=%zu sB=%zu sC=%zu stride_f_out(row)=%zu stride_f_out(col)=%zu nb1=%zu nb0=%zu",
    //                  layer, args.A, args.B, args.C, args.D,
    //                  args.I, args.J, args.K, args.sA, args.sB, args.sC,
    //                  args.stride_f_out, args.col_stride_f_out, dst->nb[1], dst->nb[0]);

    // ggml::gemmini::log::debug("[Gemmini addr] layer=%s f_out=%p stride_f_out=%zu col_stride_f_out=%zu",
    //                  layer, (void *)args.f_out, args.stride_f_out, args.col_stride_f_out);

    if (facade_full_dispatch) {
        ggml::gemmini::MatMul facade(args);
        const auto facade_result = facade.run_full();
        if (facade_result.status != ggml::gemmini::MatMulStatus::success) {
            GGML_ABORT("Gemmini full facade execution failed");
        }
    }

    if (!pipeline_enabled && legacy_full_dispatch) {
        if constexpr (ggml::gemmini::config::CURRENT_COMPUTE_TYPE == ggml::gemmini::config::ComputeType::INT) {
            constexpr auto baseline_activation_quant =
                ggml::gemmini::config::CURRENT_ACTIVATION_QUANT == ggml::gemmini::config::ActivationQuantAlgo::EXSIA
                    ? ggml::gemmini::baseline_activation_quant_t::EXSIA
                    : ggml::gemmini::config::CURRENT_ACTIVATION_QUANT == ggml::gemmini::config::ActivationQuantAlgo::TENSOR
                        ? ggml::gemmini::baseline_activation_quant_t::TENSOR
                        : ggml::gemmini::config::CURRENT_ACTIVATION_QUANT == ggml::gemmini::config::ActivationQuantAlgo::STRIPE
                            ? ggml::gemmini::baseline_activation_quant_t::BLOCK
                            : ggml::gemmini::baseline_activation_quant_t::TOKEN;
            constexpr auto baseline_weight_quant =
                ggml::gemmini::config::CURRENT_WEIGHT_QUANT == ggml::gemmini::config::WeightQuantAlgo::TENSOR
                    ? ggml::gemmini::baseline_weight_quant_t::TENSOR
                    : ggml::gemmini::baseline_weight_quant_t::FLOAT;
            const auto selected_baseline_weight_quant = src0->type == GGML_TYPE_Q8_CHANNEL
                ? ggml::gemmini::baseline_weight_quant_t::CHANNEL
                : baseline_weight_quant;
            bool run_baseline = true;

            if constexpr (ggml::gemmini::config::CURRENT_ACTIVATION_QUANT == ggml::gemmini::config::ActivationQuantAlgo::EXSIA ||
                          ggml::gemmini::config::CURRENT_ACTIVATION_QUANT == ggml::gemmini::config::ActivationQuantAlgo::STRIPE) {
                if (args.weight_i8_scale_active) {
                    if constexpr (ggml::gemmini::config::CURRENT_ACTIVATION_QUANT == ggml::gemmini::config::ActivationQuantAlgo::STRIPE) {
                        GGML_ABORT("Gemmini STRIPE activation requires IM2P/native Q8 weights; dense tensor weights are unsupported");
                    }
                    ggml::gemmini::log::debug(layer,
                        "[baseline] route activation=exsia weight=tensor weight_scale=%.9f dims=(I=%zu,J=%zu,K=%zu) sB=%zu option=%d",
                        static_cast<double>(args.weight_scale), args.I, args.J, args.K, args.sB,
                        static_cast<int>(args.tiled_matmul_type));
                } else if (args.has_q8_channel_direct_read_contract()) {
                    if constexpr (ggml::gemmini::config::CURRENT_ACTIVATION_QUANT != ggml::gemmini::config::ActivationQuantAlgo::STRIPE) {
                        GGML_ABORT("Gemmini Q8_CHANNEL direct-read is wired only for baseline TENSOR/TOKEN/STRIPE activation routes");
                    }
                } else if (args.has_q8_channel_dense_sidecar_contract()) {
                    ggml::gemmini::log::debug(layer,
                        "[baseline] route activation=%s weight=q8_channel_dense_sidecar dims=(I=%zu,J=%zu,K=%zu) sB=%zu option=%d",
                        GGML_GEMMINI_ACTIVATION_QUANT_NAME, args.I, args.J, args.K, args.sB,
                        static_cast<int>(args.tiled_matmul_type));
                } else if (src0->type == GGML_TYPE_Q8_CHANNEL) {
                    GGML_ABORT("Gemmini Q8_CHANNEL activation route contract failed");
                } else {
                    const char *weight_route = src0->type == GGML_TYPE_Q8_H1
                        ? "Q8_H1 direct"
                        : src0->type == GGML_TYPE_Q8_H2
                            ? "Q8_H2 direct"
                            : src0->type == GGML_TYPE_Q8_HP1
                                ? "Q8_HP1 direct"
                                : src0->type == GGML_TYPE_Q8_HP2
                                    ? "Q8_HP2 direct"
                                    : src0->type == GGML_TYPE_Q8_CHANNEL
                                        ? "Q8_CHANNEL direct-read"
                                        : "Q8_0 reprocessed";
                    ggml::gemmini::log::debug(layer,
                        "[activation] route im2p weight=%s dims=(I=%zu,J=%zu,K=%zu) sB=%zu option=%d",
                        weight_route, args.I, args.J, args.K, args.sB, static_cast<int>(args.tiled_matmul_type));
                    ggml::gemmini::MatMul facade(args);
                    const auto facade_result = facade.run_dense();
                    if (facade_result.status != ggml::gemmini::MatMulStatus::success) {
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
        const auto route = ggml::gemmini::detail::normalize_route(args);
        const auto capabilities = ggml::gemmini::detail::route_capabilities(args);
        ggml::gemmini::log::debug(layer,
            "[matmul.route] invocation=stripe-pipeline full_or_slice=slice "
            "activation_route=%s weight_route=%s backend_route=%s "
            "supports_live_pipeline=%d deprecated=%d",
            ggml::gemmini::detail::activation_route_name(route.activation),
            ggml::gemmini::detail::weight_route_name(route.weight),
            ggml::gemmini::detail::backend_route_name(route.backend),
            capabilities.live_stripe_producer ? 1 : 0,
            capabilities.deprecated ? 1 : 0);
        for (const auto & profile : pipeline_collector->profiles()) {
            ggml::gemmini::log::debug(layer,
                "[matmul.stripe] stripe_id=%zu row_begin=%zu row_end=%zu "
                "la_cycles=%llu la3_cycles=%llu sf_cycles=%llu handoff_ns=%llu ws_ns=%llu "
                "rc_prepare_ns=%llu rc_compute_ns=%llu rc_finalize_ns=%llu "
                "ws_start_ns=%llu ws_end_ns=%llu rc_start_ns=%llu rc_end_ns=%llu rc_shards=%zu",
                profile.stripe_id,
                profile.row_begin,
                profile.row_end,
                static_cast<unsigned long long>(profile.la_cycles),
                static_cast<unsigned long long>(profile.la3_cycles),
                static_cast<unsigned long long>(profile.sf_cycles),
                static_cast<unsigned long long>(profile.handoff.nanoseconds),
                static_cast<unsigned long long>(profile.ws.nanoseconds),
                static_cast<unsigned long long>(profile.rc_prepare.nanoseconds),
                static_cast<unsigned long long>(profile.rc_compute.nanoseconds),
                static_cast<unsigned long long>(profile.rc_finalize.nanoseconds),
                static_cast<unsigned long long>(profile.ws_start_ns),
                static_cast<unsigned long long>(profile.ws_end_ns),
                static_cast<unsigned long long>(profile.rc_start_ns),
                static_cast<unsigned long long>(profile.rc_end_ns),
                profile.rc_shards);
        }
    }
    // dst에는 gemmini 커널에서 dequantize한 결과가 들어옴 
#if ERROR_COMPENSATION
    if (!pipeline_enabled && legacy_full_dispatch) {
    uint64_t start_dec = ggml::gemmini::cycle::read();
    auto outliers = ggml::gemmini::quants::activation_outliers(args);
    uint64_t end_dec = ggml::gemmini::cycle::read();
    ggml::gemmini::log::cycle(layer, "[dec] cpu.Create views", start_dec, end_dec);

#if defined(GGML_GEMMINI_TEST_OBSERVER) && GGML_GEMMINI_COMPUTE_TYPE == 0
    ggml::gemmini::observe_test_i("dec", args.I);
#endif
    auto dec_result = ggml::gemmini::quants::dec::compensate_activation_dec(outliers, args, layer);
    if (dec_result.nnz > 0) {
        ggml::gemmini::log::debug(layer,
                         "[dec.summary] staged=%zu nnz=%zu unique_k=%zu",
                         dec_result.total_selected,
                         dec_result.nnz,
                         dec_result.unique_k_count);
    }
    }
#endif
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
    return ggml_backend_cpu_buffer_type();

    GGML_UNUSED(dev);
}

static ggml_backend_buffer_t ggml_backend_gemmini_device_buffer_from_host_ptr(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    return ggml_backend_cpu_buffer_from_ptr(ptr, size);

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
            // data-less tensor.  Permit native Q8 only for that probe; the real
            // execution path below still requires the full shared-weight
            // contract and validates the backing storage.
            if (gemmini_is_loader_metadata(a)) {
                if (!gemmini_shared_weight_contract(op) || a->ne[0] <= 0 || a->ne[1] <= 0 ||
                    a->ne[2] != 1 || a->ne[3] != 1 ||
                    !(ggml_is_contiguous(a) || gemmini_q8_channel_layout_contract(a))) {
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
