// Derived from Gemmini SDK develop fcfcf5ce; see matmul/README.md and LICENSE.gemmini*.
#include "block-quantized.hpp"
#include "execution.hpp"
#include "reference/cpu.hpp"
#include "ggml-gemmini-args.h"
#include "gemmini.h"
#include "../../ggml-npu-common/profiling/include/npu/log.hpp"
#include "../../ggml-npu-common/profiling/include/npu/cycle_reader.hpp"
#include "../../ggml-npu-common/quantization/activation/dispatch.hpp"
#include "../../ggml-npu-common/quantization/common/dequant.hpp"
#include "../../ggml-npu-common/quantization/common/math.hpp"
#include <algorithm>
#include <climits>
#include <optional>
#if defined(GGML_GEMMINI_HAS_OPENMP)
#include <omp.h>
#endif
namespace ggml::gemmini {
namespace {
int resolve_env_threads(const char *name, int fallback) {
    if (const char *env = getenv(name))
    {
      char *end = nullptr;
      const long parsed = strtol(env, &end, 10);
      if (end != env && end && *end == '\0' && parsed > 0)
        return static_cast<int>(std::min(parsed, static_cast<long>(INT_MAX)));
    }

    return fallback;
  }
int resolve_block_matmul_threads(size_t tile_pair_count) {
    int block_threads = 1;

#if defined(GGML_GEMMINI_HAS_OPENMP)
    const size_t capped_count = std::min(tile_pair_count, static_cast<size_t>(INT_MAX));
    block_threads = std::min(std::max(1, static_cast<int>(capped_count)), omp_get_max_threads());
#endif

    block_threads = resolve_env_threads("GEMMINI_OPENMP_THREADS", block_threads);

    return std::max(1, block_threads);
  }
}
DenseMatmulStatus gemmini_block_quantized_matmul(
      struct ggml_gemmini_args_t *args,
      bool use_cpu_dot,
      const char *cycle_label) {
    if (args == NULL)
      return DenseMatmulStatus::invalid_contract;

    const char *layer = args->matmul_layer.c_str();
#if defined(GGML_GEMMINI_TEST_OBSERVER)
    if (test_observe_semantic_layer(TestSemanticLayerSite::physical_block_quantized, layer))
      return DenseMatmulStatus::success;
#endif
    // H0: GGUF Q8_0/Q4_0 blocks as stored; fp16 d is the floating block scale.
    // Q4_0 nibbles are decoded to int8 codes, so both share the native hot loop.
    const bool h0 = args->weight_format == ggml_gemmini_args_t::weight_format_t::q8_h0 ||
        args->weight_format == ggml_gemmini_args_t::weight_format_t::q4_h0;
    const bool native_q8 = h0 ||
        args->weight_format == ggml_gemmini_args_t::weight_format_t::q8_h2 ||
        args->weight_format == ggml_gemmini_args_t::weight_format_t::q8_hp1 ||
        args->weight_format == ggml_gemmini_args_t::weight_format_t::q8_hp2;
    // tile size와 block size 매칭을 통해, tiled별 연산 결과를 dequantize해서 llama.cpp의 output으로 전달
    uint64_t start = ggml::npu::cycle::read();

    if (args->A == nullptr || args->f_out == nullptr)
      return DenseMatmulStatus::invalid_contract;

    if (!native_q8 && (args->B == nullptr || args->c_b == nullptr))
      return DenseMatmulStatus::invalid_contract;

    if (!native_q8)
    {
      const size_t stripe_J_check = args->stripe_J_or_rowwise_elems();
      if (stripe_J_check > 1)
      {
      if (args->s_rf_stripe == nullptr || args->R_stripe == nullptr)
        return DenseMatmulStatus::invalid_contract;
      }
      else
      {
      if (args->s_rf == nullptr || args->R == nullptr)
        return DenseMatmulStatus::invalid_contract;
      }
    }

    if (args->tile_I == 0 || args->tile_J == 0 || args->tile_K == 0)
      gemmini_set_tile_ws(args);

    const size_t dim_I = args->I;
    const size_t dim_J = args->J;
    const size_t dim_K = args->K;
    // H0 row geometry lives where its contract validates it (as in the weight reader).
    const size_t blocks_per_row =
        args->weight_format == ggml_gemmini_args_t::weight_format_t::q4_h0 ? args->native_blocks_per_row
        : args->weight_format == ggml_gemmini_args_t::weight_format_t::q8_h0 ? args->blocks_K
                                                                                  : args->blocks_per_row;
    if (dim_I == 0 || dim_J == 0 || dim_K == 0 || blocks_per_row == 0)
      return DenseMatmulStatus::invalid_contract;

    GGML_ASSERT(args->act == GEMMINI_ACT_NONE);

    size_t block_size_k = args->block_size_k > 0
                              ? static_cast<size_t>(args->block_size_k)
                              : static_cast<size_t>(QK8_0);
    if (args->weight_format == ggml_gemmini_args_t::weight_format_t::q8_h2)
      block_size_k = static_cast<size_t>(QK8_H2);
    else if (args->weight_format == ggml_gemmini_args_t::weight_format_t::q8_hp1 ||
             args->weight_format == ggml_gemmini_args_t::weight_format_t::q8_hp2)
      block_size_k = static_cast<size_t>(QK8_HP);
    if (block_size_k == 0)
      return DenseMatmulStatus::invalid_contract;

    const size_t tile_I = args->tile_I;
    const size_t tile_J = args->tile_J;
    const size_t tile_K = args->tile_K;
    const size_t out_row_stride = args->stride_f_out ? args->stride_f_out : dim_J;
    const size_t out_col_stride = args->col_stride_f_out ? args->col_stride_f_out : 1;
    const size_t scale_rows = args->blocks_J > 0 ? args->blocks_J : dim_J;
    const size_t stripe_J = args->stripe_J_or_rowwise_elems();
    const bool use_stripe_scales = (stripe_J > 1) && (args->s_rf_stripe != nullptr) && (args->R_stripe != nullptr);
    const size_t num_i_tiles = (dim_I + tile_I * DIM - 1) / (tile_I * DIM);
    const size_t num_j_tiles = (dim_J + tile_J * DIM - 1) / (tile_J * DIM);
    const size_t num_tile_ks = (dim_K + tile_K * DIM - 1) / (tile_K * DIM);
    const auto *activation_meta = std::get_if<ggml::npu::quants::act::exsia::Meta>(&args->act_quant.storage());
    const auto *tensor_meta = std::get_if<ggml::npu::quants::act::tensor::Meta>(&args->act_quant.storage());
    const auto *token_meta = std::get_if<ggml::npu::quants::act::token::Meta>(&args->act_quant.storage());
    const auto *stripe_meta = std::get_if<ggml::npu::quants::act::stripe::Meta>(&args->act_quant.storage());

    if (stripe_meta != nullptr)
    {
      const size_t rows_per_stripe = tile_I * DIM;
      const size_t expected_scale_count = (dim_I + rows_per_stripe - 1) / rows_per_stripe;
      if (stripe_meta->scales.size() != expected_scale_count)
      {
        gemmini_log_debug_layer(layer,
            "[tiled_block_matmul_auto] reject STRIPE activation scale cardinality: got=%zu expected=%zu",
            stripe_meta->scales.size(), expected_scale_count);
        return DenseMatmulStatus::invalid_contract;
      }
      for (const float scale : stripe_meta->scales)
      {
        if (!std::isfinite(scale) || scale <= 0.0f)
        {
          gemmini_log_debug_layer(layer,
              "[tiled_block_matmul_auto] reject STRIPE activation scale value: scale=%g",
              static_cast<double>(scale));
          return DenseMatmulStatus::invalid_contract;
        }
      }
    }

    if (native_q8)
    {
      bool valid_format = false;
      switch (args->weight_format) {
        case ggml_gemmini_args_t::weight_format_t::q8_h2:
          valid_format = args->has_q8_h2_contract();
          break;
        case ggml_gemmini_args_t::weight_format_t::q8_hp1:
          valid_format = args->has_q8_hp1_contract();
          break;
        case ggml_gemmini_args_t::weight_format_t::q8_hp2:
          valid_format = args->has_q8_hp2_contract();
          break;
        case ggml_gemmini_args_t::weight_format_t::q8_h0:
          valid_format = args->has_q8_h0_contract();
          break;
        case ggml_gemmini_args_t::weight_format_t::q4_h0:
          valid_format = args->has_native_matched_width_contract();
          break;
        default:
          valid_format = false;
          break;
      }
      if (!valid_format)
      {
        gemmini_log_debug_layer(layer, "[tiled_block_matmul_auto] reject native Q8 contract: format=%d", static_cast<int>(args->weight_format));
        return DenseMatmulStatus::invalid_contract;
      }
    }

    if (activation_meta != nullptr)
    {
      const int16_t invalid_theta = std::numeric_limits<int16_t>::min();
      for (size_t tile_i_idx = 0; tile_i_idx < num_i_tiles; ++tile_i_idx)
      {
        if (activation_meta->resolve_stripe_theta(static_cast<int>(tile_i_idx)) == invalid_theta)
          return DenseMatmulStatus::invalid_contract;
      }
    }

    // BLOCK activation: one fp32 scale per (row, 32-K block), applied to every
    // native block dot before accumulation (block x block, then dequantize).
    std::optional<ggml::npu::quants::act::ActivationMetadataView> block_scales;
    if (native_q8 &&
        std::holds_alternative<ggml::npu::quants::act::block::Meta>(args->act_quant.storage()))
    {
      block_scales.emplace(*args, args->activation_row_offset,
                           args->activation_row_offset + dim_I);
      if (block_size_k != ggml::npu::quants::act::block::kGroupSize || !block_scales->valid())
        return DenseMatmulStatus::invalid_contract;
    }

    for (size_t i = 0; i < dim_I; ++i)
    {
      float *row_out = args->f_out + i * out_row_stride;
      for (size_t j = 0; j < dim_J; ++j)
        row_out[j * out_col_stride] = 0.0f;
    }

    if (use_stripe_scales && !args->stripe_mode_matches_tile_j(tile_J * DIM))
    {
      gemmini_log_debug_layer(layer,
          "[tiled_block_matmul_auto] reject stripe mode contract: stripe_J=%zu tile_J_elems=%zu",
          args->stripe_J, tile_J * DIM);
      return DenseMatmulStatus::invalid_contract;
    }

    gemmini_log_debug_layer(layer,
        "[tiled_block_matmul_auto] dim=(I=%zu,J=%zu,K=%zu) tile_size: A=(I=%zu x K=%zu) B=(J=%zu x K=%zu) | num_tiles: I=%zu J=%zu K=%zu blocks_per_row=%zu stripe_J=%zu",
        dim_I, dim_J, dim_K, tile_I * DIM, tile_K * DIM, tile_J * DIM, tile_K * DIM, num_i_tiles, num_j_tiles, num_tile_ks, blocks_per_row, stripe_J);

    const elem_t *A = args->A;
    const elem_t *B = args->B;
    const size_t stride_A = args->sA ? args->sA : dim_K;
    const size_t row_stride_B = args->sB;
    if (!native_q8 && row_stride_B == 0)
      return DenseMatmulStatus::invalid_contract;

    const int16_t activation_e_s = activation_meta ? activation_meta->e_s : 0;
    const int16_t activation_rho = activation_meta ? activation_meta->rho : 0;
    const size_t tile_pair_count = num_i_tiles * num_j_tiles;

#if defined(GGML_GEMMINI_HAS_OPENMP)
    const bool use_block_openmp = use_cpu_dot && tile_pair_count > 1;
    const int block_threads = use_block_openmp ? resolve_block_matmul_threads(tile_pair_count) : 1;
    if (use_block_openmp)
      gemmini_log_debug_layer(layer,
          "[gemmini_block_quantized_matmul] openmp tile_pairs=%zu threads=%d",
          tile_pair_count, block_threads);
#else
    const bool use_block_openmp = false;
#endif

    auto run_tile_pair = [&](size_t tile_i_idx, size_t tile_j_idx) {
      static thread_local std::vector<elem_t> a_tile;
      static thread_local std::vector<int32_t> a_tile_i32;
      static thread_local std::vector<int64_t> acc32;
      static thread_local std::vector<double> acc_fp;
      static thread_local std::vector<int8_t> q4_h0_codes;  // one decoded Q4_0 block per column

      const size_t tile_i = tile_i_idx * tile_I * DIM;
      const size_t tile_j = tile_j_idx * tile_J * DIM;
      const size_t tile_i_actual = std::min(tile_I * DIM, dim_I - tile_i);
      const size_t tile_j_actual = std::min(tile_J * DIM, dim_J - tile_j);
      const int tile_row_idx = static_cast<int>(tile_i_idx);
      const size_t stripe_col_idx = tile_j_idx;

      if (!use_block_openmp)
      {
        gemmini_log_debug_layer(layer,
            "[preload] weight stripe: tile_j=%zu tile_j_actual=%zu stripe_col=%zu stripe_J=%zu use_stripe=%s",
            tile_j, tile_j_actual, stripe_col_idx, stripe_J,
            use_stripe_scales ? "yes" : "no");
      }

      if (native_q8)
        acc_fp.assign(tile_i_actual * tile_j_actual, 0.0);
      else
        acc32.assign(tile_i_actual * tile_j_actual, 0);

      for (size_t tile_k = 0; tile_k < dim_K; tile_k += tile_K * DIM)
      {
        const size_t tile_k_actual = std::min(tile_K * DIM, dim_K - tile_k);
        const size_t tile_k_padded = ((tile_k_actual + block_size_k - 1) / block_size_k) * block_size_k;
        const size_t wavefront = static_cast<size_t>(tile_row_idx) + stripe_col_idx;
        if (!use_block_openmp)
        {
          gemmini_log_debug_layer(layer,
              "[wavefront] wave=%zu tile=(row=%d,col=%zu) stripe=(I=%zu,J=%zu)",
              wavefront, tile_row_idx, stripe_col_idx,
              tile_i_actual, tile_j_actual);
        }

        if (native_q8)
          a_tile_i32.assign(tile_i_actual * tile_k_padded, 0);
        else
          a_tile.assign(tile_i_actual * tile_k_padded, 0);

        for (size_t i = 0; i < tile_i_actual; ++i)
        {
          for (size_t kk = 0; kk < tile_k_actual; ++kk)
          {
            const size_t src_i = tile_i + i;
            const size_t src_k = tile_k + kk;
            int32_t a = static_cast<int32_t>(args->transpose_A
                                                 ? A[src_k * stride_A + src_i]
                                                 : A[src_i * stride_A + src_k]);
            if (native_q8)
              a_tile_i32[i * tile_k_padded + kk] = a;
            else
              a_tile[i * tile_k_padded + kk] = static_cast<elem_t>(a);
          }
        }
        if (!use_block_openmp)
        {
          gemmini_log_debug_layer(layer,
              "[prefetch] activation tile: tile_i=%zu tile_k=%zu size=(%zu x %zu)",
              tile_i, tile_k, tile_i_actual, tile_k_actual);
        }

        std::vector<double> native_scale_lut;        // (j,weight_blk)당 1회 계산, i-루프 밖으로 hoist
        std::vector<const int8_t *> native_qs_lut;   // qs 포인터도 hoist: 포맷 무관 단일 핫 루프
        for (size_t k_block = 0; k_block < tile_k_actual;)
        {
          const size_t global_k = tile_k + k_block;
          const size_t k_in_weight_block = global_k % block_size_k;
          const size_t block_k_actual = std::min(block_size_k - k_in_weight_block, tile_k_actual - k_block);
          const size_t weight_blk = global_k / block_size_k;
          if (weight_blk >= blocks_per_row)
          {
            k_block += block_k_actual;
            continue;
          }

          if (native_q8)
          {
            native_scale_lut.assign(tile_j_actual, 0.0);
            native_qs_lut.assign(tile_j_actual, nullptr);
            if (args->weight_format == ggml_gemmini_args_t::weight_format_t::q4_h0)
              q4_h0_codes.resize(tile_j_actual * QK4_0);
            for (size_t j = 0; j < tile_j_actual; ++j)
            {
              const size_t global_j = tile_j + j;
              if (global_j >= scale_rows) continue;
              switch (args->weight_format) {
                case ggml_gemmini_args_t::weight_format_t::q8_h2: {
                  const block_q8_h2 *b = args->q8_h2_block(global_j, weight_blk);
                  if (b) { native_scale_lut[j] = static_cast<double>(b->channel_scale) * static_cast<double>(b->m) / 255.0;
                      native_qs_lut[j] = b->qs; }
                  break; }
                case ggml_gemmini_args_t::weight_format_t::q8_hp1: {
                  const block_q8_hp1 *b = args->q8_hp1_block(global_j, weight_blk);
                  if (b) { native_scale_lut[j] = b->m == INT16_MIN ? 0.0 : static_cast<double>(npu_ldexp_fast_pos(b->channel_scale, b->m));
                      native_qs_lut[j] = b->qs; }
                  break; }
                case ggml_gemmini_args_t::weight_format_t::q8_hp2: {
                  const block_q8_hp2 *b = args->q8_hp2_block(global_j, weight_blk);
                  if (b) { native_scale_lut[j] = b->m == INT16_MIN ? 0.0 : static_cast<double>(npu_ldexp_fast_pos(b->channel_scale, b->m));
                      native_qs_lut[j] = b->qs; }
                  break; }
                case ggml_gemmini_args_t::weight_format_t::q8_h0: {
                  const block_q8_0 *b = args->B_blocks + global_j * blocks_per_row + weight_blk;
                  native_scale_lut[j] = static_cast<double>(ggml_fp16_to_fp32(b->d));
                  native_qs_lut[j] = b->qs;
                  break; }
                case ggml_gemmini_args_t::weight_format_t::q4_h0: {
                  // GGUF Q4_0 nibbles: low -> q[t], high -> q[t + 16], offset 8.
                  const block_q4_0 *b = args->q4_h0_blocks + global_j * blocks_per_row + weight_blk;
                  int8_t *codes = q4_h0_codes.data() + j * QK4_0;
                  for (size_t t = 0; t < QK4_0 / 2; ++t) {
                    codes[t] = static_cast<int8_t>((b->qs[t] & 0x0F) - 8);
                    codes[t + QK4_0 / 2] = static_cast<int8_t>((b->qs[t] >> 4) - 8);
                  }
                  native_scale_lut[j] = static_cast<double>(ggml_fp16_to_fp32(b->d));
                  native_qs_lut[j] = codes;
                  break; }
                default: break;
              }
            }
          }

          if (block_scales)
          {
            // Separate from the loop below so its non-BLOCK codegen (-ffast-math) stays as is.
            for (size_t i = 0; i < tile_i_actual; ++i)
            {
              float activation_block_scale = 1.0f;
              const bool scale_ok = block_scales->scale(tile_i + i, global_k, activation_block_scale);
              GGML_ASSERT(scale_ok);
              const int32_t *a_row_i32 = a_tile_i32.data() + i * tile_k_padded + k_block;
              for (size_t j = 0; j < tile_j_actual; ++j)
              {
                const int8_t *qs = native_qs_lut[j];
                if (tile_j + j >= scale_rows || qs == nullptr)
                  continue;

                int32_t block_dot = 0;
                for (size_t kk = 0; kk < block_k_actual; ++kk)
                  block_dot += a_row_i32[kk] * static_cast<int32_t>(qs[k_in_weight_block + kk]);
                acc_fp[i * tile_j_actual + j] += static_cast<double>(block_dot) *
                    (native_scale_lut[j] * static_cast<double>(activation_block_scale));
              }
            }
            k_block += block_k_actual;
            continue;
          }

          for (size_t i = 0; i < tile_i_actual; ++i)
          {
            const elem_t *a_row = native_q8 ? nullptr : a_tile.data() + i * tile_k_padded + k_block;
            const int32_t *a_row_i32 = native_q8 ? a_tile_i32.data() + i * tile_k_padded + k_block : nullptr;
            int64_t *acc_row = native_q8 ? nullptr : acc32.data() + i * tile_j_actual;

            for (size_t j = 0; j < tile_j_actual; ++j)
            {
              const size_t global_j = tile_j + j;
              if (global_j >= scale_rows)
                continue;

              const elem_t *b_block = native_q8 ? nullptr : B + global_j * row_stride_B + global_k;
              int32_t block_dot = 0;
              double native_block_sum = 0.0;
              if (native_q8)
              {
                const int8_t *qs = native_qs_lut[j];
                if (qs == nullptr)
                  continue;

                const double weight_scale = native_scale_lut[j];
                for (size_t kk = 0; kk < block_k_actual; ++kk)
                  block_dot += a_row_i32[kk] * static_cast<int32_t>(qs[k_in_weight_block + kk]);
                native_block_sum = static_cast<double>(block_dot) * weight_scale;

                acc_fp[i * tile_j_actual + j] += native_block_sum;
                continue;
              }

              if (use_cpu_dot)
              {
                acc_t block_dot_acc = 0;
                matmul_cpu_int32(false, false, 1, 1, block_k_actual,
                    a_row, b_block, nullptr, &block_dot_acc,
                    block_k_actual, 1, 0, 1,
                    1, 1, 1,
                    GEMMINI_ACT_NONE, 0, 0, false, true);
                block_dot = static_cast<int32_t>(block_dot_acc);
              }
              else
              {
                for (size_t kk = 0; kk < block_k_actual; ++kk)
                  block_dot += static_cast<int32_t>(a_row[kk]) * static_cast<int32_t>(b_block[kk]);
              }

              const int64_t acc_before = acc_row[j];
              const size_t scale_idx = global_j * args->blocks_per_row + weight_blk;
              const uint64_t c_eff = static_cast<uint64_t>(static_cast<uint16_t>(args->c_b[scale_idx])) +
                  (use_stripe_scales
                      ? static_cast<uint64_t>(args->R_stripe[global_j / stripe_J])
                      : static_cast<uint64_t>(args->R[global_j]));
              acc_row[j] += static_cast<int64_t>(block_dot) * static_cast<int64_t>(c_eff);

              if (!use_block_openmp && i == 0 && j == 0)
              {
                  gemmini_log_debug_layer(layer,
                      "[tiled_block_matmul_auto] block-dequant i=%zu j=%zu k=%zu weight_blk=%zu "
                    "dot=%d c_b=%u R=%u c_eff=%llu acc_before=%ld acc_after=%ld stripe=%s",
                    tile_i + i, global_j, tile_k + k_block, weight_blk,
                    block_dot,
                    static_cast<unsigned int>(args->c_b[scale_idx]),
                    use_stripe_scales ? static_cast<unsigned int>(args->R_stripe[global_j / stripe_J]) : static_cast<unsigned int>(args->R[global_j]),
                    static_cast<unsigned long long>(c_eff),
                    static_cast<long>(acc_before),
                    static_cast<long>(acc_row[j]),
                    use_stripe_scales ? "1" : "0");
              }
            }
          }

          k_block += block_k_actual;
        }

        if (!use_block_openmp)
        {
          gemmini_log_debug_layer(layer,
               "[tiled_block_matmul_auto] iter I=[%zu:%zu) J=[%zu:%zu) K=[%zu:%zu) K_pad=%zu blk=[%zu:%zu) pad=%zu e_t=%d rho=%d q=%d",
              tile_i, tile_i + tile_i_actual, tile_j, tile_j + tile_j_actual,
              tile_k, tile_k + tile_k_actual, tile_k_padded,
              tile_k / block_size_k, (tile_k + tile_k_actual + block_size_k - 1) / block_size_k,
              tile_k_padded - tile_k_actual,
              static_cast<int>(activation_e_s),
               static_cast<int>(activation_rho),
              0);
        }
      }

      const int16_t tile_activation_e_s = activation_meta ? activation_meta->resolve_stripe_theta(tile_row_idx) : 0;
      const float activation_scale = stripe_meta
          ? stripe_meta->scales[tile_i_idx]
          : (activation_meta
                ? apply_activation_exponent(1.0f, tile_activation_e_s, 0)
                : (tensor_meta ? tensor_meta->scale : 1.0f));
      const ggml::npu::quants::Stripe output_stripe(tile_i_actual, tile_j_actual, tile_i, tile_j);
      float *stripe_out = args->f_out + tile_i * out_row_stride + tile_j * out_col_stride;

      if (!native_q8 && !use_block_openmp && tile_i_actual > 0 && tile_j_actual > 0 && tile_j < scale_rows)
      {
        const float s_rf_val = use_stripe_scales ? args->s_rf_stripe[stripe_col_idx] : args->s_rf[tile_j];
        const float acc_fp = static_cast<float>(static_cast<double>(acc32[0]) * static_cast<double>(s_rf_val));
        const float contrib = acc_fp * activation_scale;
        gemmini_log_debug_layer(layer,
            "[tiled_block_matmul_auto] stripe-dequant i=%zu j=%zu "
            "acc=%ld s_rf=%.9f acc*s_rf=%.6f exp=%.6f contrib=%.6f -> f_out+=%.6f stripe=%s",
            tile_i, tile_j,
            static_cast<long>(acc32[0]),
            static_cast<double>(s_rf_val),
            static_cast<double>(acc_fp),
            static_cast<double>(activation_scale),
            static_cast<double>(contrib),
            static_cast<double>(stripe_out[0] + contrib),
            use_stripe_scales ? "1" : "0");
      }

      if (native_q8)
      {
        for (size_t i = 0; i < tile_i_actual; ++i)
        {
          float *row_out = stripe_out + i * out_row_stride;
          const double *acc_row = acc_fp.data() + i * tile_j_actual;
          float row_activation_scale = activation_scale;
          if (token_meta != nullptr && tile_i + i < token_meta->scales.size())
          {
            const float scale = token_meta->scales[tile_i + i];
            row_activation_scale = std::isfinite(scale) && scale > 0.0f ? scale : 1.0f;
          }
          for (size_t j = 0; j < tile_j_actual; ++j)
            row_out[j * out_col_stride] += static_cast<float>(acc_row[j] * static_cast<double>(row_activation_scale));
        }
      }
      else if (use_stripe_scales)
      {
        ggml::npu::update_q80_r_output_impl(
            output_stripe,
            acc32.data(),
            tile_j_actual,
            [s_rf_val = args->s_rf_stripe[stripe_col_idx]](size_t) { return s_rf_val; },
            activation_scale,
            stripe_out,
            out_row_stride,
            out_col_stride);
      }
      else
      {
        ggml::npu::update_q80_r_output_impl(
            output_stripe,
            acc32.data(),
            tile_j_actual,
            [s_rf = args->s_rf](size_t j) { return s_rf[j]; },
            activation_scale,
            stripe_out,
            out_row_stride,
            out_col_stride);
      }
    };

#if defined(GGML_GEMMINI_HAS_OPENMP)
    if (use_block_openmp)
    {
#pragma omp parallel for collapse(2) schedule(static) num_threads(block_threads)
      for (ptrdiff_t tile_i_idx = 0; tile_i_idx < static_cast<ptrdiff_t>(num_i_tiles); ++tile_i_idx)
      {
        for (ptrdiff_t tile_j_idx = 0; tile_j_idx < static_cast<ptrdiff_t>(num_j_tiles); ++tile_j_idx)
        {
          run_tile_pair(static_cast<size_t>(tile_i_idx), static_cast<size_t>(tile_j_idx));
        }
      }
    }
    else
#endif
    for (size_t tile_i_idx = 0; tile_i_idx < num_i_tiles; ++tile_i_idx)
    {
      for (size_t tile_j_idx = 0; tile_j_idx < num_j_tiles; ++tile_j_idx)
      {
        run_tile_pair(tile_i_idx, tile_j_idx);
      }
    }

    uint64_t end = ggml::npu::cycle::read();
    gemmini_log_cycle(layer, cycle_label, start, end);
    return DenseMatmulStatus::success;
  }

DenseMatmulStatus gemmini_block_quantized_matmul_auto(struct ggml_gemmini_args_t *args) {
    if (args == NULL)
      return DenseMatmulStatus::invalid_contract;

    const char *layer = args->matmul_layer.c_str();
#if defined(GGML_GEMMINI_TEST_OBSERVER)
    if (test_observe_semantic_layer(TestSemanticLayerSite::physical_block_quantized_auto, layer))
      return DenseMatmulStatus::success;
#endif
    const char *cpu_cycle_label = nullptr;
    const char *ws_cycle_label = nullptr;
    const char *os_message = nullptr;

    switch (args->weight_format)
    {
      case ggml_gemmini_args_t::weight_format_t::q8_h2:
        cpu_cycle_label = "[gemmini_block_quantized_matmul_auto] cpu.Q8_H2 tiled matmul";
        ws_cycle_label = "[gemmini_block_quantized_matmul] ws.Q8_H2 tiled matmul";
        os_message = "[gemmini_block_quantized_matmul_auto] Q8_H2 OS mode is unsupported";
        break;
      case ggml_gemmini_args_t::weight_format_t::q8_hp1:
        cpu_cycle_label = "[gemmini_block_quantized_matmul_auto] cpu.Q8_HP1 tiled matmul";
        ws_cycle_label = "[gemmini_block_quantized_matmul] ws.Q8_HP1 tiled matmul";
        os_message = "[gemmini_block_quantized_matmul_auto] Q8_HP1 OS mode is unsupported";
        break;
      case ggml_gemmini_args_t::weight_format_t::q8_hp2:
        cpu_cycle_label = "[gemmini_block_quantized_matmul_auto] cpu.Q8_HP2 tiled matmul";
        ws_cycle_label = "[gemmini_block_quantized_matmul] ws.Q8_HP2 tiled matmul";
        os_message = "[gemmini_block_quantized_matmul_auto] Q8_HP2 OS mode is unsupported";
        break;
      case ggml_gemmini_args_t::weight_format_t::q8_channel:
        if (!args->has_q8_channel_direct_read_contract()) {
          gemmini_log_debug_layer(layer,
              "[gemmini_block_quantized_matmul_auto] reject malformed Q8_CHANNEL direct-read contract");
          return DenseMatmulStatus::invalid_contract;
        } else {
          gemmini_log_debug_layer(layer,
              "[gemmini_block_quantized_matmul_auto] Q8_CHANNEL direct-read requires baseline/DEC software route; native hardware subroute would require repack");
          return DenseMatmulStatus::unsupported;
        }
      case ggml_gemmini_args_t::weight_format_t::q8_h0:
      case ggml_gemmini_args_t::weight_format_t::q4_h0:
        // A floating fp16 block scale has no integer factor for the WS array:
        // H0 runs block x block, then dequantize, on the CPU path only.
        if (args->tiled_matmul_type != CPU)
        {
          gemmini_log_debug_layer(layer,
              "[gemmini_block_quantized_matmul_auto] H0 weight format=%d is CPU-only",
              static_cast<int>(args->weight_format));
          return DenseMatmulStatus::unsupported;
        }
        return gemmini_block_quantized_matmul(args, true,
            args->weight_format == ggml_gemmini_args_t::weight_format_t::q8_h0
                ? "[gemmini_block_quantized_matmul_auto] cpu.Q8_0 tiled matmul"
                : "[gemmini_block_quantized_matmul_auto] cpu.Q4_0 tiled matmul");
      default:
        gemmini_log_debug_layer(layer,
            "[gemmini_block_quantized_matmul_auto] unsupported block weight format=%d",
            static_cast<int>(args->weight_format));
        return DenseMatmulStatus::unsupported;
    }

    if (args->tiled_matmul_type == CPU)
    {
      return gemmini_block_quantized_matmul(args, true, cpu_cycle_label);
    }
    if (args->tiled_matmul_type == WS)
    {
      return gemmini_block_quantized_matmul(args, false, ws_cycle_label);
    }

    gemmini_log_debug_layer(layer, os_message);
    return DenseMatmulStatus::unsupported;
  }
} // namespace ggml::gemmini
