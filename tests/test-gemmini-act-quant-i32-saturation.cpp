#include "../ggml/include/ggml.h"

#include "../ggml/src/ggml-gemmini/ggml-gemmini-args.h"
#include "../ggml/src/ggml-gemmini/residual/rmd/rmd-builder.hpp"
#include "../ggml/src/ggml-gemmini/residual/rmd/rmd-compose.hpp"
#include "../ggml/src/ggml-gemmini/quants/act/stripe/stripe.hpp"
#include "../ggml/src/ggml-gemmini/quants/act/tensor/tensor.hpp"
#include "../ggml/src/ggml-gemmini/quants/act/token/token.hpp"
#include "../ggml/src/ggml-gemmini/quants/act/quantize.hpp"

#include <cfloat>
#include <cfenv>
#include <cstdlib>
#include <cstdio>
#include <limits>
#include <new>
#include <vector>

namespace act = ggml::gemmini::quants::act;

namespace {

bool fail_next_allocation = false;

}

void * operator new(std::size_t size)
{
    if (fail_next_allocation) {
        fail_next_allocation = false;
        throw std::bad_alloc();
    }
    if (void * memory = std::malloc(size == 0 ? 1 : size)) {
        return memory;
    }
    throw std::bad_alloc();
}

void * operator new[](std::size_t size)
{
    return ::operator new(size);
}

void operator delete(void * memory) noexcept { std::free(memory); }
void operator delete[](void * memory) noexcept { std::free(memory); }
void operator delete(void * memory, std::size_t) noexcept { std::free(memory); }
void operator delete[](void * memory, std::size_t) noexcept { std::free(memory); }

static bool check_saturated_reconstruction(const char * name,
                                          const ggml_gemmini_args_t & args,
                                          int32_t expected)
{
    namespace rmd = ggml::gemmini::rmd;
    const auto & packets = ggml::gemmini::quants::activation_rmd_packets(args);
    const auto & direct = ggml::gemmini::quants::activation_direct_residuals(args);
    const bool packet_route = args.residual_route == act::ResidualRoute::ws_packet;
    if (args.act_quant.kind() == act::MetaKind::none ||
        (packet_route ? packets.empty() || !direct.empty() : direct.empty() || !packets.empty())) {
        std::fprintf(stderr, "FAIL: %s did not commit metadata and the selected residual payload\n", name);
        return false;
    }
    int64_t residual = 0;
    const size_t outlier_k = args.K - 1;
    for (const auto & packet : packets) {
        if (!packet || rmd::validate_packet(*packet) != rmd::RmdStatus::success) return false;
        for (const auto & block : packet->blocks) {
            for (size_t k = 0; k < block.compact_k_count; ++k) {
                if (block.global_k_begin + packet->k_indices[block.k_index_offset + k] != outlier_k)
                    continue;
                for (size_t lane = 0; lane < block.active_lane_count; ++lane) {
                    int32_t digit = 0;
                    if (rmd::read_packet_digit(*packet, block, lane, 0, k, digit) !=
                            rmd::RmdStatus::success) return false;
                    residual += static_cast<int64_t>(digit) *
                        (int64_t{1} << (packet->digit_bits * block.lane_ids[lane]));
                }
            }
        }
    }
    for (const auto & payload : direct) {
        if (!payload) return false;
        for (const auto & event : payload->events) {
            if (event.local_row == 0 && event.original_k == outlier_k) residual += event.residual;
        }
    }
    const int32_t clipped = expected < 0 ? ggml::gemmini::config::GGML_GEMMINI_ACTIVATION_QMIN :
                                         ggml::gemmini::config::GGML_GEMMINI_ACTIVATION_QMAX;
    if (args.A.get(0, outlier_k) != clipped ||
        static_cast<int64_t>(clipped) + residual != expected) {
        std::fprintf(stderr, "FAIL: %s clipped activation plus residual did not reconstruct INT32 endpoint\n", name);
        return false;
    }
    return true;
}

template<typename Meta, typename Quantize>
bool check_quantizer(const char *name, Quantize quantize)
{
    auto run = [&](float outlier, const int32_t * expected = nullptr) {
        std::vector<float> source(17, 1.0f);
        source.back() = outlier;

        ggml_tensor tensor{};
        tensor.type = GGML_TYPE_F32;
        tensor.data = source.data();

        for (const auto route : {act::ResidualRoute::ws_packet, act::ResidualRoute::cpu_direct}) {
            ggml_gemmini_args_t args{};
            args.I = 1;
            args.J = 1;
            args.K = source.size();
            args.A.allocate(args.I, args.K, GGML_GEMMINI_ACTIVATION_BITS);
            args.sA = args.K;
            args.residual_route = route;
            args.act_quant.storage().template emplace<Meta>();
            if (!quantize(&tensor, args) ||
                (expected && !check_saturated_reconstruction(name, args, *expected))) return false;
        }
        return true;
    };

    if (!run(16.0f)) {
        std::fprintf(stderr, "FAIL: %s rejected a representable finite outlier\n", name);
        return false;
    }
    const int32_t maximum = std::numeric_limits<int32_t>::max();
    const int32_t minimum = std::numeric_limits<int32_t>::min();
    std::feclearexcept(FE_ALL_EXCEPT);
    const bool accepted_overflow = run(FLT_MAX, &maximum);
    if (std::fetestexcept(FE_INVALID | FE_OVERFLOW) != 0) {
        std::fprintf(stderr, "FAIL: %s raised a floating exception before clamping to INT32\n", name);
        return false;
    }
    if (!accepted_overflow) {
        std::fprintf(stderr, "FAIL: %s failed to preserve a finite outlier saturated to INT32_MAX\n", name);
        return false;
    }
    std::feclearexcept(FE_ALL_EXCEPT);
    const bool accepted_negative_overflow = run(-FLT_MAX, &minimum);
    if (std::fetestexcept(FE_INVALID | FE_OVERFLOW) != 0) {
        std::fprintf(stderr, "FAIL: %s raised a floating exception while clamping to INT32_MIN\n", name);
        return false;
    }
    if (!accepted_negative_overflow) {
        std::fprintf(stderr, "FAIL: %s failed to preserve a finite outlier saturated to INT32_MIN\n", name);
        return false;
    }
    return true;
}

template<typename Meta, typename Quantize>
bool check_quantizer_zeroes_nonfinite(const char *name, Quantize quantize)
{
    const float values[] = {
        std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(),
    };
    for (float value : values) {
        std::vector<float> source(17, 1.0f);
        source[8] = value;
        std::vector<elem_t> quantized(source.size(), 42);
        ggml_tensor tensor{};
        tensor.type = GGML_TYPE_F32;
        tensor.data = source.data();
        ggml_gemmini_args_t args{};
        args.I = 1;
        args.J = 1;
        args.K = source.size();
        args.A.allocate(args.I, args.K, GGML_GEMMINI_ACTIVATION_BITS);
        args.sA = args.K;
        args.act_quant.storage().template emplace<Meta>();
        if (!quantize(&tensor, args) || args.A.get(0, 8) != 0) {
            std::fprintf(stderr, "FAIL: %s did not convert NaN/Inf to zero\n", name);
            return false;
        }
    }
    return true;
}

static bool check_public_quantizer_zeroes_nonfinite()
{
    const float values[] = {
        std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(),
    };
    for (float value : values) {
        std::vector<float> source(17, 1.0f);
        source[8] = value;
        std::vector<elem_t> quantized(source.size(), 42);
        ggml_tensor tensor{};
        tensor.type = GGML_TYPE_F32;
        tensor.data = source.data();
        ggml_gemmini_args_t args{};
        args.I = 1;
        args.J = 1;
        args.K = source.size();
        args.A.allocate(args.I, args.K, GGML_GEMMINI_ACTIVATION_BITS);
        args.sA = args.K;

        if (!ggml::gemmini::quants::quantize_activation(&tensor, args)) {
            std::fputs("FAIL: public activation quantizer rejected NaN/Inf instead of zeroing it\n", stderr);
            return false;
        }
        if (args.act_quant.kind() == act::MetaKind::none) {
            std::fputs("FAIL: accepted NaN/Inf did not commit activation metadata\n", stderr);
            return false;
        }
        if (args.A.get(0, 8) != 0) {
            std::fputs("FAIL: NaN/Inf was not converted to zero\n", stderr);
            return false;
        }
    }
    return true;
}

static bool check_public_quantizer_saturates_and_commits()
{
    for (float outlier : {FLT_MAX, -FLT_MAX}) {
        for (const auto route : {act::ResidualRoute::ws_packet, act::ResidualRoute::cpu_direct}) {
            std::vector<float> source(17, 1.0f);
            source.back() = outlier;
            ggml_tensor tensor{};
            tensor.type = GGML_TYPE_F32;
            tensor.data = source.data();
            ggml_gemmini_args_t args{};
            args.I = 1;
            args.J = 1;
            args.K = source.size();
            args.A.allocate(args.I, args.K, GGML_GEMMINI_ACTIVATION_BITS);
            args.sA = args.K;
            args.residual_route = route;
            const int32_t expected = outlier < 0 ? std::numeric_limits<int32_t>::min() :
                                                   std::numeric_limits<int32_t>::max();
            if (!ggml::gemmini::quants::quantize_activation(&tensor, args) ||
                !check_saturated_reconstruction("public quantizer", args, expected)) {
                std::fputs("FAIL: public activation quantizer did not commit saturated INT32 activation\n", stderr);
                return false;
            }
        }
    }
    return true;
}

static bool check_exact_packet_slice_reuses_handle()
{
    ggml::gemmini::rmd::RmdStripeBuilder builder;
    builder.reset(7, 16, 2, 32, 3);
    if (!builder.add_residual(0, 5, 257)) {
        std::fputs("FAIL: exact-slice fixture build failed\n", stderr);
        return false;
    }
    const auto packet = builder.finish();
    if (!packet) {
        std::fputs("FAIL: exact-slice fixture packet missing\n", stderr);
        return false;
    }
    ggml::gemmini::rmd::RmdStatus status{};
    const auto sliced = ggml::gemmini::rmd::slice_packets({packet}, 16, 18, 7, status);
    if (status != ggml::gemmini::rmd::RmdStatus::success || sliced != packet) {
        std::fputs("FAIL: exact packet slice rebuilt instead of reusing its handle\n", stderr);
        return false;
    }
    return true;
}

static bool check_radix_compose_allows_final_int64_cancellation()
{
    const auto contract =
        ggml::gemmini::rmd::balanced_radix_contract(
            GGML_GEMMINI_ACTIVATION_BITS);
    ggml::gemmini::rmd::RmdStripeBuilder builder;
    builder.reset(0, 0, 1, 32, 1);
    if (contract.radix == 0 ||
        !builder.add_residual(0, 0, static_cast<int32_t>(contract.radix + 1))) {
        return false;
    }
    const auto packet = builder.finish();
    if (!packet || packet->blocks.size() != 1) {
        return false;
    }
    const auto &block = packet->blocks.front();
    ggml::gemmini::rmd::CompressedOutput output;
    output.domain = ggml::gemmini::rmd::CompressedOutput::Domain::block_scaled_int64;
    output.j_padded = packet->j_padded;
    output.values.assign(packet->total_output_values, 0);
    for (uint8_t position = 0; position < block.active_lane_count; ++position) {
        const size_t offset = block.output_value_offset +
            static_cast<size_t>(position) * block.lane_stride_values;
        switch (block.lane_ids[position]) {
        case 0: output.values[offset] = std::numeric_limits<int64_t>::max(); break;
        case 1: output.values[offset] = -1; break;
        default: break;
        }
    }
    ggml::gemmini::rmd::Correction correction =
        ggml::gemmini::rmd::BlockScaledInt64Correction{};
    const auto status = ggml::gemmini::rmd::compose_rmd_output(*packet, output, correction);
    const int64_t expected =
        std::numeric_limits<int64_t>::max() - contract.radix;
    const auto * integer =
        std::get_if<ggml::gemmini::rmd::BlockScaledInt64Correction>(&correction);
    if (status != ggml::gemmini::rmd::RmdStatus::success || integer == nullptr ||
        integer->values != std::vector<int64_t>{expected}) {
        std::fputs("FAIL: radix compose rejected an in-range final INT64 after lane cancellation\n", stderr);
        return false;
    }
    for (uint8_t position = 0; position < block.active_lane_count; ++position) {
        if (block.lane_ids[position] == 1) {
            const size_t offset = block.output_value_offset +
                static_cast<size_t>(position) * block.lane_stride_values;
            output.values[offset] = 1;
        }
    }
    correction = ggml::gemmini::rmd::BlockScaledInt64Correction{{41, 42}};
    const auto unchanged = correction;
    const auto * unchanged_integer =
        std::get_if<ggml::gemmini::rmd::BlockScaledInt64Correction>(&unchanged);
    if (ggml::gemmini::rmd::compose_rmd_output(*packet, output, correction) !=
            ggml::gemmini::rmd::RmdStatus::overflow || unchanged_integer == nullptr ||
        std::get_if<ggml::gemmini::rmd::BlockScaledInt64Correction>(&correction) == nullptr ||
        std::get<ggml::gemmini::rmd::BlockScaledInt64Correction>(correction).values !=
            unchanged_integer->values) {
        std::fputs("FAIL: radix compose overflow modified caller output\n", stderr);
        return false;
    }
    return true;
}

static bool check_backend_neutral_checked_merge()
{
    constexpr size_t rows = 6;
    constexpr size_t columns = 2;
    constexpr size_t row_begin = 3;
    constexpr size_t row_end = 5;
    constexpr size_t row_stride = 7;
    constexpr size_t col_stride = 2;

    std::vector<block_q8_h1> weights(columns * 2);
    for (size_t j = 0; j < columns; ++j) {
        for (size_t block = 0; block < 2; ++block) {
            block_q8_h1 & weight = weights[j * 2 + block];
            weight.c_b = 1;
            weight.R = 0;
            weight.s_rf = j == 0 ? 0.5f : 2.0f;
        }
    }

    ggml_gemmini_args_t args{};
    args.I = rows;
    args.J = columns;
    args.K = 64;
    args.weight_format = ggml_gemmini_args_t::im2p_weight_format_t::q8_h1;
    args.q8_h1_blocks = weights.data();
    args.q8_h1_block_count = weights.size();
    args.q8_h1_rows = columns;
    args.blocks_per_row = 2;
    args.native_weight_bytes = weights.size() * sizeof(block_q8_h1);
    args.activation_row_offset = 1;
    auto & meta = args.act_quant.storage().emplace<act::token::Meta>();
    meta.scales = { 11.0f, 13.0f, 17.0f, 19.0f, 3.0f, 5.0f, 23.0f };
    args.stride_f_out = row_stride;
    args.col_stride_f_out = col_stride;

    ggml::gemmini::rmd::RmdStripeBuilder builder;
    builder.reset(4, row_begin, row_end - row_begin, args.K, columns);
    if (!builder.add_residual(0, 0, 1) || !builder.add_residual(1, 32, 1)) {
        std::fputs("FAIL: checked-merge packet fixture build failed\n", stderr);
        return false;
    }
    const auto packet = builder.finish();
    if (!packet) {
        std::fputs("FAIL: checked-merge packet fixture missing\n", stderr);
        return false;
    }

    const ggml::gemmini::rmd::Correction correction =
        ggml::gemmini::rmd::BlockScaledInt64Correction{{4, -3, 7, 5}};
    const std::vector<float> sentinel(rows * row_stride, -123.0f);
    std::vector<float> direct = sentinel;
    args.f_out = direct.data();
    const auto direct_status = ggml::gemmini::rmd::merge_rmd_correction(
        args, row_begin, row_end, correction);

    std::vector<float> wrapped = sentinel;
    args.f_out = wrapped.data();
    size_t nonzero_count = 99;
    const auto wrapped_status = ggml::gemmini::rmd::merge_rmd_correction(
        args, *packet, correction, &nonzero_count);
    if (direct_status != ggml::gemmini::rmd::RmdStatus::success ||
        wrapped_status != ggml::gemmini::rmd::RmdStatus::success || direct != wrapped ||
        nonzero_count != 4) {
        std::fputs("FAIL: direct and packet checked merges differ\n", stderr);
        return false;
    }
    if (direct[3 * row_stride] != -117.0f ||
        direct[3 * row_stride + 2] != -141.0f ||
        direct[4 * row_stride] != -105.5f ||
        direct[4 * row_stride + 2] != -73.0f) {
        std::fputs("FAIL: checked merge lost row offset, tail, scales, or output strides\n", stderr);
        return false;
    }
    for (size_t i = 0; i < direct.size(); ++i) {
        const bool written = i == 3 * row_stride || i == 3 * row_stride + 2 ||
            i == 4 * row_stride || i == 4 * row_stride + 2;
        if (!written && direct[i] != sentinel[i]) {
            std::fputs("FAIL: checked merge wrote outside its strided row range\n", stderr);
            return false;
        }
    }

    const ggml::gemmini::rmd::Correction sparse_correction =
        ggml::gemmini::rmd::BlockScaledInt64Correction{{0, -3, 7, 0}};
    for (auto & weight : weights) weight.s_rf = 0.0f;
    wrapped = sentinel;
    nonzero_count = 99;
    if (ggml::gemmini::rmd::merge_rmd_correction(
            args, *packet, sparse_correction, &nonzero_count) !=
            ggml::gemmini::rmd::RmdStatus::success ||
        nonzero_count != 2 || wrapped != sentinel) {
        std::fputs("FAIL: nonzero count must use raw correction before zero column scale\n", stderr);
        return false;
    }
    weights[0].s_rf = weights[1].s_rf = 0.5f;
    weights[2].s_rf = weights[3].s_rf = 2.0f;

    ggml::gemmini::rmd::RmdStripeBuilder block_zero_builder;
    block_zero_builder.reset(5, row_begin, row_end - row_begin, args.K, columns);
    if (!block_zero_builder.add_residual(0, 0, 1)) {
        std::fputs("FAIL: packet scale-scope fixture build failed\n", stderr);
        return false;
    }
    const auto block_zero_packet = block_zero_builder.finish();
    if (!block_zero_packet || block_zero_packet->blocks.size() != 1 ||
        block_zero_packet->blocks.front().block_id != 0) {
        std::fputs("FAIL: packet scale-scope fixture did not isolate block zero\n", stderr);
        return false;
    }

    weights[1].s_rf = 9.0f;
    weights[3].s_rf = 10.0f;
    wrapped = sentinel;
    args.f_out = wrapped.data();
    if (ggml::gemmini::rmd::merge_rmd_correction(args, *block_zero_packet, correction) !=
            ggml::gemmini::rmd::RmdStatus::success || wrapped != direct) {
        std::fputs("FAIL: packet merge rejected mismatched s_rf in an untouched block\n", stderr);
        return false;
    }

    wrapped = sentinel;
    args.f_out = wrapped.data();
    if (ggml::gemmini::rmd::merge_rmd_correction(args, *packet, correction) !=
            ggml::gemmini::rmd::RmdStatus::unsupported_route || wrapped != sentinel) {
        std::fputs("FAIL: packet merge accepted touched mismatched s_rf or modified output\n", stderr);
        return false;
    }

    wrapped = sentinel;
    args.f_out = wrapped.data();
    if (ggml::gemmini::rmd::merge_rmd_correction(
            args, row_begin, row_end, correction) !=
            ggml::gemmini::rmd::RmdStatus::unsupported_route || wrapped != sentinel) {
        std::fputs("FAIL: direct merge did not conservatively validate every Q8_H1 block\n", stderr);
        return false;
    }
    weights[1].s_rf = 0.5f;
    weights[3].s_rf = 2.0f;

    auto fails_without_writing = [&](const char * name,
                                     ggml::gemmini::rmd::RmdStatus expected,
                                     size_t begin, size_t end,
                                     const ggml::gemmini::rmd::Correction & values) {
        direct = sentinel;
        args.f_out = direct.data();
        size_t failure_count = 99;
        const auto status = ggml::gemmini::rmd::merge_rmd_correction(
            args, begin, end, values, &failure_count);
        if (status != expected || direct != sentinel || failure_count != 99) {
            std::fprintf(stderr, "FAIL: %s did not preserve sentinel output\n", name);
            return false;
        }
        return true;
    };

    meta.scales[5] = std::numeric_limits<float>::quiet_NaN();
    if (!fails_without_writing("invalid activation scale",
            ggml::gemmini::rmd::RmdStatus::invalid_arguments,
            row_begin, row_end, correction)) {
        return false;
    }
    meta.scales[5] = 5.0f;
    if (!fails_without_writing("invalid global row range",
            ggml::gemmini::rmd::RmdStatus::invalid_arguments,
            row_end, row_begin, correction) ||
        !fails_without_writing("row tail beyond args.I",
            ggml::gemmini::rmd::RmdStatus::invalid_arguments,
            row_begin, rows + 1, correction) ||
        !fails_without_writing("invalid correction dimensions",
            ggml::gemmini::rmd::RmdStatus::invalid_arguments,
            row_begin, row_end,
            ggml::gemmini::rmd::BlockScaledInt64Correction{{1, 2, 3}})) {
        return false;
    }
    args.stride_f_out = std::numeric_limits<size_t>::max();
    if (!fails_without_writing("overflowing output dimensions",
            ggml::gemmini::rmd::RmdStatus::invalid_arguments,
            row_begin, row_end, correction)) {
        return false;
    }
    args.stride_f_out = row_stride;

    weights[0].s_rf = weights[1].s_rf = std::numeric_limits<float>::quiet_NaN();
    if (!fails_without_writing("invalid column weight scale",
            ggml::gemmini::rmd::RmdStatus::unsupported_route,
            row_begin, row_end, correction)) {
        return false;
    }
    weights[0].s_rf = weights[1].s_rf = 0.5f;

    const ggml::gemmini::rmd::Correction overflowing =
        ggml::gemmini::rmd::BlockScaledInt64Correction{{
            0, 1, std::numeric_limits<int64_t>::max(), 1,
        }};
    weights[0].s_rf = weights[1].s_rf = std::numeric_limits<float>::max();
    if (!fails_without_writing("late floating merge overflow",
            ggml::gemmini::rmd::RmdStatus::overflow,
            row_begin, row_end, overflowing)) {
        return false;
    }
    weights[0].s_rf = weights[1].s_rf = 0.5f;

    direct = sentinel;
    args.f_out = direct.data();
    fail_next_allocation = true;
    const auto allocation_status = ggml::gemmini::rmd::merge_rmd_correction(
        args, row_begin, row_end, correction);
    fail_next_allocation = false;
    if (allocation_status != ggml::gemmini::rmd::RmdStatus::allocation_failure ||
        direct != sentinel) {
        std::fputs("FAIL: merge allocation failure modified sentinel output\n", stderr);
        return false;
    }
    return true;
}

int main()
{
    const bool ok =
        check_quantizer<act::tensor::Meta>("TENSOR", act::tensor::quantize) &&
        check_quantizer<act::token::Meta>("TOKEN", act::token::quantize) &&
        check_quantizer<act::stripe::Meta>("STRIPE", act::stripe::quantize) &&
        check_quantizer_zeroes_nonfinite<act::tensor::Meta>("TENSOR", act::tensor::quantize) &&
        check_quantizer_zeroes_nonfinite<act::token::Meta>("TOKEN", act::token::quantize) &&
        check_quantizer_zeroes_nonfinite<act::stripe::Meta>("STRIPE", act::stripe::quantize) &&
        check_public_quantizer_zeroes_nonfinite() &&
        check_public_quantizer_saturates_and_commits() &&
        check_exact_packet_slice_reuses_handle() &&
        check_radix_compose_allows_final_int64_cancellation() &&
        check_backend_neutral_checked_merge();
    if (ok)
        std::puts("PASS: activation bounds, radix compose, and backend-neutral checked merge");
    return ok ? 0 : 1;
}
