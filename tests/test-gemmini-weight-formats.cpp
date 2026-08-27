#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-gemmini.h"
#include "../ggml/src/ggml-gemmini/ggml-gemmini-q4-h1-reprocess.hpp"
#include "../ggml/src/ggml-gemmini/quants/common/weight_reader.hpp"
#include "../ggml/src/ggml-quants.h"
#include "../src/llama-quant.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

namespace {

namespace wreader = ggml::gemmini::quants::wreader;
namespace wroute = ggml::gemmini::quants::wroute;

using Format = ggml_gemmini_args_t::im2p_weight_format_t;
using ReaderStatus = wreader::WeightReaderStatus;
using Route = wroute::WeightRouteKind;
using RouteStatus = wroute::WeightRouteStatus;
using ScaleDomain = wroute::WeightScaleDomain;

bool check(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
    }
    return condition;
}

float max_abs_error(const std::vector<float> & lhs, const std::vector<float> & rhs) {
    float error = 0.0f;
    for (size_t i = 0; i < lhs.size(); ++i) {
        error = std::max(error, std::fabs(lhs[i] - rhs[i]));
    }
    return error;
}

bool round_trip(enum ggml_type type, float error_limit) {
    constexpr int64_t rows = 2;
    constexpr int64_t columns = 64;
    std::vector<float> source(rows * columns);
    for (int64_t i = 0; i < rows * columns; ++i) {
        source[i] = std::sin(static_cast<float>(i) * 0.17f) * 9.0f +
                    std::cos(static_cast<float>(i) * 0.031f);
    }
    source[0] = 0.0f;
    source[17] = -12.0f;
    source[63] = 11.5f;

    const size_t row_size = ggml_row_size(type, columns);
    std::vector<uint8_t> quantized(rows * row_size);
    const size_t written = ggml_quantize_chunk(
        type, source.data(), quantized.data(), 0, rows, columns, nullptr);
    if (!check(written == quantized.size(), "quantized byte count mismatch")) {
        return false;
    }
    if (!check(
            ggml_validate_row_data(type, quantized.data(), quantized.size()),
            "quantized row validation failed")) {
        return false;
    }

    const ggml_type_traits * traits = ggml_get_type_traits(type);
    if (!check(traits != nullptr && traits->to_float != nullptr, "missing to_float trait")) {
        return false;
    }

    std::vector<float> decoded(source.size());
    for (int64_t row = 0; row < rows; ++row) {
        traits->to_float(
            quantized.data() + row * row_size,
            decoded.data() + row * columns,
            columns);
    }

    return check(
        max_abs_error(source, decoded) <= error_limit,
        "round-trip error exceeded format limit");
}

enum class Family : uint8_t {
    H0,
    H1,
    HP1,
};

struct ReaderFixture {
    ggml_gemmini_args_t args{};
    block_q4_h0 q4_h0{};
    block_q4_h1 q4_h1{};
    block_q4_hp1 q4_hp1{};
    block_q8_0 q8_h0{};
    block_q8_h1 q8_h1{};
    block_q8_hp1 q8_hp1{};
    block_q16_h0 q16_h0{};
    block_q16_h1 q16_h1{};
    block_q16_hp1 q16_hp1{};

    ReaderFixture(uint8_t bits, Family family) {
        args.J = 1;
        args.K = 32;
        args.block_size_k = 32;
        args.native_block_count = 1;
        args.native_blocks_per_row = 1;

        std::fill(std::begin(q4_h0.qs), std::end(q4_h0.qs), uint8_t{0x88});
        std::fill(std::begin(q4_h1.qs), std::end(q4_h1.qs), uint8_t{0x88});
        std::fill(std::begin(q4_hp1.qs), std::end(q4_hp1.qs), uint8_t{0x88});
        q4_h0.qs[0] = 0x80;
        q4_h1.qs[0] = 0x80;
        q4_hp1.qs[0] = 0x80;
        q4_h0.qs[15] = 0xf1;
        q4_h1.qs[15] = 0xf1;
        q4_hp1.qs[15] = 0xf1;

        q8_h0.qs[0] = std::numeric_limits<int8_t>::min();
        q8_h0.qs[16] = 0;
        q8_h0.qs[31] = std::numeric_limits<int8_t>::max();
        q8_h1.qs[0] = std::numeric_limits<int8_t>::min();
        q8_h1.qs[16] = 0;
        q8_h1.qs[31] = std::numeric_limits<int8_t>::max();
        q8_hp1.qs[0] = std::numeric_limits<int8_t>::min();
        q8_hp1.qs[16] = 0;
        q8_hp1.qs[31] = std::numeric_limits<int8_t>::max();

        q16_h0.qs[0] = std::numeric_limits<int16_t>::min();
        q16_h0.qs[16] = 0;
        q16_h0.qs[31] = std::numeric_limits<int16_t>::max();
        q16_h1.qs[0] = std::numeric_limits<int16_t>::min();
        q16_h1.qs[16] = 0;
        q16_h1.qs[31] = std::numeric_limits<int16_t>::max();
        q16_hp1.qs[0] = std::numeric_limits<int16_t>::min();
        q16_hp1.qs[16] = 0;
        q16_hp1.qs[31] = std::numeric_limits<int16_t>::max();

        q4_h0.d = ggml_fp32_to_fp16(0.375f);
        q8_h0.d = ggml_fp32_to_fp16(0.375f);
        q16_h0.d = ggml_fp32_to_fp16(0.375f);

        q4_h1.c_b = 3;
        q4_h1.R = 4;
        q4_h1.s_rf = 0.25f;
        q8_h1.c_b = 3;
        q8_h1.R = 4;
        q8_h1.s_rf = 0.25f;
        q16_h1.c_b = 3;
        q16_h1.R = 4;
        q16_h1.s_rf = 0.25f;

        q4_hp1.m = 3;
        q4_hp1.channel_scale = 0.5f;
        q8_hp1.m = 3;
        q8_hp1.channel_scale = 0.5f;
        q16_hp1.m = 3;
        q16_hp1.channel_scale = 0.5f;

        if (bits == 4) {
            if (family == Family::H0) {
                args.weight_format = Format::q4_h0;
                args.q4_h0_blocks = &q4_h0;
                args.native_weight_bytes = sizeof(q4_h0);
            } else if (family == Family::H1) {
                args.weight_format = Format::q4_h1;
                args.q4_h1_blocks = &q4_h1;
                args.native_weight_bytes = sizeof(q4_h1);
            } else {
                args.weight_format = Format::q4_hp1;
                args.q4_hp1_blocks = &q4_hp1;
                args.native_weight_bytes = sizeof(q4_hp1);
            }
        } else if (bits == 8) {
            if (family == Family::H0) {
                args.weight_format = Format::q8_h0;
                args.B_blocks = &q8_h0;
                args.blocks_J = 1;
                args.blocks_K = 1;
                args.native_weight_bytes = sizeof(q8_h0);
            } else if (family == Family::H1) {
                args.weight_format = Format::q8_h1;
                args.q8_h1_blocks = &q8_h1;
                args.q8_h1_block_count = 1;
                args.q8_h1_rows = 1;
                args.blocks_per_row = 1;
                args.native_weight_bytes = sizeof(q8_h1);
            } else {
                args.weight_format = Format::q8_hp1;
                args.q8_hp1_blocks = &q8_hp1;
                args.q8_hp1_block_count = 1;
                args.q8_hp1_blocks_per_row = 1;
                args.native_weight_bytes = sizeof(q8_hp1);
            }
        } else {
            if (family == Family::H0) {
                args.weight_format = Format::q16_h0;
                args.q16_h0_blocks = &q16_h0;
                args.native_weight_bytes = sizeof(q16_h0);
            } else if (family == Family::H1) {
                args.weight_format = Format::q16_h1;
                args.q16_h1_blocks = &q16_h1;
                args.native_weight_bytes = sizeof(q16_h1);
            } else {
                args.weight_format = Format::q16_hp1;
                args.q16_hp1_blocks = &q16_hp1;
                args.native_weight_bytes = sizeof(q16_hp1);
            }
        }
    }

    wroute::WeightRoutePlan resolve() const {
        return wroute::resolve_weight_route_plan(args, wroute::WeightScaleInfoMode::Residual);
    }
};

struct HappyCase {
    uint8_t bits;
    Family family;
    Route route;
    ScaleDomain scale_domain;
    int32_t minimum;
    int32_t maximum;
    uint64_t integer_scale;
    float column_scale;
    float floating_scale;
};

bool test_reader_happy_table() {
    constexpr std::array<HappyCase, 9> cases = {{
        {4,  Family::H0,  Route::H0,  ScaleDomain::FloatingBlock,            -8,     7, 1, 1.0f, 0.375f},
        {4,  Family::H1,  Route::H1,  ScaleDomain::IntegerBlockTimesColumn,  -8,     7, 7, 0.25f, 1.0f},
        {4,  Family::HP1, Route::HP1, ScaleDomain::IntegerBlockTimesColumn,  -8,     7, 8, 0.5f,  1.0f},
        {8,  Family::H0,  Route::H0,  ScaleDomain::FloatingBlock,          -128,   127, 1, 1.0f, 0.375f},
        {8,  Family::H1,  Route::H1,  ScaleDomain::IntegerBlockTimesColumn,-128,   127, 7, 0.25f, 1.0f},
        {8,  Family::HP1, Route::HP1, ScaleDomain::IntegerBlockTimesColumn,-128,   127, 8, 0.5f,  1.0f},
        {16, Family::H0,  Route::H0,  ScaleDomain::FloatingBlock,        -32768, 32767, 1, 1.0f, 0.375f},
        {16, Family::H1,  Route::H1,  ScaleDomain::IntegerBlockTimesColumn,-32768, 32767, 7, 0.25f, 1.0f},
        {16, Family::HP1, Route::HP1, ScaleDomain::IntegerBlockTimesColumn,-32768, 32767, 8, 0.5f,  1.0f},
    }};

    bool ok = true;
    for (const HappyCase & test : cases) {
        ReaderFixture fixture(test.bits, test.family);
        const wroute::WeightRoutePlan plan = fixture.resolve();
        if (!check(plan.valid && plan.status == RouteStatus::Success,
                   "happy route resolves") ||
            !check(plan.route == test.route, "route family is width-independent") ||
            !check(plan.weight_bits == test.bits, "route records explicit weight width") ||
            !check(plan.scale_domain == test.scale_domain, "route records scale domain") ||
            !check(wroute::weight_route_status(plan, wroute::WeightExecutionPath::CpuDirect) ==
                       RouteStatus::Success,
                   "happy route supports CPU-direct")) {
            ok = false;
            continue;
        }

        const std::array<size_t, 3> positions = {0, 16, 31};
        const std::array<int32_t, 3> expected = {test.minimum, 0, test.maximum};
        for (size_t i = 0; i < positions.size(); ++i) {
            const wreader::WeightCodeResult code =
                wreader::read_code(fixture.args, plan, 0, positions[i]);
            ok = check(code.status == ReaderStatus::Success && code.value == expected[i],
                       "signed min/zero/max code decodes") && ok;
        }
        if (test.bits == 4) {
            const wreader::WeightCodeResult split_half_low =
                wreader::read_code(fixture.args, plan, 0, 15);
            ok = check(
                split_half_low.status == ReaderStatus::Success &&
                    split_half_low.value == -7,
                "Q4 split-half low nibble decodes logical K=15") && ok;
        }

        const wreader::WeightScaleResult scale =
            wreader::read_scale(fixture.args, plan, 0, 0);
        ok = check(scale.status == ReaderStatus::Success, "scale metadata reads") && ok;
        ok = check(scale.domain == test.scale_domain, "reader preserves scale domain") && ok;
        ok = check(scale.integer_block_scale == test.integer_scale,
                   "integer block factor matches fixture") && ok;
        ok = check(scale.column_scale == test.column_scale,
                   "column float factor matches fixture") && ok;
        ok = check(scale.floating_block_scale == test.floating_scale,
                   "floating block scale matches fixture") && ok;

        const float expected_effective = test.scale_domain == ScaleDomain::FloatingBlock ?
            test.floating_scale : static_cast<float>(test.integer_scale) * test.column_scale;
        ok = check(wroute::route_weight_scale(plan, fixture.args, 0, 0) == expected_effective,
                   "effective scale matches independent factorization") && ok;
        ok = check(
            wroute::weight_route_status(plan, wroute::WeightExecutionPath::Compact) ==
                (test.family == Family::H0 ? RouteStatus::UnsupportedExecution : RouteStatus::Success),
            "family exposes the expected compact capability") && ok;
    }
    return ok;
}

bool test_q4_h0_matches_canonical_dequantization() {
    ReaderFixture fixture(4, Family::H0);
    for (size_t i = 0; i < std::size(fixture.q4_h0.qs); ++i) {
        fixture.q4_h0.qs[i] =
            static_cast<uint8_t>(i | ((15 - i) << 4));
    }

    std::array<float, QK4_0> canonical{};
    dequantize_row_q4_0(&fixture.q4_h0, canonical.data(), canonical.size());

    const wroute::WeightRoutePlan plan = fixture.resolve();
    if (!check(plan.valid && plan.status == RouteStatus::Success,
               "Q4_H0 canonical comparison route resolves")) {
        return false;
    }

    bool ok = true;
    for (size_t k = 0; k < canonical.size(); ++k) {
        const wreader::WeightCodeResult code =
            wreader::read_code(fixture.args, plan, 0, k);
        const wreader::WeightScaleResult scale =
            wreader::read_scale(fixture.args, plan, 0, 0);
        const float decoded =
            static_cast<float>(code.value) * scale.floating_block_scale;
        ok = check(
                 code.status == ReaderStatus::Success &&
                     scale.status == ReaderStatus::Success &&
                     decoded == canonical[k],
                 "Q4_H0 reader matches canonical Q4_0 dequantization") && ok;
    }
    return ok;
}

bool test_reader_failure_table() {
    bool ok = true;

    ReaderFixture missing_extent(8, Family::H1);
    missing_extent.args.native_weight_bytes = 0;
    wroute::WeightRoutePlan plan = missing_extent.resolve();
    ok = check(!plan.valid && plan.status == RouteStatus::InvalidMetadata,
               "zero native byte extent rejects before reader dispatch") && ok;

    ReaderFixture truncated_q4(4, Family::H1);
    truncated_q4.args.native_weight_bytes = sizeof(block_q4_h1) - 1;
    plan = truncated_q4.resolve();
    ok = check(!plan.valid && plan.status == RouteStatus::InvalidMetadata,
               "truncated Q4 block is typed invalid metadata") && ok;

    ReaderFixture missing_q16_scale(16, Family::H1);
    missing_q16_scale.args.native_weight_bytes = offsetof(block_q16_h1, s_rf);
    plan = missing_q16_scale.resolve();
    ok = check(!plan.valid && plan.status == RouteStatus::InvalidMetadata,
               "Q16 block missing its scale is typed invalid metadata") && ok;

    ReaderFixture negative_hp1(8, Family::HP1);
    negative_hp1.q8_hp1.m = -1;
    plan = negative_hp1.resolve();
    ok = check(!plan.valid && plan.status == RouteStatus::InvalidMetadata,
               "negative HP1 exponent is typed invalid metadata") && ok;

    ReaderFixture overflowing_hp1(16, Family::HP1);
    overflowing_hp1.q16_hp1.m = 63;
    plan = overflowing_hp1.resolve();
    ok = check(!plan.valid && plan.status == RouteStatus::InvalidMetadata,
               "overflowing HP1 exponent is typed invalid metadata") && ok;

    ReaderFixture bad_padding(4, Family::H1);
    bad_padding.q4_h1.padding[0] = 1;
    plan = bad_padding.resolve();
    ok = check(!plan.valid && plan.status == RouteStatus::InvalidMetadata,
               "nonzero H1 padding is typed invalid metadata") && ok;

    ReaderFixture missing_blocks(16, Family::H0);
    missing_blocks.args.q16_h0_blocks = nullptr;
    plan = missing_blocks.resolve();
    ok = check(!plan.valid && plan.status == RouteStatus::InvalidMetadata,
               "missing native blocks are typed invalid metadata") && ok;

    ReaderFixture nan_scale(8, Family::H1);
    uint32_t nan_bits = 0x7fc00000u;
    std::memcpy(&nan_scale.q8_h1.s_rf, &nan_bits, sizeof(nan_bits));
    plan = nan_scale.resolve();
    ok = check(!plan.valid && plan.status == RouteStatus::InvalidMetadata,
               "non-finite H1 column scale is typed invalid metadata") && ok;

    ReaderFixture bounded_reader(4, Family::H0);
    plan = bounded_reader.resolve();
    ok = check(wreader::read_code(bounded_reader.args, plan, 1, 0).status ==
                   ReaderStatus::InvalidArguments &&
               wreader::read_code(bounded_reader.args, plan, 0, 32).status ==
                   ReaderStatus::InvalidArguments,
               "reader rejects row and K bounds without touching storage") && ok;

    ReaderFixture overflow_shape(4, Family::H0);
    overflow_shape.args.J = std::numeric_limits<size_t>::max();
    overflow_shape.args.K = 64;
    overflow_shape.args.native_blocks_per_row = 2;
    overflow_shape.args.native_block_count = std::numeric_limits<size_t>::max();
    overflow_shape.args.native_weight_bytes = 0;
    plan = overflow_shape.resolve();
    ok = check(!plan.valid && plan.status == RouteStatus::InvalidMetadata,
               "overflowing block geometry is typed invalid metadata") && ok;

    for (uint8_t bits : {uint8_t{4}, uint8_t{8}, uint8_t{16}}) {
        ReaderFixture h0(bits, Family::H0);
        plan = h0.resolve();
        ok = check(
            wroute::weight_route_status(plan, wroute::WeightExecutionPath::Compact) ==
                RouteStatus::UnsupportedExecution,
            "H0 compact request is typed unsupported") && ok;
    }

    block_q8_h2 h2{};
    ggml_gemmini_args_t h2_args{};
    h2_args.J = 1;
    h2_args.K = 32;
    h2_args.weight_format = Format::q8_h2;
    h2_args.q8_h2_blocks = &h2;
    h2_args.q8_h2_block_count = 1;
    h2_args.q8_h2_blocks_per_row = 1;
    plan = wroute::resolve_weight_route_plan(h2_args, wroute::WeightScaleInfoMode::Residual);
    ok = check(!plan.valid && plan.status == RouteStatus::UnsupportedFormat,
               "H2 is a typed unsupported family") && ok;

    block_q8_hp2 hp2{};
    ggml_gemmini_args_t hp2_args{};
    hp2_args.J = 1;
    hp2_args.K = 32;
    hp2_args.weight_format = Format::q8_hp2;
    hp2_args.q8_hp2_blocks = &hp2;
    hp2_args.q8_hp2_block_count = 1;
    hp2_args.q8_hp2_blocks_per_row = 1;
    plan = wroute::resolve_weight_route_plan(hp2_args, wroute::WeightScaleInfoMode::Residual);
    ok = check(!plan.valid && plan.status == RouteStatus::UnsupportedFormat,
               "HP2 is a typed unsupported family") && ok;

    return ok;
}

bool test_q4_h1_is_canonical_q4_0_reprocessing() {
    constexpr int64_t columns = 64;
    std::array<float, columns> source{};
    for (int64_t i = 0; i < columns; ++i) {
        source[i] = static_cast<float>((i * 11) % 29 - 14) / 4.0f;
    }
    source[0] = 8.0f;
    source[1] = -6.75f;
    source[32] = -8.0f;
    source[33] = 6.75f;

    std::array<block_q4_h1, 2> offline{};
    quantize_row_q4_h1_ref(source.data(), offline.data(), columns);

    std::array<block_q4_0, 2> q4_0{};
    quantize_row_q4_0_ref(source.data(), q4_0.data(), columns);

    ggml_tensor tensor{};
    tensor.type = GGML_TYPE_Q4_0;
    tensor.data = q4_0.data();
    tensor.ne[0] = columns;
    tensor.ne[1] = 1;
    tensor.ne[2] = 1;
    tensor.ne[3] = 1;
    tensor.nb[0] = sizeof(block_q4_0);
    tensor.nb[1] = sizeof(q4_0);
    tensor.nb[2] = sizeof(q4_0);
    tensor.nb[3] = sizeof(q4_0);

    std::vector<block_q4_h1> runtime;
    size_t blocks_per_row = 0;
    size_t logical_rows = 0;
    bool ok = check(
                  ggml::gemmini::prepare_q4_0_rows_for_q4_h1(
                      &tensor, runtime, &blocks_per_row, &logical_rows),
                  "runtime Q4_0 to Q4_H1 reprocessing succeeds") &&
        check(
            blocks_per_row == offline.size() && logical_rows == 1 &&
                runtime.size() == offline.size(),
            "runtime Q4_H1 geometry matches offline geometry") &&
        check(
            std::memcmp(offline.data(), runtime.data(), sizeof(offline)) == 0,
            "offline Q4_H1 equals runtime reprocessing of canonical Q4_0");

    uint16_t non_finite_bits = 0x7c00;
    std::memcpy(&q4_0[0].d, &non_finite_bits, sizeof(non_finite_bits));
    runtime.clear();
    ok = check(
             !ggml::gemmini::prepare_q4_0_rows_for_q4_h1(
                 &tensor, runtime, nullptr, nullptr),
             "runtime Q4_0 reprocessing rejects non-finite block scale") &&
        ok;
    return ok;
}

bool test_q4_h1_preserves_positive_q4_0_scale() {
    block_q4_0 source{};
    source.d = ggml_fp32_to_fp16(0.25f);
    for (size_t i = 0; i < std::size(source.qs); ++i) {
        source.qs[i] = static_cast<uint8_t>(i | ((15 - i) << 4));
    }

    block_q4_h1 converted{};
    if (!check(
            reprocess_row_q4_0_to_q4_h1_ref(&source, &converted, QK4_0),
            "positive-scale Q4_0 block reprocessing succeeds")) {
        return false;
    }

    std::array<float, QK4_0> canonical{};
    std::array<float, QK4_0> reprocessed{};
    dequantize_row_q4_0(&source, canonical.data(), QK4_0);
    dequantize_row_q4_h1(&converted, reprocessed.data(), QK4_0);
    return check(
        std::memcmp(canonical.data(), reprocessed.data(), sizeof(canonical)) == 0,
        "positive-scale Q4_0 values survive Q4_H1 reprocessing exactly");
}

bool test_q4_h1_flips_negative_q4_0_scale_codes() {
    block_q4_0 source{};
    source.d = ggml_fp32_to_fp16(-0.25f);
    for (size_t i = 0; i < std::size(source.qs); ++i) {
        source.qs[i] = static_cast<uint8_t>(i | ((15 - i) << 4));
    }

    block_q4_h1 converted{};
    if (!check(
            reprocess_row_q4_0_to_q4_h1_ref(&source, &converted, QK4_0),
            "negative-scale Q4_0 block reprocessing succeeds")) {
        return false;
    }

    std::array<float, QK4_0> canonical{};
    std::array<float, QK4_0> reprocessed{};
    dequantize_row_q4_0(&source, canonical.data(), QK4_0);
    dequantize_row_q4_h1(&converted, reprocessed.data(), QK4_0);
    bool ok = true;
    for (size_t i = 0; i < canonical.size(); ++i) {
        const uint8_t packed = source.qs[i % (QK4_0 / 2)];
        const uint8_t code = i < QK4_0 / 2 ? packed & 0x0f : packed >> 4;
        const float expected = code == 0 ? 1.75f : canonical[i];
        ok = check(
                 reprocessed[i] == expected,
                 "negative-scale Q4_0 conversion changes only unrepresentable +8 code") &&
             ok;
    }
    return ok;
}

bool test_q4_h1_narrow_scale_range_preserves_magnitude() {
    constexpr int64_t columns = 64;
    std::array<float, columns> source{};
    for (int64_t i = 0; i < columns; ++i) {
        const float block_max = i < 32 ? 1.0f : 1.001f;
        source[i] = block_max * static_cast<float>((i % 17) - 8) / 8.0f;
    }

    std::array<block_q4_0, 2> canonical{};
    quantize_row_q4_0_ref(source.data(), canonical.data(), columns);

    std::array<block_q4_h1, 2> quantized{};
    quantize_row_q4_h1_ref(source.data(), quantized.data(), columns);

    bool ok = true;
    for (size_t i = 0; i < quantized.size(); ++i) {
        const float source_scale = ggml_fp16_to_fp32(canonical[i].d);
        const float expected = std::fabs(source_scale);
        const float actual =
            quantized[i].s_rf * (static_cast<float>(quantized[i].R) + quantized[i].c_b);
        if (!(std::fabs(actual - expected) <= expected * 0.01f)) {
            std::fprintf(
                stderr,
                "FAIL: Q4_H1 narrow scale range collapsed: block=%zu expected=%g actual=%g R=%u\n",
                i,
                expected,
                actual,
                static_cast<unsigned>(quantized[i].R));
            ok = false;
        }
    }
    return ok;
}

bool test_q4_hp1_power_of_two_scale_uses_available_codes() {
    constexpr int64_t columns = 32;
    std::array<float, columns> source{};
    for (int64_t i = 0; i < columns; ++i) {
        source[i] = static_cast<float>(i - 16) / 16.0f;
    }

    block_q4_hp1 quantized{};
    if (!check(
            quantize_row_q4_hp1_ref(source.data(), &quantized, columns),
            "Q4_HP1 distributed-row quantization failed")) {
        return false;
    }

    std::array<float, columns> decoded{};
    dequantize_row_q4_hp1(&quantized, decoded.data(), columns);
    double squared_error = 0.0;
    for (int64_t i = 0; i < columns; ++i) {
        const double error = static_cast<double>(decoded[i]) - source[i];
        squared_error += error * error;
    }
    const double mean_squared_error = squared_error / columns;
    if (!(mean_squared_error < 0.002)) {
        std::fprintf(
            stderr,
            "FAIL: Q4_HP1 power-of-two scale wastes code range: mse=%g scale=%g exponent=%d\n",
            mean_squared_error,
            quantized.channel_scale,
            static_cast<int>(quantized.m));
        return false;
    }
    return true;
}

bool test_legacy_round_trips() {
    bool ok = true;
    ok = check(ggml_blck_size(GGML_TYPE_Q4_H1) == 32, "Q4_H1 block size") && ok;
    ok = check(ggml_blck_size(GGML_TYPE_Q4_HP1) == 32, "Q4_HP1 block size") && ok;
    ok = check(ggml_blck_size(GGML_TYPE_Q16_0) == 32, "Q16_0 block size") && ok;
    ok = check(ggml_blck_size(GGML_TYPE_Q16_H1) == 32, "Q16_H1 block size") && ok;
    ok = check(ggml_blck_size(GGML_TYPE_Q16_HP1) == 32, "Q16_HP1 block size") && ok;

    ok = round_trip(GGML_TYPE_Q4_H1, 2.5f) && ok;
    ok = round_trip(GGML_TYPE_Q4_HP1, 2.5f) && ok;
    ok = test_q4_h1_narrow_scale_range_preserves_magnitude() && ok;
    ok = test_q4_hp1_power_of_two_scale_uses_available_codes() && ok;
    ok = round_trip(GGML_TYPE_Q16_0, 0.01f) && ok;
    ok = round_trip(GGML_TYPE_Q16_H1, 0.02f) && ok;
    ok = round_trip(GGML_TYPE_Q16_HP1, 0.02f) && ok;
    return ok;
}

bool test_gemmini_q4_default_output_policy() {
    bool ok = true;
    struct Case {
        llama_ftype ftype;
        bool pure;
        bool is_output_weight;
        bool is_token_embedding_weight;
        ggml_type expected;
        const char * message;
    };
    const std::array<Case, 7> cases = {{
        { LLAMA_FTYPE_MOSTLY_Q4_0,  false, true,  false, GGML_TYPE_COUNT, "Q4_0 output keeps standard policy" },
        { LLAMA_FTYPE_MOSTLY_Q4_H1, false, true,  false, GGML_TYPE_F16,  "Q4_H1 output stays F16" },
        { LLAMA_FTYPE_MOSTLY_Q4_HP1,false, true,  false, GGML_TYPE_F16,  "Q4_HP1 output stays F16" },
        { LLAMA_FTYPE_MOSTLY_Q4_H1, false, true,  true,  GGML_TYPE_F16,  "Q4_H1 tied token/output stays F16" },
        { LLAMA_FTYPE_MOSTLY_Q4_H1, false, false, true,  GGML_TYPE_F16,  "Q4_H1 untied token embedding stays F16" },
        { LLAMA_FTYPE_MOSTLY_Q4_H1, true,  true,  true,  GGML_TYPE_COUNT, "pure Q4_H1 bypasses mixed policy" },
        { LLAMA_FTYPE_MOSTLY_Q4_H1, false, false, false, GGML_TYPE_COUNT, "ordinary Q4_H1 tensor stays quantized" },
    }};
    for (const Case & test : cases) {
        ok = check(
                 llama_quantize_gemmini_q4_default_tensor_type(
                     test.ftype, test.pure, test.is_output_weight, test.is_token_embedding_weight) == test.expected,
                 test.message) && ok;
    }
    return ok;
}

bool test_q4_hp1_loader_contract() {
#if GGML_GEMMINI_ACTIVATION_BITS == 4 && GGML_GEMMINI_WEIGHT_BITS == 4
    ggml_init_params params = {
        /* .mem_size   = */ ggml_tensor_overhead() * 4,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!check(ctx != nullptr, "failed to create loader-contract context")) {
        return false;
    }

    ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_HP1, 32, 2);
    ggml_tensor * activation = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, 1);
    ggml_tensor * op = ggml_mul_mat(ctx, weight, activation);

    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(ggml_backend_gemmini_reg(), 0);
    ggml_backend_buffer_t buffer =
        ggml_backend_buft_alloc_buffer(ggml_backend_dev_buffer_type(dev), 0);
    weight->buffer = buffer;
    const bool supported = ggml_backend_dev_supports_op(dev, op);
    weight->buffer = nullptr;
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);

    return check(supported, "production GEMMINI rejects Q4_HP1 loader metadata");
#else
    return true;
#endif
}

enum class Selection {
    All,
    HappyTable,
    FailureTable,
    Invalid,
};

Selection parse_selection(int argc, char ** argv) {
    if (argc == 1) {
        return Selection::All;
    }
    if (argc != 2) {
        return Selection::Invalid;
    }
    if (std::strcmp(argv[1], "--case=happy-table") == 0) {
        return Selection::HappyTable;
    }
    if (std::strcmp(argv[1], "--case=failure-table") == 0) {
        return Selection::FailureTable;
    }
    return Selection::Invalid;
}

} // namespace

int main(int argc, char ** argv) {
    const Selection selection = parse_selection(argc, argv);
    if (selection == Selection::Invalid) {
        std::fputs("usage: gemmini_weight_formats [--case=happy-table|--case=failure-table]\n", stderr);
        return 2;
    }

    bool ok = true;
    if (selection == Selection::All || selection == Selection::HappyTable) {
        ok = test_gemmini_q4_default_output_policy() && ok;
        ok = test_legacy_round_trips() && ok;
        ok = test_q4_h1_is_canonical_q4_0_reprocessing() && ok;
        ok = test_q4_h1_preserves_positive_q4_0_scale() && ok;
        ok = test_q4_h1_flips_negative_q4_0_scale_codes() && ok;
        ok = test_q4_hp1_loader_contract() && ok;
        ok = test_reader_happy_table() && ok;
        ok = test_q4_h0_matches_canonical_dequantization() && ok;
    }
    if (selection == Selection::All || selection == Selection::FailureTable) {
        ok = test_reader_failure_table() && ok;
    }
    if (ok) {
        const char *message = selection == Selection::All ?
            "PASS: residual weight happy and failure tables" :
            (selection == Selection::FailureTable ?
                "PASS: residual weight failure table" :
                "PASS: residual weight happy table");
        std::puts(message);
    }
    return ok ? 0 : 1;
}
