#include "ggml.h"
#include "../ggml/src/ggml-gemmini/quants/common/weight_reader.hpp"

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
        q4_h0.qs[15] = 0xf8;
        q4_h1.qs[15] = 0xf8;
        q4_hp1.qs[15] = 0xf8;

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

bool test_legacy_round_trips() {
    bool ok = true;
    ok = check(ggml_blck_size(GGML_TYPE_Q4_H1) == 32, "Q4_H1 block size") && ok;
    ok = check(ggml_blck_size(GGML_TYPE_Q4_HP1) == 32, "Q4_HP1 block size") && ok;
    ok = check(ggml_blck_size(GGML_TYPE_Q16_0) == 32, "Q16_0 block size") && ok;
    ok = check(ggml_blck_size(GGML_TYPE_Q16_H1) == 32, "Q16_H1 block size") && ok;
    ok = check(ggml_blck_size(GGML_TYPE_Q16_HP1) == 32, "Q16_HP1 block size") && ok;

    ok = round_trip(GGML_TYPE_Q4_H1, 2.5f) && ok;
    ok = round_trip(GGML_TYPE_Q4_HP1, 2.5f) && ok;
    ok = round_trip(GGML_TYPE_Q16_0, 0.01f) && ok;
    ok = round_trip(GGML_TYPE_Q16_H1, 0.02f) && ok;
    ok = round_trip(GGML_TYPE_Q16_HP1, 0.02f) && ok;
    return ok;
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
        ok = test_legacy_round_trips() && ok;
        ok = test_reader_happy_table() && ok;
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
