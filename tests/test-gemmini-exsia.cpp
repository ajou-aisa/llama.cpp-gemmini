#include "../ggml/src/ggml-gemmini/ggml-gemmini-args.h"
#include "../ggml/src/ggml-gemmini/quants/act/dispatch.hpp"
#include "../ggml/src/ggml-gemmini/quants/act/quantize.hpp"
#include "../ggml/src/ggml-gemmini/quants/act/exsia/types.hpp"
#include "../ggml/src/ggml-gemmini/quants/act/exsia/exsia.hpp"
#include "../ggml/src/ggml-gemmini/quants/act/stripe/stripe.hpp"
#include "../ggml/src/ggml-gemmini/quants/act/stripe/types.hpp"
#include "../ggml/src/ggml-gemmini/quants/act/token/types.hpp"
#include "../ggml/src/ggml-gemmini/quants/common/weight_reader.hpp"

#include <ggml.h>
#ifndef GEMMINI_EXSIA_WRITER_TEST_ONLY
#include "../ggml/src/ggml-gemmini/residual/direct/direct-builder.hpp"
#include "../ggml/src/ggml-gemmini/residual/direct/direct-executor.hpp"
#include "../ggml/src/ggml-gemmini/residual/rmd/rmd-builder.hpp"
#include "../ggml/src/ggml-gemmini/residual/rmd/rmd-compose.hpp"
#include "../ggml/src/ggml-gemmini/residual/rmd/rmd-executor.hpp"
#include "../ggml/src/ggml-gemmini/residual/rmd/rmd-reference.hpp"
#include <gemmini.h>
#endif

#include <cstdio>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace ggml::gemmini;

bool check(bool value, const char * message) {
    if (!value)
        std::fprintf(stderr, "FAIL: %s\n", message);
    return value;
}

#ifndef GEMMINI_EXSIA_WRITER_TEST_ONLY
bool test_meta_rho_invariant() {
    quants::act::exsia::Meta meta;
    const bool initialized =
        meta.rho == config::GGML_GEMMINI_ACTIVATION_RHO;
    meta.rho = std::numeric_limits<int16_t>::max();
    meta.theta.push_back(1);
    meta.reset();
    return check(initialized, "ExSIA metadata initializes width-native rho") &&
        check(meta.rho == config::GGML_GEMMINI_ACTIVATION_RHO,
              "ExSIA metadata reset restores width-native rho") &&
        check(meta.theta.empty(), "ExSIA metadata reset clears theta");
}

bool test_non_outlier_clipping_policy() {
#if GGML_GEMMINI_ACTIVATION_BITS == 4
    constexpr size_t columns = 32;
    std::array<float, columns> source{};
    source.fill(1.9f);
    ggml_tensor tensor{};
    tensor.type = GGML_TYPE_F32;
    tensor.data = source.data();

    ggml_gemmini_args_t args{};
    args.I = 1;
    args.J = 1;
    args.K = columns;
    args.sA = columns;
    args.tile_I = 1;
    args.tile_J = 1;
    args.tile_K = 1;
    args.activation_rows_per_stripe = DIM;
    args.residual_route = residual::ResidualRoute::cpu_direct;
    if (!args.A.allocate(1, columns, 4)) {
        return false;
    }

    quants::act::exsia::Meta meta;
    quants::act::exsia::ExSIA exsia;
    const bool quantized = exsia.run(meta, &tensor, args);
    bool clipped_to_qmax = quantized;
    for (size_t column = 0; column < columns; ++column) {
        clipped_to_qmax =
            args.A.get(0, column) == config::GGML_GEMMINI_ACTIVATION_QMAX &&
            clipped_to_qmax;
    }
    const bool residuals_discarded =
        std::all_of(exsia.state().residual.begin(),
                    exsia.state().residual.end(),
                    [](int32_t value) { return value == 0; });
    return check(quantized && meta.theta == std::vector<int16_t>{-2},
                 "equal A4 values select the expected ExSIA theta") &&
        check(clipped_to_qmax,
              "non-outlier A4 overflow remains clipped to qmax") &&
        check(residuals_discarded && meta.rmd_packets.empty() &&
                  meta.direct_residuals.empty(),
              "non-outlier clipping remains residual-free");
#else
    return true;
#endif
}

const std::vector<rmd::OutputValue> * integer_values(const rmd::Correction & correction) {
    const auto * typed = std::get_if<rmd::BlockScaledInt64Correction>(&correction);
    return typed == nullptr ? nullptr : &typed->values;
}

bool integer_values_equal(const rmd::Correction & correction,
                          const std::vector<rmd::OutputValue> & expected) {
    const auto * values = integer_values(correction);
    return values != nullptr && *values == expected;
}

struct GeometryPublicationTrace {
    std::array<std::pair<size_t, size_t>, 2> rows{};
    size_t publications = 0;
    size_t direct_handles = 0;
    size_t packet_handles = 0;
};

bool capture_geometry_publication(
    void * opaque,
    const quants::act::exsia::StripeReadyEvent & event) {
    auto & trace = *static_cast<GeometryPublicationTrace *>(opaque);
    if (trace.publications >= trace.rows.size()) return false;
    trace.rows[trace.publications++] = {event.row_begin, event.row_end};
    trace.direct_handles += event.direct_residual != nullptr ? 1 : 0;
    trace.packet_handles += event.rmd_packet != nullptr ? 1 : 0;
    return true;
}

bool test_activation_stripe_geometry_contract() {
    constexpr size_t rows = 33;
    constexpr size_t cols = 32;
    std::vector<float> source(rows * cols, 0.5f);
    source[0] = 32.0f;
    ggml_tensor tensor{};
    tensor.type = GGML_TYPE_F32;
    tensor.data = source.data();

    ggml_gemmini_args_t exsia_args{};
    exsia_args.I = rows; exsia_args.J = 8; exsia_args.K = cols;
    exsia_args.sA = cols;
    exsia_args.tile_I = 2; exsia_args.tile_J = 3; exsia_args.tile_K = 4;
    exsia_args.activation_rows_per_stripe = 32;
    exsia_args.residual_route = residual::ResidualRoute::cpu_direct;
    if (!exsia_args.A.allocate(
            rows, cols, static_cast<uint8_t>(GGML_GEMMINI_ACTIVATION_BITS))) {
        return false;
    }
    quants::act::exsia::Meta exsia_meta;
    GeometryPublicationTrace trace;
    quants::act::exsia::StripeReadySink sink{&trace, capture_geometry_publication};
    quants::act::exsia::ExSIA exsia;
    exsia.set_execution_mode(quants::act::exsia::ExSIAState::ExecutionMode::Sequential);
    const bool exsia_ok = exsia.run(exsia_meta, &tensor, exsia_args, &sink);

    ggml_gemmini_args_t stripe_args{};
    stripe_args.I = rows; stripe_args.J = 8; stripe_args.K = cols;
    stripe_args.sA = cols;
    stripe_args.tile_I = 2; stripe_args.tile_J = 3; stripe_args.tile_K = 4;
    stripe_args.activation_rows_per_stripe = 32;
    stripe_args.residual_route = residual::ResidualRoute::cpu_direct;
    stripe_args.act_quant.storage().emplace<quants::act::stripe::Meta>();
    if (!stripe_args.A.allocate(
            rows, cols, static_cast<uint8_t>(GGML_GEMMINI_ACTIVATION_BITS))) {
        return false;
    }
    const bool stripe_ok = quants::act::stripe::quantize(&tensor, stripe_args);
    const auto * stripe_meta = std::get_if<quants::act::stripe::Meta>(&stripe_args.act_quant.storage());

    ggml_gemmini_args_t bad_exsia_args = exsia_args;
    bad_exsia_args.activation_rows_per_stripe = 16;
    for (size_t row = 0; row < rows; ++row)
        for (size_t col = 0; col < cols; ++col)
            bad_exsia_args.A.set(row, col, 17);
    const std::vector<uint8_t> direct_sentinel_before = *bad_exsia_args.A.bytes;
    quants::act::exsia::Meta bad_exsia_meta;
    GeometryPublicationTrace bad_trace;
    quants::act::exsia::StripeReadySink bad_sink{&bad_trace, capture_geometry_publication};
    quants::act::exsia::ExSIA bad_exsia;
    const bool bad_exsia_ok = bad_exsia.run(bad_exsia_meta, &tensor, bad_exsia_args, &bad_sink);
    const bool direct_sentinel_unchanged =
        *bad_exsia_args.A.bytes == direct_sentinel_before;

    ggml_gemmini_args_t public_bad_args{};
    public_bad_args.I = rows; public_bad_args.J = 8; public_bad_args.K = cols;
    public_bad_args.sA = cols;
    public_bad_args.tile_I = 2; public_bad_args.tile_J = 3; public_bad_args.tile_K = 4;
    public_bad_args.activation_rows_per_stripe = 16;
    public_bad_args.residual_route = residual::ResidualRoute::cpu_direct;
    if (!public_bad_args.A.allocate(
            rows, cols, static_cast<uint8_t>(GGML_GEMMINI_ACTIVATION_BITS))) {
        return false;
    }
    for (size_t row = 0; row < rows; ++row)
        for (size_t col = 0; col < cols; ++col)
            public_bad_args.A.set(row, col, 23);
    const std::vector<uint8_t> public_sentinel_before = *public_bad_args.A.bytes;
    const bool public_bad_ok = quants::quantize_activation(&tensor, public_bad_args);
    const bool public_sentinel_unchanged =
        *public_bad_args.A.bytes == public_sentinel_before;

    ggml_gemmini_args_t bad_stripe_args = stripe_args;
    bad_stripe_args.activation_rows_per_stripe = 16;
    bad_stripe_args.act_quant.storage().emplace<quants::act::stripe::Meta>();
    const bool bad_stripe_ok = quants::act::stripe::quantize(&tensor, bad_stripe_args);
    const auto * bad_stripe_meta = std::get_if<quants::act::stripe::Meta>(&bad_stripe_args.act_quant.storage());

    std::vector<float> exsia_decoded(rows * cols);
    ggml_gemmini_args_t exsia_consumer_args = exsia_args;
    exsia_consumer_args.act_quant.storage().emplace<quants::act::exsia::Meta>(exsia_meta);
    const bool exsia_dequantized = quants::act::exsia::dequantize_activation(
        exsia_decoded.data(), cols, 1, rows, cols, exsia_consumer_args);
    ggml_gemmini_args_t bad_exsia_consumer_args = exsia_consumer_args;
    bad_exsia_consumer_args.activation_rows_per_stripe = 16;
    const bool bad_exsia_dequantized = quants::act::exsia::dequantize_activation(
        exsia_decoded.data(), cols, 1, rows, cols, bad_exsia_consumer_args);

    std::vector<float> stripe_decoded(rows * cols);
    const bool stripe_dequantized = quants::act::stripe::dequantize_activation(
        stripe_decoded.data(), cols, 1, rows, cols, stripe_args);
    ggml_gemmini_args_t bad_stripe_consumer_args = stripe_args;
    bad_stripe_consumer_args.activation_rows_per_stripe = 16;
    const bool bad_stripe_dequantized = quants::act::stripe::dequantize_activation(
        stripe_decoded.data(), cols, 1, rows, cols, bad_stripe_consumer_args);

    const bool direct_atomic_ok =
        check(!bad_exsia_ok && bad_exsia.state().failure_code ==
                  quants::act::exsia::ExSIAState::FailureCode::InvalidInput &&
                  bad_trace.publications == 0 && bad_exsia_meta.theta.empty() &&
                  bad_exsia.state().stripe.empty() && bad_exsia.state().residual.empty() &&
                  direct_sentinel_unchanged,
              "direct ExSIA mismatch is typed, atomic, and has zero side effects");
    const bool public_atomic_ok =
        check(!public_bad_ok && public_sentinel_unchanged &&
                  std::holds_alternative<quants::act::NoneMeta>(public_bad_args.act_quant.storage()),
              "public ExSIA mismatch preserves the activation sentinel");
#if GGML_GEMMINI_ENABLE_RMD
    const bool residual_producer_ok =
        check(trace.direct_handles > 0 && trace.packet_handles == 0 &&
                  !exsia_meta.direct_residuals.empty(),
              "RMD-enabled ExSIA publishes direct residual payloads");
#else
    const bool residual_producer_ok =
        check(trace.direct_handles == 0 && trace.packet_handles == 0 &&
                  exsia_meta.direct_residuals.empty() &&
                  exsia_meta.rmd_packets.empty(),
              "RMD-disabled ExSIA publishes no residual payloads");
#endif

    const bool ok =
        check(exsia_ok && stripe_ok, "ExSIA and STRIPE accept matching geometry") &&
        check(exsia_args.tile_I == 2 && exsia_args.tile_J == 3 && exsia_args.tile_K == 4 &&
                  stripe_args.tile_I == 2 && stripe_args.tile_J == 3 && stripe_args.tile_K == 4,
              "quantization preserves all auto-selected tile factors") &&
        check(trace.publications == 2 && trace.rows[0] == std::make_pair<size_t, size_t>(0, 32) &&
                  trace.rows[1] == std::make_pair<size_t, size_t>(32, 33),
              "ExSIA publishes contiguous 0-32 and final 32-33 rows") &&
        check(exsia_meta.rho == config::GGML_GEMMINI_ACTIVATION_RHO,
              "successful ExSIA publishes width-aware rho") &&
        check(exsia_meta.theta.size() == 2 && stripe_meta != nullptr && stripe_meta->scales.size() == 2,
              "ExSIA and STRIPE produce two stripes from identical geometry") &&
        check(exsia_dequantized && stripe_dequantized &&
                  !bad_exsia_dequantized && !bad_stripe_dequantized,
              "ExSIA and STRIPE consumers accept only matching geometry") &&
        direct_atomic_ok && public_atomic_ok && residual_producer_ok &&
        check(!bad_stripe_ok && bad_stripe_meta != nullptr && bad_stripe_meta->scales.empty() &&
                  bad_stripe_meta->rmd_packets.empty() && bad_stripe_meta->direct_residuals.empty(),
              "STRIPE metadata mismatch has zero metadata side effects");
    if (ok) std::printf("STRIPE_GEOMETRY_QA tile=2/3/4 exsia_rows=0-32,32-33 stripe_rows=0-32,32-33 mismatch=InvalidInput direct_before=17 direct_after=17 public_before=23 public_after=23 publications=0 allocations=0\n");
    return ok;
}

bool test_cpu_direct_residual_dequantization() {
    ggml_gemmini_args_t args{};
    args.I = 2;
    args.J = 1;
    args.K = 2;
    args.sA = args.K;
    args.activation_rows_per_stripe = 2;
    args.residual_route = residual::ResidualRoute::cpu_direct;
    if (!args.A.allocate(
            args.I, args.K, static_cast<uint8_t>(GGML_GEMMINI_ACTIVATION_BITS))) {
        return false;
    }
    const std::array<int32_t, 4> activation = {1, 2, 3, 4};
    for (size_t row = 0; row < args.I; ++row)
        for (size_t col = 0; col < args.K; ++col)
            if (!args.A.set(row, col, activation[row * args.K + col])) return false;

    residual::DirectStripeBuilder first;
    first.reset(0, 0, 1, args.K, args.J);
    if (!first.add_residual(0, 0, 2)) return false;
    residual::DirectStripeBuilder second;
    second.reset(1, 1, 1, args.K, args.J);
    if (!second.add_residual(0, 1, -1)) return false;

    auto & meta = args.act_quant.storage().emplace<quants::act::exsia::Meta>();
    meta.theta = {1};
    meta.direct_residuals = {first.finish(), second.finish()};

    std::array<float, 4> decoded = {-99.0f, -99.0f, -99.0f, -99.0f};
    const bool success = quants::act::exsia::dequantize_activation(
        decoded.data(), args.K, 1, args.I, args.K, args);
    const std::array<float, 4> expected = {6.0f, 4.0f, 6.0f, 6.0f};

    ggml_gemmini_args_t sliced = args;
    sliced.I = 1;
    sliced.A = args.A.slice_rows(1, 1);
    sliced.activation_row_offset = 1;
    std::array<float, 2> sliced_output = {-11.0f, -11.0f};
    const bool sliced_success = quants::act::exsia::dequantize_activation(
        sliced_output.data(), args.K, 1, 1, args.K, sliced);

    ggml_gemmini_args_t wrong_route = args;
    wrong_route.residual_route = residual::ResidualRoute::ws_packet;
    std::array<float, 4> wrong_route_output = {-17.0f, -17.0f, -17.0f, -17.0f};
    const bool wrong_route_success = quants::act::exsia::dequantize_activation(
        wrong_route_output.data(), args.K, 1, args.I, args.K, wrong_route);

    ggml_gemmini_args_t overflow = args;
    residual::DirectStripeBuilder overflow_builder;
    overflow_builder.reset(0, 0, 1, args.K, args.J);
    if (!overflow_builder.add_residual(
            0, 0, std::numeric_limits<int32_t>::max())) {
        return false;
    }
    auto & overflow_meta =
        std::get<quants::act::exsia::Meta>(overflow.act_quant.storage());
    overflow_meta.direct_residuals = {overflow_builder.finish()};
    std::array<float, 4> overflow_output = {-31.0f, -31.0f, -31.0f, -31.0f};
    const bool overflow_success = quants::act::exsia::dequantize_activation(
        overflow_output.data(), args.K, 1, args.I, args.K, overflow);

    return check(success && decoded == expected,
                 "CPU-direct residuals participate in activation dequantization") &&
        check(sliced_success &&
                  sliced_output == std::array<float, 2>{6.0f, 6.0f},
              "CPU-direct dequantization maps sliced global rows") &&
        check(!wrong_route_success &&
                  wrong_route_output == std::array<float, 4>{
                      -17.0f, -17.0f, -17.0f, -17.0f},
              "dequantization rejects wrong-route residual metadata atomically") &&
        check(!overflow_success &&
                  overflow_output == std::array<float, 4>{
                      -31.0f, -31.0f, -31.0f, -31.0f},
              "dequantization rejects residual addition overflow atomically");
}

bool test_exsia_baseline() {
    elem_t activation = 3;
    elem_t weight = 4;
    float output = 0.0f;
    ggml_gemmini_args_t args{};
    args.I = 1;
    args.J = 1;
    args.K = 1;
    args.A.allocate(1, 1, GGML_GEMMINI_ACTIVATION_BITS);
    args.A.set(0, 0, activation);
    args.B = &weight;
    args.f_out = &output;
    args.sA = 1;
    args.sB = 1;
    args.stride_f_out = 1;
    args.col_stride_f_out = 1;
    args.weight_i8_scale_active = true;
    args.weight_scale = 1.0f;
    args.tiled_matmul_type = CPU;
    auto & meta = args.act_quant.storage().emplace<quants::act::exsia::Meta>();
    meta.theta = { 0 };
    tiled_matmul_auto_baseline(&args, baseline_activation_quant_t::EXSIA,
                               baseline_weight_quant_t::TENSOR);
    return check(output == 12.0f, "ExSIA baseline output") &&
        check(meta.rmd_packets.empty(), "empty residual has no RMD packet");
}

bool test_dispatch_modes() {
    elem_t activation = 3;
    elem_t weight = 4;
    float output = 0.0f;
    ggml_gemmini_args_t args{};
    args.I = args.J = args.K = 1;
    args.A.allocate(1, 1, GGML_GEMMINI_ACTIVATION_BITS);
    args.A.set(0, 0, activation);
    args.B = &weight;
    args.f_out = &output;
    args.sA = args.sB = args.stride_f_out = args.col_stride_f_out = 1;
    args.weight_i8_scale_active = true;
    args.weight_scale = 1.0f;
    args.tiled_matmul_type = CPU;
    auto & meta = args.act_quant.storage().emplace<quants::act::exsia::Meta>();
    meta.theta = { 0 };
    const auto & packets = quants::act::rmd_packets(args);
    return check(packets.empty(), "dispatch exposes empty RMD packet list");
}

bool test_rmd_cpu_ws_routes() {
    elem_t weight = 4;
    ggml_gemmini_args_t args{};
    args.I = args.J = args.K = 1;
    args.B = &weight;
    args.sB = 1;
    args.weight_i8_scale_active = true;
    args.weight_scale = 1.0f;

    rmd::RmdStripeBuilder builder;
    builder.reset(0, 0, 1, 1, 1);
    if (!builder.add_residual(0, 0, 256)) {
        return check(false, "RMD route packet input accepted");
    }
    const rmd::StripePacketHandle packet = builder.finish();
    if (!check(packet != nullptr, "RMD route packet built")) {
        return false;
    }

    rmd::CompressedOutput cpu_output;
    rmd::RmdExecutionMetrics route_metrics{};
    args.tiled_matmul_type = OS;
    const rmd::RmdStatus cpu =
        rmd::execute_rmd_stripe_reference(args, *packet, cpu_output, &route_metrics);
    std::vector<rmd::ReferenceResidual> route_events = {{0, 0, 256}};
    std::vector<rmd::OutputValue> route_direct;
    rmd::Correction route_packet = rmd::BlockScaledInt64Correction{};
    const auto route_direct_status = rmd::reference_direct_correction(args, 1, route_events, route_direct);
    const auto route_packet_status = rmd::compose_rmd_output(*packet, cpu_output, route_packet);
    const auto * route_packet_values = integer_values(route_packet);
    if (!check(cpu == rmd::RmdStatus::success && route_direct_status == rmd::RmdStatus::success &&
               route_packet_status == rmd::RmdStatus::success && route_packet_values != nullptr &&
               route_direct == std::vector<rmd::OutputValue>{1024} &&
               route_direct == *route_packet_values && !cpu_output.values.empty() && cpu_output.values.front() == 4,
               "RMD reference API ignores the runtime matmul route")) {
        return false;
    }
    const rmd::StripePacket packet_before = *packet;
    rmd::CompressedOutput output;
    output.j_padded = 7;
    output.values = { 11, 22, 33 };
    const rmd::CompressedOutput unchanged = output;
    rmd::StripePacket malformed = *packet;
    const rmd::BlockDescriptor & first = malformed.blocks.front();
    malformed.stacked_activation.signed_int8[first.activation_offset + first.compact_k_count] = 1;
    if (!check(rmd::execute_rmd_stripe_ws(args, malformed, output) ==
                   rmd::RmdStatus::invalid_packet &&
               output.j_padded == unchanged.j_padded && output.values == unchanged.values,
               "explicit RMD WS validates packets before host rejection")) {
        return false;
    }
    rmd::RmdExecutionMetrics ws_metrics{};
    const rmd::RmdStatus ws = rmd::execute_rmd_stripe_ws(args, *packet, output,
                                                         &ws_metrics);
#if defined(__riscv)
    rmd::Correction route_ws = rmd::BlockScaledInt64Correction{};
    const auto route_ws_status = rmd::compose_rmd_output(*packet, output, route_ws);
    const auto * route_ws_values = integer_values(route_ws);
    std::printf("RMD_ORACLE single dense_direct=%lld packet_scalar=%lld ws=%lld\n",
                static_cast<long long>(route_direct.front()), static_cast<long long>(route_packet_values->front()),
                static_cast<long long>(route_ws_values == nullptr ? 0 : route_ws_values->front()));
    const bool ws_result = ws == rmd::RmdStatus::success && route_ws_status == rmd::RmdStatus::success &&
        route_ws_values != nullptr && *route_ws_values == *route_packet_values &&
        *route_ws_values == route_direct;
#else
    std::printf("RMD_ORACLE single dense_direct=%lld packet_scalar=%lld ws=unsupported\n",
                static_cast<long long>(route_direct.front()), static_cast<long long>(route_packet_values->front()));
    const bool ws_result = ws == rmd::RmdStatus::unsupported_route &&
        output.j_padded == unchanged.j_padded && output.values == unchanged.values;
#endif
    if (!check(ws_result, "explicit RMD WS matches reference or rejects unsupported host")) {
        return false;
    }
#if defined(GGML_GEMMINI_TESTING) && defined(__riscv)
    if (!check(route_metrics.ws_observations.empty() && ws_metrics.ws_observations.size() == 1 &&
               ws_metrics.ws_observations.front().lane_id == 1 &&
               ws_metrics.ws_observations.front().raw_value == 4 &&
               ws_metrics.ws_observations.front().block_scale == 1 &&
               ws_metrics.ws_observations.front().composed_value == 1024,
               "RMD WS observation records actual Gemmini call")) return false;
#elif defined(GGML_GEMMINI_TESTING)
    if (!check(route_metrics.ws_observations.empty() && ws_metrics.ws_observations.empty(),
               "host WS remains unsupported without observations")) return false;
#endif
#if defined(GGML_GEMMINI_TESTING)
    ggml_gemmini_args_t cancel_args{};
    std::array<elem_t, 4> cancel_weights = {127, 127, 2, 1};
    cancel_args.I = 1; cancel_args.J = 1; cancel_args.K = 4;
    cancel_args.B = cancel_weights.data(); cancel_args.sB = 1;
    cancel_args.weight_i8_scale_active = true; cancel_args.weight_scale = 1.0f;
    rmd::RmdStripeBuilder cancel_builder;
    cancel_builder.reset(0, 0, 1, 4, 1);
    if (!cancel_builder.add_residual(0, 0, 1) ||
        !cancel_builder.add_residual(0, 1, 1) ||
        !cancel_builder.add_residual(0, 2, 1) ||
        !cancel_builder.add_residual(0, 3, -256)) {
        return check(false, "RMD cancellation packet input accepted");
    }
    const auto cancel_packet = cancel_builder.finish();
    rmd::CompressedOutput cancel_output;
    rmd::RmdExecutionMetrics cancel_metrics{};
    const auto cancel_status = rmd::execute_rmd_stripe_reference(cancel_args, *cancel_packet,
                                                  cancel_output, &cancel_metrics);
    rmd::Correction cancel_composed = rmd::BlockScaledInt64Correction{};
    const auto cancel_compose_status = rmd::compose_rmd_output(*cancel_packet, cancel_output, cancel_composed);
    const auto * cancel_composed_values = integer_values(cancel_composed);
    std::printf("RMD_STAGE cancellation_packet_scalar status=%d compose=%d correction=%lld nonzero_count=%zu\n", static_cast<int>(cancel_status), static_cast<int>(cancel_compose_status), static_cast<long long>(cancel_composed_values == nullptr || cancel_composed_values->empty() ? 0 : cancel_composed_values->front()), cancel_composed_values == nullptr || cancel_composed_values->empty() ? size_t{0} : size_t{cancel_composed_values->front() != 0});
    std::vector<rmd::ReferenceResidual> cancel_events = {{0, 0, 1}, {0, 1, 1}, {0, 2, 1}, {0, 3, -256}};
    std::vector<rmd::OutputValue> cancel_direct;
    const auto cancel_direct_status = rmd::reference_direct_correction(cancel_args, 1, cancel_events, cancel_direct);
    if (!check(cancel_status == rmd::RmdStatus::success && cancel_compose_status == rmd::RmdStatus::success &&
               cancel_direct_status == rmd::RmdStatus::success &&
               cancel_direct == std::vector<rmd::OutputValue>{0} &&
               cancel_composed_values != nullptr && cancel_direct == *cancel_composed_values,
               "RMD cancellation reference correction")) return false;
#if defined(__riscv)
    rmd::CompressedOutput ws_cancel_output;
    rmd::RmdExecutionMetrics ws_cancel_metrics{};
    const auto ws_cancel_status = rmd::execute_rmd_stripe_ws(cancel_args, *cancel_packet,
                                                               ws_cancel_output, &ws_cancel_metrics);
    rmd::Correction ws_cancel_composed = rmd::BlockScaledInt64Correction{};
    const auto ws_cancel_compose_status = rmd::compose_rmd_output(*cancel_packet, ws_cancel_output,
                                                                    ws_cancel_composed);
    const auto * ws_cancel_values = integer_values(ws_cancel_composed);
    std::printf("RMD_STAGE ws raw_lanes=256,-1 correction=%lld nonzero_count=%zu\n",
                static_cast<long long>(ws_cancel_values == nullptr || ws_cancel_values->empty() ? 0 : ws_cancel_values->front()),
                ws_cancel_values == nullptr || ws_cancel_values->empty() ? size_t{0} : size_t{ws_cancel_values->front() != 0});
    std::printf("RMD_ORACLE cancellation dense_direct=%lld packet_scalar=%lld ws=%lld\n",
                static_cast<long long>(cancel_direct.front()), static_cast<long long>(cancel_composed_values->front()),
                static_cast<long long>(ws_cancel_values == nullptr ? 0 : ws_cancel_values->front()));
    if (!check(ws_cancel_status == rmd::RmdStatus::success &&
               ws_cancel_compose_status == rmd::RmdStatus::success && ws_cancel_values != nullptr &&
               *ws_cancel_values == *cancel_composed_values && *ws_cancel_values == cancel_direct &&
               !ws_cancel_values->empty() && ws_cancel_values->front() == 0,
               "RMD cancellation WS correction")) return false;
#else
    std::printf("RMD_ORACLE cancellation dense_direct=%lld packet_scalar=%lld ws=unsupported\n",
                static_cast<long long>(cancel_direct.front()), static_cast<long long>(cancel_composed_values->front()));
#endif
#endif
    return check(packet->blocks.size() == packet_before.blocks.size() &&
              std::memcmp(packet->blocks.data(), packet_before.blocks.data(),
                          packet->blocks.size() * sizeof(rmd::BlockDescriptor)) == 0 &&
              packet->k_indices == packet_before.k_indices &&
              packet->stacked_activation == packet_before.stacked_activation,
              "compact execution preserves packet bytes");
}

bool test_q8_srmd_software_ws_routing() {
    using residual::DirectExecutionMetrics;
    using residual::DirectStripeBuilder;
    using residual::ResidualEvent;

    constexpr size_t rows = 1;
    constexpr size_t columns = 1;
    constexpr size_t logical_k = 2 * QK8_0;
    constexpr size_t blocks_per_row = 2;
    const std::array<ResidualEvent, 4> events = {{
        {0, 0, 128}, {0, 1, -32768}, {0, QK8_0, -129}, {0, QK8_0 + 1, 32768},
    }};

    std::array<block_q8_h1, blocks_per_row> native_weights{};
    native_weights[0].qs[0] = 2;
    native_weights[0].qs[1] = 1;
    native_weights[0].c_b = 1;
    native_weights[0].R = 2;
    native_weights[0].s_rf = 1.0f;
    native_weights[1].qs[0] = -4;
    native_weights[1].qs[1] = -2;
    native_weights[1].c_b = 3;
    native_weights[1].R = 4;
    native_weights[1].s_rf = 1.0f;

    DirectStripeBuilder direct_builder;
    direct_builder.reset(17, 0, rows, logical_k, columns);
    rmd::RmdStripeBuilder packet_builder;
    packet_builder.reset(17, 0, rows, logical_k, columns);
    for (const ResidualEvent & event : events) {
        if (!direct_builder.add_residual(event.local_row, event.original_k, event.residual) ||
            !packet_builder.add_residual(event.local_row, event.original_k, event.residual)) {
            return check(false, "Q8 SRMD signed multi-block fixture accepted");
        }
    }
    const auto direct_payload = direct_builder.finish();
    const auto packet = packet_builder.finish();
    if (!check(direct_payload != nullptr && packet != nullptr && packet->blocks.size() == 2,
               "Q8 SRMD direct and compact fixtures built") ||
        !check(direct_payload->events == std::vector<ResidualEvent>(events.begin(), events.end()),
               "CPU payload preserves original INT32 residuals without radix conversion")) {
        return false;
    }

    auto make_native_args = [&] {
        ggml_gemmini_args_t args{};
        args.I = rows;
        args.J = columns;
        args.K = logical_k;
        args.weight_format = ggml_gemmini_args_t::im2p_weight_format_t::q8_h1;
        args.q8_h1_blocks = native_weights.data();
        args.q8_h1_block_count = native_weights.size();
        args.q8_h1_rows = columns;
        args.blocks_per_row = blocks_per_row;
        args.native_weight_bytes = native_weights.size() * sizeof(block_q8_h1);
        args.block_size_k = QK8_0;
        args.tiled_matmul_type = WS;
        args.full_C = true;
        return args;
    };

    ggml_gemmini_args_t cpu_args = make_native_args();
    cpu_args.residual_route = residual::ResidualRoute::cpu_direct;
    rmd::Correction direct_output = rmd::BlockScaledInt64Correction{};
    DirectExecutionMetrics direct_metrics{};
    if (!check(residual::execute_direct_stripe(cpu_args, *direct_payload, direct_output,
                                                &direct_metrics) == rmd::RmdStatus::success,
               "CPU Q8_H1 residual uses direct executor") ||
        !check(direct_metrics.event_count == events.size() && direct_metrics.call_count == 1,
               "CPU executes original INT32 residual events directly")) {
        return false;
    }
    const auto * direct_correction_ptr = integer_values(direct_output);
    if (!check(direct_correction_ptr != nullptr,
               "CPU Q8_H1 residual retains its integer correction domain")) return false;
    const auto & direct_correction = *direct_correction_ptr;

    auto verify_software_ws = [&](const char * route, ggml_gemmini_args_t args) {
        rmd::CompressedOutput compressed;
        rmd::RmdExecutionMetrics metrics{};
        const rmd::RmdStatus status = rmd::execute_rmd_stripe_ws(args, *packet, compressed, &metrics);
        if (status != rmd::RmdStatus::success) {
            std::fprintf(stderr,
                         "FAIL: %s WS must select compact SRMD software executor; status=%s\n",
                         route, rmd::rmd_status_message(status));
            return false;
        }
        if (!check(metrics.ws_call_count == 0,
                   "Q8 software WS route must not invoke hardware tiled_matmul") ||
            !check(metrics.packet_call_count == 1,
                   "Q8 software WS route executes one compact packet")) {
            return false;
        }

        for (size_t block_index = 0; block_index < packet->blocks.size(); ++block_index) {
            const rmd::BlockDescriptor & block = packet->blocks[block_index];
            const int64_t block_scale = block_index == 0 ? 3 : 7;
            for (size_t lane_position = 0; lane_position < block.active_lane_count; ++lane_position) {
                const uint8_t lane_id = block.lane_ids[lane_position];
                int64_t raw_lane = 0;
                for (const ResidualEvent & event : events) {
                    if (event.original_k / QK8_0 != block.block_id) continue;
                    rmd::BalancedDigits digits{};
                    if (!rmd::decompose_balanced_radix256(event.residual, digits)) return false;
                    const int8_t code = native_weights[block_index].qs[event.original_k % QK8_0];
                    raw_lane += static_cast<int64_t>(digits.digits[lane_id]) * code;
                }
                const size_t output_index = block.output_value_offset +
                    lane_position * block.lane_stride_values;
                if (!check(compressed.values[output_index] == raw_lane * block_scale,
                           "integer K-block scale is applied before radix/cross-block composition")) {
                    return false;
                }
            }
        }

        rmd::Correction composed = rmd::BlockScaledInt64Correction{};
        return check(rmd::compose_rmd_output(*packet, compressed, composed) ==
                         rmd::RmdStatus::success,
                     "Q8 software WS compressed output composes") &&
            check(integer_values_equal(composed, direct_correction),
                  "signed radix lanes and 128 carry boundaries match direct INT32 residuals");
    };

    if (!verify_software_ws("native Q8_H1", make_native_args())) {
        return false;
    }

    std::array<elem_t, logical_k> derived_weights{};
    derived_weights[0] = native_weights[0].qs[0];
    derived_weights[1] = native_weights[0].qs[1];
    derived_weights[QK8_0] = native_weights[1].qs[0];
    derived_weights[QK8_0 + 1] = native_weights[1].qs[1];
    const std::array<uint8_t, blocks_per_row> derived_codes = {1, 5};
    const std::array<float, columns> derived_s_rf = {1.0f};
    const std::array<uint16_t, columns> derived_r = {2};
    ggml_gemmini_args_t derived_args{};
    derived_args.I = rows;
    derived_args.J = columns;
    derived_args.K = logical_k;
    derived_args.B = derived_weights.data();
    derived_args.sB = logical_k;
    derived_args.transpose_B = true;
    derived_args.weight_format =
        ggml_gemmini_args_t::im2p_weight_format_t::q8_0_unpacked_to_h1;
    derived_args.c_b = derived_codes.data();
    derived_args.s_rf = derived_s_rf.data();
    derived_args.R = derived_r.data();
    derived_args.blocks_per_row = blocks_per_row;
    derived_args.blocks_J = columns;
    derived_args.block_size_k = QK8_0;
    derived_args.tiled_matmul_type = WS;
    derived_args.full_C = true;
    if (!verify_software_ws("Q8_0-derived Q8_H1", derived_args)) {
        return false;
    }

    elem_t tensor_weight = 5;
    ggml_gemmini_args_t tensor_args{};
    tensor_args.I = tensor_args.J = tensor_args.K = 1;
    tensor_args.B = &tensor_weight;
    tensor_args.sB = 1;
    tensor_args.weight_i8_scale_active = true;
    tensor_args.weight_scale = 1.0f;
    tensor_args.tiled_matmul_type = WS;
    rmd::RmdStripeBuilder tensor_builder;
    tensor_builder.reset(0, 0, 1, 1, 1);
    tensor_builder.add_residual(0, 0, 128);
    const auto tensor_packet = tensor_builder.finish();
    if (!check(tensor_packet != nullptr, "i8_tensor hardware-route packet built")) {
        return false;
    }
    rmd::CompressedOutput tensor_output;
    rmd::RmdExecutionMetrics tensor_metrics{};
    const rmd::RmdStatus tensor_status = rmd::execute_rmd_stripe_ws(
        tensor_args, *tensor_packet, tensor_output, &tensor_metrics);
#if defined(__riscv)
    return check(tensor_status == rmd::RmdStatus::success && tensor_metrics.ws_call_count != 0,
                 "i8_tensor WS residual route remains hardware tiled_matmul");
#else
    return check(tensor_status == rmd::RmdStatus::unsupported_route &&
                     tensor_metrics.packet_call_count == 0 && tensor_metrics.ws_call_count == 0,
                 "i8_tensor WS residual route remains hardware-only on host");
#endif
}

bool test_q8_hp1_srmd_software_ws_routing() {
    using residual::DirectStripeBuilder;
    using residual::ResidualEvent;

    constexpr size_t rows = 1;
    constexpr size_t columns = 1;
    constexpr size_t logical_k = 2 * QK8_HP;
    constexpr size_t blocks_per_row = 2;
    const std::array<ResidualEvent, 4> events = {{
        {0, 0, 128}, {0, 1, -32768}, {0, QK8_HP, -129}, {0, QK8_HP + 1, 32768},
    }};

    std::array<block_q8_hp1, blocks_per_row> weights{};
    weights[0].qs[0] = 2;
    weights[0].qs[1] = 1;
    weights[0].m = 1;
    weights[0].channel_scale = 0.25f;
    weights[1].qs[0] = -4;
    weights[1].qs[1] = -2;
    weights[1].m = 3;
    weights[1].channel_scale = 0.25f;

    DirectStripeBuilder direct_builder;
    direct_builder.reset(23, 0, rows, logical_k, columns);
    rmd::RmdStripeBuilder packet_builder;
    packet_builder.reset(23, 0, rows, logical_k, columns);
    for (const ResidualEvent & event : events) {
        if (!direct_builder.add_residual(event.local_row, event.original_k, event.residual) ||
            !packet_builder.add_residual(event.local_row, event.original_k, event.residual)) {
            return check(false, "Q8_HP1 SRMD signed multi-block fixture accepted");
        }
    }
    const auto direct_payload = direct_builder.finish();
    const auto packet = packet_builder.finish();
    if (!check(direct_payload != nullptr && packet != nullptr && packet->blocks.size() == 2,
               "Q8_HP1 direct and compact fixtures built")) {
        return false;
    }

    ggml_gemmini_args_t args{};
    args.I = rows;
    args.J = columns;
    args.K = logical_k;
    args.weight_format = ggml_gemmini_args_t::im2p_weight_format_t::q8_hp1;
    args.q8_hp1_blocks = weights.data();
    args.q8_hp1_block_count = weights.size();
    args.q8_hp1_blocks_per_row = blocks_per_row;
    args.blocks_per_row = blocks_per_row;
    args.native_weight_bytes = weights.size() * sizeof(block_q8_hp1);
    args.block_size_k = QK8_HP;
    args.tiled_matmul_type = WS;
    args.full_C = true;

    rmd::Correction direct_output = rmd::BlockScaledInt64Correction{};
    if (!check(residual::execute_direct_stripe(args, *direct_payload, direct_output) ==
                   rmd::RmdStatus::success,
               "CPU Q8_HP1 residual uses original INT32 direct executor")) {
        return false;
    }
    const auto * direct_correction_ptr = integer_values(direct_output);
    if (!check(direct_correction_ptr != nullptr,
               "CPU Q8_HP1 residual retains its integer correction domain")) return false;
    const auto & direct_correction = *direct_correction_ptr;

    rmd::CompressedOutput compressed;
    rmd::RmdExecutionMetrics metrics{};
    const rmd::RmdStatus status =
        rmd::execute_rmd_stripe_ws(args, *packet, compressed, &metrics);
    if (status != rmd::RmdStatus::success) {
        std::fprintf(stderr,
                     "FAIL: Q8_HP1 WS must select compact SRMD software executor; status=%s\n",
                     rmd::rmd_status_message(status));
        return false;
    }
    if (!check(metrics.ws_call_count == 0,
               "Q8_HP1 software WS route must not invoke hardware tiled_matmul") ||
        !check(metrics.packet_call_count == 1,
               "Q8_HP1 software WS route executes one compact packet")) {
        return false;
    }

    for (size_t block_index = 0; block_index < packet->blocks.size(); ++block_index) {
        const rmd::BlockDescriptor & block = packet->blocks[block_index];
        const int64_t block_scale = int64_t{1} << weights[block_index].m;
        for (size_t lane_position = 0; lane_position < block.active_lane_count; ++lane_position) {
            const uint8_t lane_id = block.lane_ids[lane_position];
            int64_t raw_lane = 0;
            for (const ResidualEvent & event : events) {
                if (event.original_k / QK8_HP != block.block_id) continue;
                rmd::BalancedDigits digits{};
                if (!rmd::decompose_balanced_radix256(event.residual, digits)) return false;
                raw_lane += static_cast<int64_t>(digits.digits[lane_id]) *
                    weights[block_index].qs[event.original_k % QK8_HP];
            }
            const size_t output_index = block.output_value_offset +
                lane_position * block.lane_stride_values;
            if (!check(compressed.values[output_index] == raw_lane * block_scale,
                       "Q8_HP1 exponent scale is applied before radix/block composition")) {
                return false;
            }
        }
    }

    rmd::Correction composed = rmd::BlockScaledInt64Correction{};
    if (!check(rmd::compose_rmd_output(*packet, compressed, composed) ==
                   rmd::RmdStatus::success,
               "Q8_HP1 software WS compressed output composes") ||
        !check(integer_values_equal(composed, direct_correction),
               "Q8_HP1 signed radix and integer exponent match direct correction")) {
        return false;
    }
    std::vector<rmd::OutputValue> reference;
    std::vector<rmd::ReferenceResidual> reference_residuals;
    reference_residuals.reserve(events.size());
    for (const ResidualEvent & event : events) {
        reference_residuals.push_back(
            { static_cast<uint32_t>(event.local_row),
              static_cast<uint32_t>(event.original_k),
              event.residual });
    }
    if (!check(rmd::reference_rmd_correction(
                   args, rows, reference_residuals, reference) == rmd::RmdStatus::success,
               "Q8_HP1 independent reference accepts native blocks") ||
        !check(integer_values_equal(composed, reference),
               "Q8_HP1 software WS matches independent reference correction")) {
        return false;
    }

    std::vector<float> merged(1, 0.0f);
    args.f_out = merged.data();
    auto & meta = args.act_quant.storage().emplace<quants::act::exsia::Meta>();
    meta.theta = { 1 };
    if (!check(rmd::merge_rmd_correction(args, *packet, composed) ==
                   rmd::RmdStatus::success,
               "Q8_HP1 correction merges with shared scales") ||
        !check(merged[0] == static_cast<float>(direct_correction[0]) * 0.25f * 2.0f,
               "Q8_HP1 channel and activation scales apply once at FP32 merge")) {
        return false;
    }

    auto invalid_weights = weights;
    invalid_weights[1].channel_scale = 0.5f;
    args.q8_hp1_blocks = invalid_weights.data();
    if (!check(rmd::merge_rmd_correction(args, *packet, composed) ==
                   rmd::RmdStatus::unsupported_route,
               "Q8_HP1 rejects inconsistent per-column channel scales")) {
        return false;
    }
    invalid_weights = weights;
    invalid_weights[1].m = 63;
    args.q8_hp1_blocks = invalid_weights.data();
    rmd::CompressedOutput invalid_output;
    if (!check(rmd::execute_rmd_stripe_ws(args, *packet, invalid_output, nullptr) ==
                   rmd::RmdStatus::overflow,
               "Q8_HP1 rejects unrepresentable exponent scale")) {
        return false;
    }
    invalid_weights[1].m = INT16_MIN;
    args.q8_hp1_blocks = invalid_weights.data();
    if (!check(rmd::execute_rmd_stripe_ws(args, *packet, invalid_output, nullptr) ==
                   rmd::RmdStatus::success,
               "Q8_HP1 zero-block sentinel executes safely")) {
        return false;
    }
    return true;
}

bool test_rmd_ws_contract_probe() {
#if !defined(__riscv)
    std::printf("RMD_RAW_WS supported=0 status=unsupported\n");
    return true;
#else
    constexpr size_t stride = DIM;
    constexpr elem_t sentinel = static_cast<elem_t>(-91);
    struct alignas(64) Physical {
        elem_t a[32 * DIM];
        elem_t b[DIM * DIM];
        alignas(64) acc_t c[32 * DIM];
        acc_t guard_before;
        acc_t guard_after;
    };
    struct Result { acc_t c0, c1, r16c0, r16c1; size_t changed, first; bool guards; };
    auto run = [sentinel, stride](Physical & p, size_t I, size_t tile_I, size_t J, size_t K) {
        std::fill(p.c, p.c + 32 * DIM, sentinel);
        p.guard_before = -37; p.guard_after = -73;
        asm volatile("" ::: "memory");
        tiled_matmul(I, J, K, p.a, p.b, nullptr, p.c, stride, stride, stride, stride,
                     1.0f, 1.0f, 1.0f, NO_ACTIVATION, ACC_SCALE_IDENTITY,
                     ACC_SCALE_IDENTITY, false, tile_I, 1, 1,
                     false, false, true, false, 0, WS);
        asm volatile("" ::: "memory");
        Result r{p.c[0], p.c[1], 0, 0, 0, 32 * DIM,
                 p.guard_before == -37 && p.guard_after == -73};
        if (I > 16) { r.r16c0 = p.c[16 * DIM]; r.r16c1 = p.c[16 * DIM + 1]; }
        for (size_t n = 0; n < I * DIM; ++n)
            if (p.c[n] != sentinel) { ++r.changed; r.first = std::min(r.first, n); }
        return r;
    };
    auto setup = [](Physical & p, size_t I) {
        std::fill(std::begin(p.a), std::end(p.a), elem_t{0});
        std::fill(std::begin(p.b), std::end(p.b), elem_t{0});
        p.a[0] = 1; p.a[1] = 2;
        if (I > 16) { p.a[16 * DIM] = 1; p.a[16 * DIM + 1] = 2; }
        p.b[0] = 3; p.b[1] = 7; p.b[DIM] = 5; p.b[DIM + 1] = 11;
    };
    bool all_ok = true;
    for (const auto & spec : {std::array<size_t, 4>{16,1,2,2}, {16,1,DIM,DIM},
                              {32,1,2,2}, {32,1,DIM,DIM}, {32,2,2,2}, {32,2,DIM,DIM}}) {
        Physical p{}; setup(p, spec[0]); const Result r = run(p, spec[0], spec[1], spec[2], spec[3]);
        const bool present = spec[0] > 16;
        std::printf("RMD_RAW_WS I=%zu tile_I=%zu J=%zu K=%zu row0_c0=%lld row0_c1=%lld row16_c0=%lld row16_c1=%lld row16_present=%d changed_count=%zu first_changed_index=%zu guards_ok=%d a_mod64=%zu b_mod64=%zu c_mod64=%zu\n",
            spec[0], spec[1], spec[2], spec[3], (long long)r.c0, (long long)r.c1,
            (long long)r.r16c0, (long long)r.r16c1, present ? 1 : 0, r.changed, r.first,
            r.guards ? 1 : 0, (uintptr_t)p.a % 64, (uintptr_t)p.b % 64, (uintptr_t)p.c % 64);
        all_ok = all_ok && r.c0 == 13 && r.c1 == 29 && (!present || (r.r16c0 == 13 && r.r16c1 == 29)) && r.guards;
    }
    const bool raw_ok = check(all_ok, "RMD WS raw I-tile grid");
    // Keep this fixture deliberately independent of the raw-grid sentinel probe above:
    // it mirrors execute_dense's dense WS setup and then calls the resolved Gemmini API.
    alignas(64) elem_t dense_a[2] = {1, 2};
    alignas(64) elem_t dense_b[4] = {1, 3, 2, 4}; // physical [J][K]
    alignas(64) float dense_out[2] = {-99.0f, -99.0f};
    ggml_gemmini_args_t dense_args{};
    dense_args.I = 1; dense_args.J = 2; dense_args.K = 2;
    dense_args.A.allocate(1, 2, GGML_GEMMINI_ACTIVATION_BITS);
    dense_args.A.set(0, 0, dense_a[0]);
    dense_args.A.set(0, 1, dense_a[1]);
    dense_args.B = dense_b; dense_args.f_out = dense_out;
    dense_args.sA = 2; dense_args.sB = 2; dense_args.stride_f_out = 2;
    dense_args.col_stride_f_out = 1; dense_args.weight_i8_scale_active = true;
    dense_args.weight_scale = 1.0f; dense_args.transpose_B = true;
    dense_args.tiled_matmul_type = WS;
    auto & dense_meta = dense_args.act_quant.storage().emplace<quants::act::exsia::Meta>();
    dense_meta.theta = { 0 };
    gemmini_set_tile_ws(&dense_args);
    dense_args.full_C = true; dense_args.low_D = false;
    tiled_matmul_auto_baseline(&dense_args, baseline_activation_quant_t::EXSIA,
                               baseline_weight_quant_t::TENSOR);
    std::printf("DENSE_CLONE helper_out=%g,%g expected=7,10 I=%zu J=%zu K=%zu B_s=%zu transpose_B=%d tile_I=%zu tile_J=%zu tile_K=%zu scale_B=%g scale_D=%g act=%d weightA=%u full_C=%d low_D=%d a_mod64=%zu b_mod64=%zu out_mod64=%zu\n",
                dense_out[0], dense_out[1], dense_args.I, dense_args.J, dense_args.K,
                dense_args.sB, dense_args.transpose_B ? 1 : 0, dense_args.tile_I,
                dense_args.tile_J, dense_args.tile_K, (double)dense_args.scale_B,
                (double)dense_args.scale_D, dense_args.act, dense_args.weightA,
                dense_args.full_C ? 1 : 0, dense_args.low_D ? 1 : 0,
                (uintptr_t)dense_a % 64, (uintptr_t)dense_b % 64, (uintptr_t)dense_out % 64);
    struct WsCallConfig {
        size_t I = 1, J = 2, K = 2, stride_A = 2, stride_B = 2, stride_D = 0, stride_C = 2;
        const void * D = nullptr; scale_t scale_A = 1, scale_B = 1;
        scale_acc_t scale_D = 1; int act = NO_ACTIVATION;
        acc_scale_t scale = ACC_SCALE_IDENTITY, bert_scale = ACC_SCALE_IDENTITY;
        bool repeating_bias = false; size_t tile_I = 1, tile_J = 1, tile_K = 1;
        bool transpose_A = false, transpose_B = true, full_C = true, low_D = false;
        uint8_t weightA = 0; tiled_matmul_type_t type = WS;
    };
    struct WsCallBuffer {
        acc_t acc_guard_before;
        alignas(64) acc_t acc[2];
        acc_t acc_guard_after;
        elem_t elem_guard_before;
        alignas(64) elem_t elem[2];
        elem_t elem_guard_after;
    };
    const acc_t expected0 = 7, expected1 = 10;
    const WsCallConfig base{1, 2, 2, 2, 2, 0, 2, nullptr, 1.0f,
        dense_args.scale_B, dense_args.scale_D, dense_args.act, dense_args.scale,
        dense_args.bert_scale, dense_args.repeating_bias, dense_args.tile_I,
        dense_args.tile_J, dense_args.tile_K, false, dense_args.transpose_B, true,
        dense_args.low_D, dense_args.weightA, WS};
    auto run_ws = [&](const char * name, const WsCallConfig & cfg, const elem_t * b,
                      WsCallBuffer & buf) {
        std::fill(std::begin(buf.acc), std::end(buf.acc), static_cast<acc_t>(-91));
        std::fill(std::begin(buf.elem), std::end(buf.elem), static_cast<elem_t>(-91));
        buf.acc_guard_before = -37; buf.acc_guard_after = -73;
        buf.elem_guard_before = -37; buf.elem_guard_after = -73;
        void * c = cfg.full_C ? static_cast<void *>(buf.acc) : static_cast<void *>(buf.elem);
        tiled_matmul(cfg.I, cfg.J, cfg.K, dense_a, b, cfg.D, c, cfg.stride_A,
                     cfg.stride_B, cfg.stride_D, cfg.stride_C, cfg.scale_A,
                     cfg.scale_B, cfg.scale_D, cfg.act, cfg.scale, cfg.bert_scale,
                     cfg.repeating_bias, cfg.tile_I, cfg.tile_J, cfg.tile_K,
                     cfg.transpose_A, cfg.transpose_B, cfg.full_C, cfg.low_D,
                     cfg.weightA, cfg.type);
        const acc_t c0 = cfg.full_C ? buf.acc[0] : static_cast<acc_t>(buf.elem[0]);
        const acc_t c1 = cfg.full_C ? buf.acc[1] : static_cast<acc_t>(buf.elem[1]);
        const int changed_count = (c0 != -91) + (c1 != -91);
        const int first_changed_index = c0 != -91 ? 0 : (c1 != -91 ? 1 : 2);
        std::printf("WS_CALL_DELTA name=%s expected=7,10 c0=%lld c1=%lld changed_count=%d first_changed_index=%d D=%s transpose_B=%d tile_I=%zu tile_J=%zu tile_K=%zu scales=%g,%g,%g act=%d weightA=%u full_C=%d low_D=%d c_mod64=%zu guards=%d\n",
                    name, (long long)c0, (long long)c1, changed_count, first_changed_index,
                    cfg.D ? "zero" : "null",
                    cfg.transpose_B ? 1 : 0, cfg.tile_I, cfg.tile_J, cfg.tile_K,
                    (double)cfg.scale_A, (double)cfg.scale_B, (double)cfg.scale_D,
                    cfg.act, cfg.weightA, cfg.full_C ? 1 : 0, cfg.low_D ? 1 : 0,
                    reinterpret_cast<uintptr_t>(c) % 64,
                    (cfg.full_C ? buf.acc_guard_before == -37 && buf.acc_guard_after == -73
                                 : buf.elem_guard_before == -37 && buf.elem_guard_after == -73) ? 1 : 0);
        return std::array<acc_t, 2>{c0, c1};
    };
    WsCallBuffer ws_buf{};
    const auto exact = run_ws("exact_clone", base, dense_b, ws_buf);
    WsCallConfig delta = base; alignas(64) acc_t zero_d[2] = {0, 0};
    delta.D = zero_d; run_ws("explicit_zero_D", delta, dense_b, ws_buf);
    alignas(64) elem_t kj_b[4] = {1, 2, 3, 4};
    delta = base; delta.transpose_B = false; run_ws("KJ_representation", delta, kj_b, ws_buf);
    delta = base; delta.tile_I = delta.tile_J = delta.tile_K = 1;
    run_ws("fixed_tiles_baseline_equal", delta, dense_b, ws_buf);
    run_ws("identity_settings_baseline_equal", base, dense_b, ws_buf);
    run_ws("weightA_zero_baseline_equal", base, dense_b, ws_buf);
    for (const auto & d : {false, true}) { delta = base; delta.D = d ? static_cast<const void *>(zero_d) : nullptr; delta.full_C = true; run_ws(d ? "zero_D_full_acc_C" : "null_full_acc_C", delta, dense_b, ws_buf); }
    for (const auto & d : {false, true}) { delta = base; delta.D = d ? static_cast<const void *>(zero_d) : nullptr; delta.full_C = false; run_ws(d ? "zero_D_narrow_elem_C" : "null_narrow_elem_C", delta, dense_b, ws_buf); }
    const bool dense_ok = check(dense_out[0] == 7.0f && dense_out[1] == 10.0f,
                                "dense WS clone output") &&
        check(exact[0] == expected0 && exact[1] == expected1, "exact WS clone output");
    return raw_ok && dense_ok;
#endif
}

bool test_rmd_cpu_direct_parity() {
    constexpr size_t rows = 17;
    constexpr size_t columns = 3;
    constexpr size_t logical_k = 65;
    constexpr size_t native_block_size = QK8_0;
    constexpr size_t native_blocks_per_row =
        (logical_k + native_block_size - 1) / native_block_size;

    std::vector<int32_t> residuals(rows * logical_k, 0);
    residuals[0 * logical_k + 0] = 128;
    residuals[0 * logical_k + 31] = -129;
    residuals[0 * logical_k + 32] = 65536;
    residuals[0 * logical_k + 64] = 256;
    residuals[1 * logical_k + 15] = -256;
    residuals[1 * logical_k + 47] = rmd::kSigned21Max;
    residuals[1 * logical_k + 63] = 129;
    residuals[16 * logical_k + 1] = -65536;
    residuals[16 * logical_k + 33] = rmd::kSigned21Min;

    std::vector<elem_t> baseline_activation(rows * logical_k);
    for (size_t row = 0; row < rows; ++row) {
        for (size_t k = 0; k < logical_k; ++k) {
            baseline_activation[row * logical_k + k] =
                static_cast<int8_t>(static_cast<int>((row * 13 + k * 5) % 255) - 127);
        }
    }

    std::vector<block_q8_h1> weights(columns * native_blocks_per_row);
    for (size_t j = 0; j < columns; ++j) {
        for (size_t block_id = 0; block_id < native_blocks_per_row; ++block_id) {
            block_q8_h1 & block = weights[j * native_blocks_per_row + block_id];
            for (size_t local_k = 0; local_k < native_block_size; ++local_k) {
                const int value = static_cast<int>((j + 1) * 17 + block_id * 29 + local_k * 7);
                block.qs[local_k] = static_cast<int8_t>(value % 255 - 127);
            }
            block.c_b = static_cast<uint8_t>(2 + j + 3 * block_id);
            block.s_rf = 1.0f;
            block.R = static_cast<uint16_t>(5 + j);
        }
    }

    ggml_gemmini_args_t args{};
    args.I = rows;
    args.J = columns;
    args.K = logical_k;
    args.weight_format = ggml_gemmini_args_t::im2p_weight_format_t::q8_h1;
    args.q8_h1_blocks = weights.data();
    args.q8_h1_block_count = weights.size();
    args.q8_h1_rows = columns;
    args.blocks_per_row = native_blocks_per_row;
    args.native_weight_bytes = weights.size() * sizeof(block_q8_h1);
    args.block_size_k = native_block_size;
    args.tiled_matmul_type = CPU;
    auto & meta = args.act_quant.storage().emplace<quants::act::exsia::Meta>();
    meta.theta = { 0 };

    rmd::RmdStripeBuilder builder;
    builder.reset(0, 0, rows, logical_k, columns);
    for (size_t row = 0; row < rows; ++row) {
        for (size_t k = 0; k < logical_k; ++k) {
            const int32_t residual = residuals[row * logical_k + k];
            if (residual != 0 && !builder.add_residual(row, k, residual)) {
                return check(false, "RMD direct-parity residual accepted");
            }
        }
    }
    const rmd::StripePacketHandle packet = builder.finish();
    if (!check(packet != nullptr, "RMD direct-parity packet built")) {
        return false;
    }

    rmd::CompressedOutput compressed;
    rmd::RmdExecutionMetrics metrics{};
    const rmd::RmdStatus execution_status =
        rmd::execute_rmd_stripe_reference(args, *packet, compressed, &metrics);
    if (execution_status != rmd::RmdStatus::success) {
        std::fprintf(stderr, "FAIL: RMD direct-parity CPU execution: %s\n",
                     rmd::rmd_status_message(execution_status));
        return false;
    }
    size_t expected_calls = 0;
    size_t expected_i_tiles = 0;
    const size_t j_tiles = (packet->logical_j + rmd::kArrayDim - 1) / rmd::kArrayDim;
    for (const rmd::BlockDescriptor & block : packet->blocks) {
        const size_t calls = j_tiles * (block.padded_k_count / rmd::kArrayDim);
        expected_calls += calls;
        expected_i_tiles += calls * block.active_lane_count *
            (block.rows_padded / rmd::kArrayDim);
    }
    if (!check(expected_i_tiles > expected_calls,
               "RMD reuse fixture spans multiple stacked I tiles") ||
        !check(metrics.matmul_call_count == expected_calls,
               "RMD executor coalesces lane/M tiles into one weight-loading call") ||
        !check(metrics.baseline_stacked_i_tile_count == expected_i_tiles,
               "RMD executor reports baseline stacked I tiles") ||
        !check(metrics.stacked_i_tile_count <= expected_i_tiles,
               "RMD lane partition never increases stacked I tiles") ||
        !check(metrics.weight_values_gathered ==
                   metrics.weight_baseline_address_resolutions &&
                   metrics.weight_address_resolutions <
                       metrics.weight_baseline_address_resolutions,
               "RMD executor reports reduced production weight addressing")) {
        return false;
    }
    std::printf("RMD stacked schedule: matmul_calls=%zu lane_groups=%zu "
                "stacked_i_tiles=%zu/%zu avoided_B_loads=%zu\n",
                metrics.matmul_call_count, metrics.lane_group_count,
                metrics.stacked_i_tile_count, metrics.baseline_stacked_i_tile_count,
                metrics.stacked_i_tile_count - metrics.matmul_call_count);
    rmd::Correction actual_correction = rmd::BlockScaledInt64Correction{};
    if (!check(rmd::compose_rmd_output(*packet, compressed, actual_correction) == rmd::RmdStatus::success,
               "RMD direct-parity composition")) {
        return false;
    }
    const auto * actual_values = integer_values(actual_correction);
    if (!check(actual_values != nullptr, "RMD direct-parity composition retains integer domain")) {
        return false;
    }
    const auto & actual = *actual_values;

    std::vector<int64_t> expected(rows * columns, 0);
    std::vector<int64_t> baseline(rows * columns, 0);
    std::vector<int64_t> direct_full(rows * columns, 0);
    for (size_t row = 0; row < rows; ++row) {
        for (size_t j = 0; j < columns; ++j) {
            for (size_t k = 0; k < logical_k; ++k) {
                const int32_t residual = residuals[row * logical_k + k];
                const block_q8_h1 & block =
                    weights[j * native_blocks_per_row + k / native_block_size];
                const int64_t scaled_weight = static_cast<int64_t>(block.qs[k % native_block_size]) *
                    static_cast<int64_t>(static_cast<uint64_t>(block.c_b) + block.R);
                const size_t index = row * columns + j;
                expected[index] += static_cast<int64_t>(residual) * scaled_weight;
                baseline[index] += static_cast<int64_t>(baseline_activation[row * logical_k + k]) *
                    scaled_weight;
                direct_full[index] +=
                    (static_cast<int64_t>(baseline_activation[row * logical_k + k]) + residual) *
                    scaled_weight;
            }
        }
    }

    bool exact = actual == expected;
    std::printf("RMD CPU exact parity: M=%zu J=%zu K=%zu rmd_block=%zu\n",
                rows, columns, logical_k, rmd::kBlockSize);
    for (size_t row = 0; row < rows; ++row) {
        for (size_t j = 0; j < columns; ++j) {
            const size_t index = row * columns + j;
            const int64_t compensated_full = baseline[index] + actual[index];
            std::printf("  C[%zu,%zu] residual_direct=%lld compensation=%lld "
                        "baseline=%lld direct_full=%lld baseline+comp=%lld\n", row, j,
                        static_cast<long long>(expected[index]),
                        static_cast<long long>(actual[index]),
                        static_cast<long long>(baseline[index]),
                        static_cast<long long>(direct_full[index]),
                        static_cast<long long>(compensated_full));
            exact = exact && compensated_full == direct_full[index];
        }
    }
    if (!check(exact, "direct matmul equals baseline plus RMD CPU compensation")) {
        return false;
    }

    std::vector<float> merged(rows * columns, 0.0f);
    args.f_out = merged.data();
    args.stride_f_out = columns;
    args.col_stride_f_out = 1;
    if (!check(rmd::merge_rmd_correction(args, *packet, actual_correction) == rmd::RmdStatus::success,
               "RMD direct-parity merge")) {
        return false;
    }
    for (size_t index = 0; index < expected.size(); ++index) {
        const int64_t saturated = std::clamp(
            expected[index],
            static_cast<int64_t>(std::numeric_limits<int32_t>::min()),
            static_cast<int64_t>(std::numeric_limits<int32_t>::max()));
        if (!check(merged[index] == static_cast<float>(saturated),
                   "merged RMD correction saturates after complete composition")) {
            return false;
        }
    }

#if defined(__riscv)
    rmd::CompressedOutput ws_compressed;
    const rmd::RmdStatus ws_execution_status =
        rmd::execute_rmd_stripe_ws(args, *packet, ws_compressed);
    rmd::Correction ws_actual = rmd::BlockScaledInt64Correction{};
    const rmd::RmdStatus ws_compose_status =
        rmd::compose_rmd_output(*packet, ws_compressed, ws_actual);
    const auto * ws_actual_values = integer_values(ws_actual);
    std::vector<float> ws_merged(rows * columns);
    for (size_t index = 0; index < ws_merged.size(); ++index) {
        ws_merged[index] = static_cast<float>(baseline[index]);
    }
    args.f_out = ws_merged.data();
    const rmd::RmdStatus ws_merge_status =
        rmd::merge_rmd_correction(args, *packet, ws_actual);
    size_t mismatch_count = 0;
    size_t first_row = rows, first_column = columns;
    float first_actual = 0.0f, first_expected = 0.0f;
    if (ws_execution_status == rmd::RmdStatus::success &&
        ws_compose_status == rmd::RmdStatus::success &&
        ws_merge_status == rmd::RmdStatus::success && ws_actual_values != nullptr &&
        ws_actual_values->size() == expected.size()) {
        for (size_t row = 0; row < rows; ++row) {
            for (size_t j = 0; j < columns; ++j) {
                const size_t index = row * columns + j;
                const int64_t saturated = std::clamp(
                    expected[index],
                    static_cast<int64_t>(std::numeric_limits<int32_t>::min()),
                    static_cast<int64_t>(std::numeric_limits<int32_t>::max()));
                const float wanted = static_cast<float>(baseline[index]) +
                    static_cast<float>(saturated);
                if (ws_merged[index] != wanted) {
                    ++mismatch_count;
                    if (first_row == rows) {
                        first_row = row;
                        first_column = j;
                        first_actual = ws_merged[index];
                        first_expected = wanted;
                    }
                }
            }
        }
    } else {
        mismatch_count = rows * columns;
    }
    std::printf("RMD_PLACEMENT mismatch_count=%zu first_row=%zu first_column=%zu first_actual=%g first_expected=%g\n",
                mismatch_count, first_row, first_column, first_actual, first_expected);
    if (!check(mismatch_count == 0, "RMD WS full output placement parity")) {
        return false;
    }
#endif

    merged.assign(rows * columns, 7.0f);
    const std::vector<float> unchanged = merged;
    auto & invalid_meta = args.act_quant.storage().emplace<quants::act::token::Meta>();
    invalid_meta.scales.assign(rows, 1.0f);
    invalid_meta.scales[1] = std::numeric_limits<float>::quiet_NaN();
    if (!check(rmd::merge_rmd_correction(args, *packet, actual_correction) ==
                   rmd::RmdStatus::invalid_arguments && merged == unchanged,
               "RMD merge failure preserves caller output")) {
        return false;
    }
    return true;
}

bool test_direct_cpu_executor() {
    using residual::DirectStripeBuilder;
    using residual::DirectStripePayload;
    using residual::DirectStripePayloadHandle;
    using residual::ResidualEvent;

    auto build_payload = [](size_t rows, size_t k_count, size_t j_count,
                            const std::vector<ResidualEvent> & events) {
        DirectStripeBuilder builder;
        builder.reset(3, 7, rows, k_count, j_count);
        for (const ResidualEvent & event : events) {
            if (!builder.add_residual(event.local_row, event.original_k, event.residual))
                return DirectStripePayloadHandle{};
        }
        return builder.finish();
    };
    auto reference = [](const ggml_gemmini_args_t & args,
                        const DirectStripePayload & payload,
                        std::vector<rmd::OutputValue> & output) {
        std::vector<rmd::ReferenceResidual> events;
        for (const ResidualEvent & event : payload.events) {
            events.push_back({static_cast<uint32_t>(event.local_row),
                              static_cast<uint32_t>(event.original_k), event.residual});
        }
        return rmd::reference_direct_correction(args, payload.row_count, events, output);
    };

    constexpr size_t rows = 4, columns = 19, logical_k = 65, blocks_per_row = 3;
    std::vector<block_q8_h1> native_weights(columns * blocks_per_row);
    for (size_t j = 0; j < columns; ++j) {
        for (size_t block_id = 0; block_id < blocks_per_row; ++block_id) {
            block_q8_h1 & block = native_weights[j * blocks_per_row + block_id];
            for (size_t k = 0; k < QK8_0; ++k)
                block.qs[k] = static_cast<int8_t>((j * 19 + block_id * 31 + k * 7) % 255 - 127);
            block.qs[1] = block.qs[0];
            block.c_b = static_cast<uint8_t>(2 + j + block_id);
            block.R = static_cast<uint16_t>(5 + block_id * 3);
            block.s_rf = 1.0f;
        }
    }
    const auto native_payload = build_payload(rows, logical_k, columns, {
        {0, 0, std::numeric_limits<int32_t>::max()},
        {0, 1, -std::numeric_limits<int32_t>::max()},
        {0, 31, 129}, {0, 32, -65537}, {0, 64, 16777217},
        {1, 32, 256}, {2, 32, -257}, {3, 7, -16777217}, {3, 64, 65536},
    });
    if (!check(native_payload != nullptr, "direct native sparse payload built")) return false;

    ggml_gemmini_args_t native_args{};
    native_args.I = rows; native_args.J = columns; native_args.K = logical_k;
    native_args.weight_format = ggml_gemmini_args_t::im2p_weight_format_t::q8_h1;
    native_args.q8_h1_blocks = native_weights.data();
    native_args.q8_h1_block_count = native_weights.size();
    native_args.q8_h1_rows = columns;
    native_args.blocks_per_row = blocks_per_row;
    native_args.native_weight_bytes = native_weights.size() * sizeof(block_q8_h1);
    native_args.block_size_k = QK8_0;
    std::vector<rmd::OutputValue> native_expected;
    rmd::Correction native_actual = rmd::BlockScaledInt64Correction{{91, 92, 93}};
    residual::DirectExecutionMetrics native_metrics{};
    if (!check(reference(native_args, *native_payload, native_expected) == rmd::RmdStatus::success,
               "direct native reference succeeds")) {
        return false;
    }
    quants::wreader::test_reset_weight_reader_counters();
    if (
        !check(residual::execute_direct_stripe(
                   native_args, *native_payload, native_actual, &native_metrics) ==
                   rmd::RmdStatus::success,
               "direct native execution succeeds") ||
        !check(integer_values_equal(native_actual, native_expected),
               "direct sparse/decode/reused-K/multi-block/tail/cancellation parity") ||
        !check(quants::wreader::test_weight_reader_storage_validations() == 1,
               "direct executor validates immutable weight storage once per call") ||
        !check(native_metrics.native_q8_values ==
                   native_payload->events.size() * native_args.J,
               "direct executor reports native Q8 values through shared reader") ||
        !check(native_metrics.j_tile_count == (native_args.J + 15) / 16,
               "direct executor publishes independent J tiles for parallel service")) return false;

    constexpr size_t dense_rows = 2, dense_columns = 5, dense_k = 37;
    std::vector<elem_t> dense_weights(dense_k * dense_columns);
    for (size_t k = 0; k < dense_k; ++k)
        for (size_t j = 0; j < dense_columns; ++j)
            dense_weights[k * dense_columns + j] = static_cast<elem_t>((k * 11 + j * 17) % 255 - 127);
    const auto dense_payload = build_payload(dense_rows, dense_k, dense_columns,
        {{0, 0, 128}, {0, 31, -129}, {0, 32, 65536}, {1, 9, -16777217}, {1, 36, 257}});
    ggml_gemmini_args_t dense_args{};
    dense_args.I = dense_rows; dense_args.J = dense_columns; dense_args.K = dense_k;
    dense_args.B = dense_weights.data(); dense_args.sB = dense_columns;
    dense_args.weight_i8_scale_active = true; dense_args.weight_scale = 0.25f;
    std::vector<rmd::OutputValue> dense_expected;
    rmd::Correction dense_actual = rmd::BlockScaledInt64Correction{{-7}};
    if (!check(dense_payload != nullptr, "direct dense payload built") ||
        !check(reference(dense_args, *dense_payload, dense_expected) == rmd::RmdStatus::success,
               "direct dense reference succeeds") ||
        !check(residual::execute_direct_stripe(dense_args, *dense_payload, dense_actual) ==
                   rmd::RmdStatus::success && integer_values_equal(dense_actual, dense_expected),
               "direct dense route parity")) return false;

    const std::vector<rmd::OutputValue> sentinel = {11, 22, 33};
    rmd::Correction failed = rmd::BlockScaledInt64Correction{sentinel};
    DirectStripePayload malformed = *dense_payload;
    std::swap(malformed.events[0], malformed.events[1]);
    if (!check(residual::execute_direct_stripe(dense_args, malformed, failed) ==
                   rmd::RmdStatus::invalid_packet && integer_values_equal(failed, sentinel),
               "direct invalid payload fails atomically")) return false;
    ggml_gemmini_args_t unsupported_args = dense_args;
    unsupported_args.weight_format = ggml_gemmini_args_t::im2p_weight_format_t::q8_h0;
    if (!check(residual::execute_direct_stripe(unsupported_args, *dense_payload, failed) ==
                   rmd::RmdStatus::unsupported_route && integer_values_equal(failed, sentinel),
               "direct unsupported route fails atomically")) return false;

    DirectStripePayload overflow_shape = *dense_payload;
    overflow_shape.row_begin = 0;
    overflow_shape.row_count = std::numeric_limits<size_t>::max() / 2 + 1;
    overflow_shape.logical_j = 3;
    ggml_gemmini_args_t overflow_shape_args = dense_args; overflow_shape_args.J = 3;
    if (!check(residual::execute_direct_stripe(overflow_shape_args, overflow_shape, failed) ==
                   rmd::RmdStatus::overflow && integer_values_equal(failed, sentinel),
               "direct output shape overflow fails atomically")) return false;
    DirectStripePayload allocation_shape = *dense_payload;
    allocation_shape.row_count = std::numeric_limits<size_t>::max() / 8;
    allocation_shape.logical_j = 2;
    ggml_gemmini_args_t allocation_args = dense_args; allocation_args.J = 2;
    if (!check(residual::execute_direct_stripe(allocation_args, allocation_shape, failed) ==
                   rmd::RmdStatus::allocation_failure && integer_values_equal(failed, sentinel),
               "direct impossible allocation fails atomically")) return false;

    constexpr size_t overflow_k = 17 * QK8_0;
    std::vector<block_q8_h1> overflow_weights(17);
    for (block_q8_h1 & block : overflow_weights) {
        std::fill(std::begin(block.qs), std::end(block.qs), static_cast<int8_t>(127));
        block.c_b = std::numeric_limits<uint8_t>::max();
        block.R = std::numeric_limits<uint16_t>::max(); block.s_rf = 1.0f;
    }
    std::vector<ResidualEvent> overflow_events;
    for (size_t k = 0; k < overflow_k; ++k)
        overflow_events.push_back({0, k, std::numeric_limits<int32_t>::max()});
    const auto overflow_payload = build_payload(1, overflow_k, 1, overflow_events);
    ggml_gemmini_args_t overflow_args{};
    overflow_args.I = overflow_args.J = 1; overflow_args.K = overflow_k;
    overflow_args.weight_format = ggml_gemmini_args_t::im2p_weight_format_t::q8_h1;
    overflow_args.q8_h1_blocks = overflow_weights.data();
    overflow_args.q8_h1_block_count = overflow_weights.size();
    overflow_args.q8_h1_rows = 1; overflow_args.blocks_per_row = overflow_weights.size();
    overflow_args.native_weight_bytes = overflow_weights.size() * sizeof(block_q8_h1);
    overflow_args.block_size_k = QK8_0;
    return check(overflow_payload != nullptr, "direct overflow payload built") &&
        check(residual::execute_direct_stripe(overflow_args, *overflow_payload, failed) ==
                  rmd::RmdStatus::overflow && integer_values_equal(failed, sentinel),
              "direct arithmetic overflow fails atomically");
}

bool test_rmd_lane_partition() {
    constexpr size_t logical_k = 32;
    std::vector<int32_t> residuals(logical_k, 0);
    for (size_t k = 0; k < 10; ++k) {
        residuals[k] = 257;
        residuals[16 + k] = 65536;
    }

    block_q8_h1 weights{};
    for (size_t k = 0; k < logical_k; ++k) {
        weights.qs[k] = static_cast<int8_t>(k + 1);
    }
    weights.c_b = 1;
    weights.s_rf = 1.0f;
    weights.R = 0;

    ggml_gemmini_args_t args{};
    args.I = 1;
    args.J = 1;
    args.K = logical_k;
    args.weight_format = ggml_gemmini_args_t::im2p_weight_format_t::q8_h1;
    args.q8_h1_blocks = &weights;
    args.q8_h1_block_count = 1;
    args.q8_h1_rows = 1;
    args.blocks_per_row = 1;
    args.native_weight_bytes = sizeof(weights);
    args.block_size_k = logical_k;
    args.tiled_matmul_type = CPU;

    rmd::RmdStripeBuilder builder;
    builder.reset(0, 0, 1, logical_k, 1);
    for (size_t k = 0; k < logical_k; ++k) {
        if (residuals[k] != 0 && !builder.add_residual(0, k, residuals[k])) {
            return check(false, "RMD lane-partition residual accepted");
        }
    }
    const rmd::StripePacketHandle packet = builder.finish();
    if (!check(packet != nullptr, "RMD lane-partition packet built")) {
        return false;
    }

    rmd::CompressedOutput compressed;
    rmd::RmdExecutionMetrics metrics{};
    if (!check(rmd::execute_rmd_stripe_reference(args, *packet, compressed, &metrics) ==
                   rmd::RmdStatus::success,
               "RMD lane-partition execution")) {
        return false;
    }
    rmd::Correction actual = rmd::BlockScaledInt64Correction{};
    if (!check(rmd::compose_rmd_output(*packet, compressed, actual) ==
                   rmd::RmdStatus::success,
               "RMD lane-partition composition")) {
        return false;
    }

    int64_t expected = 0;
    for (size_t k = 0; k < logical_k; ++k) {
        expected += static_cast<int64_t>(residuals[k]) * weights.qs[k];
    }
    constexpr size_t expected_k_tiles = (logical_k + DIM - 1) / DIM;
    return check(integer_values_equal(actual, std::vector<rmd::OutputValue>{expected}),
                  "RMD lane partition preserves exact output") &&
        check(metrics.matmul_call_count == expected_k_tiles,
               "RMD lane partition preserves DIM-aware B-load count") &&
        check(metrics.active_lanes == 3 &&
                  metrics.lane_group_count == expected_k_tiles,
               "RMD lane partition uses the minimal DIM-aware lane groups") &&
        check(metrics.baseline_stacked_i_tile_count == 3 * expected_k_tiles &&
                   metrics.stacked_i_tile_count == 3,
               "RMD lane partition preserves optimal stacked I-by-K tiles");
}

bool test_rmd_weight_gather() {
    constexpr size_t logical_k = 65;
    constexpr size_t columns = 3;
    constexpr size_t blocks_per_row = 3;
    std::vector<block_q8_h1> h1(columns * blocks_per_row);
    for (size_t j = 0; j < columns; ++j) {
        for (size_t block = 0; block < blocks_per_row; ++block) {
            for (size_t k = 0; k < QK8_0; ++k) {
                h1[j * blocks_per_row + block].qs[k] =
                    static_cast<int8_t>(j * 37 + block * 11 + k - 64);
            }
        }
    }

    ggml_gemmini_args_t args{};
    args.J = columns;
    args.K = logical_k;
    args.weight_format = ggml_gemmini_args_t::im2p_weight_format_t::q8_h1;
    args.q8_h1_blocks = h1.data();
    args.q8_h1_block_count = h1.size();
    args.q8_h1_rows = columns;
    args.blocks_per_row = blocks_per_row;
    args.native_weight_bytes = h1.size() * sizeof(block_q8_h1);
    args.block_size_k = QK8_0;

    const std::array<uint16_t, 3> local_k = { 0, 15, 31 };
    std::array<elem_t, rmd::kArrayDim * rmd::kArrayDim> tile{};
    tile.fill(0x55);
    rmd::RmdExecutionMetrics metrics{};
    if (!check(rmd::gather_weight_tile_for_test(
                   args, 1, local_k.data(), local_k.size(), 0, columns,
                   tile.data(), rmd::kArrayDim, &metrics) == rmd::RmdStatus::success,
               "native H1 gather succeeds")) {
        return false;
    }
    for (size_t k = 0; k < local_k.size(); ++k) {
        for (size_t j = 0; j < columns; ++j) {
            const int8_t expected = h1[j * blocks_per_row + 1].qs[local_k[k]];
            if (!check(tile[k * rmd::kArrayDim + j] == expected,
                       "native H1 gather matches scalar layout")) {
                return false;
            }
        }
    }
    if (!check(metrics.weight_values_gathered == local_k.size() * columns &&
               metrics.weight_baseline_address_resolutions == local_k.size() * columns &&
               metrics.weight_address_resolutions == columns,
               "native H1 gather reports row-base resolution counts")) {
        return false;
    }

    const std::array<std::pair<uint32_t, uint16_t>, 3> h1_boundaries = {
        std::pair<uint32_t, uint16_t>{0, 31},
        std::pair<uint32_t, uint16_t>{1, 0},
        std::pair<uint32_t, uint16_t>{2, 0},
    };
    for (const auto & [block_id, block_local_k] : h1_boundaries) {
        tile.fill(0x55);
        if (!check(rmd::gather_weight_tile_for_test(
                       args, block_id, &block_local_k, 1, 0, 1,
                       tile.data(), rmd::kArrayDim) == rmd::RmdStatus::success &&
                       tile[0] == h1[block_id].qs[block_local_k],
                   "native H1 gather preserves K=31/32/64 boundaries")) {
            return false;
        }
    }

    const auto unchanged_tile = tile;
    const auto unchanged_metrics = metrics;
    if (!check(rmd::gather_weight_tile_for_test(
                   args, 3, local_k.data(), local_k.size(), 0, columns,
                   tile.data(), rmd::kArrayDim, &metrics) == rmd::RmdStatus::execution_failed &&
               tile == unchanged_tile &&
               metrics.weight_values_gathered == unchanged_metrics.weight_values_gathered &&
               metrics.weight_baseline_address_resolutions ==
                   unchanged_metrics.weight_baseline_address_resolutions &&
               metrics.weight_address_resolutions == unchanged_metrics.weight_address_resolutions,
               "failed H1 gather preserves destination and metrics")) {
        return false;
    }

    constexpr size_t dense_j = 5;
    constexpr size_t dense_k = 40;
    constexpr size_t jxk_stride = dense_k + 7;
    constexpr size_t kxj_stride = dense_j + 9;
    std::vector<elem_t> dense_jxk(dense_j * jxk_stride, 0);
    std::vector<elem_t> dense_kxj(dense_k * kxj_stride, 0);
    for (size_t j = 0; j < dense_j; ++j) {
        for (size_t k = 0; k < dense_k; ++k) {
            const int8_t value = static_cast<int8_t>(j * 17 + k - 63);
            dense_jxk[j * jxk_stride + k] = value;
            dense_kxj[k * kxj_stride + j] = value;
        }
    }
    const std::array<uint16_t, 3> dense_local_k = { 0, 3, 7 };

    args = {};
    args.J = dense_j;
    args.K = dense_k;
    args.B = dense_jxk.data();
    args.sB = jxk_stride;
    args.transpose_B = true;
    args.weight_i8_scale_active = true;
    args.weight_scale = 1.0f;
    tile.fill(0x55);
    metrics = {};
    if (!check(rmd::gather_weight_tile_for_test(
                   args, 1, dense_local_k.data(), dense_local_k.size(), 1, 3,
                   tile.data(), rmd::kArrayDim, &metrics) == rmd::RmdStatus::success,
               "dense JxK gather succeeds")) {
        return false;
    }
    for (size_t k = 0; k < dense_local_k.size(); ++k) {
        const size_t global_k = QK8_0 + dense_local_k[k];
        for (size_t col = 0; col < 3; ++col) {
            if (!check(tile[k * rmd::kArrayDim + col] ==
                           dense_jxk[(1 + col) * jxk_stride + global_k],
                       "dense JxK gather preserves padded stride")) {
                return false;
            }
        }
    }
    if (!check(metrics.weight_address_resolutions == 3,
               "dense JxK resolves once per valid column")) {
        return false;
    }

    const auto valid_jxk_tile = tile;
    const auto valid_jxk_metrics = metrics;
    args.sB = dense_k - 1;
    if (!check(rmd::gather_weight_tile_for_test(
                   args, 1, dense_local_k.data(), dense_local_k.size(), 1, 3,
                   tile.data(), rmd::kArrayDim, &metrics) == rmd::RmdStatus::execution_failed &&
                   tile == valid_jxk_tile &&
                   metrics.weight_values_gathered == valid_jxk_metrics.weight_values_gathered &&
                   metrics.weight_address_resolutions == valid_jxk_metrics.weight_address_resolutions,
               "invalid dense stride preserves destination and metrics")) {
        return false;
    }

    args.B = dense_kxj.data();
    args.sB = kxj_stride;
    args.transpose_B = false;
    tile.fill(0x55);
    metrics = {};
    if (!check(rmd::gather_weight_tile_for_test(
                   args, 1, dense_local_k.data(), dense_local_k.size(), 1, 3,
                   tile.data(), rmd::kArrayDim, &metrics) == rmd::RmdStatus::success,
               "dense KxJ gather succeeds")) {
        return false;
    }
    for (size_t k = 0; k < dense_local_k.size(); ++k) {
        const size_t global_k = QK8_0 + dense_local_k[k];
        for (size_t col = 0; col < 3; ++col) {
            if (!check(tile[k * rmd::kArrayDim + col] ==
                           dense_kxj[global_k * kxj_stride + 1 + col],
                       "dense KxJ gather preserves padded stride")) {
                return false;
            }
        }
    }
    return check(metrics.weight_values_gathered == dense_local_k.size() * 3 &&
                 metrics.weight_baseline_address_resolutions == dense_local_k.size() * 3 &&
                 metrics.weight_address_resolutions == dense_local_k.size(),
                 "dense KxJ resolves once per valid K row");
}

struct GatherBenchResult {
    std::string layout;
    size_t iterations = 0;
    uint64_t scalar_checksum = 0;
    uint64_t candidate_checksum = 0;
    double scalar_median_ns_per_tile = 0.0;
    double candidate_median_ns_per_tile = 0.0;
    double candidate_ratio = 0.0;
    double min_batch_ms = 0.0;
    size_t baseline_resolutions_per_tile = 0;
    size_t candidate_resolutions_per_tile = 0;
    bool checksum_match = false;
};

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

GatherBenchResult benchmark_gather_layout(const std::string & layout,
                                          const ggml_gemmini_args_t & args,
                                          uint32_t block_count,
                                          const std::array<uint16_t, 10> & local_k) {
    using Clock = std::chrono::steady_clock;
    constexpr double minimum_batch_ns = 100000000.0;
    auto run = [&](bool scalar, size_t iterations, uint64_t & checksum,
                   rmd::RmdExecutionMetrics & metrics) {
        const auto start = Clock::now();
        const rmd::RmdStatus status = scalar
            ? rmd::repeat_scalar_weight_tile_gather_for_test(
                  args, block_count, local_k.data(), local_k.size(), 0, args.J,
                  iterations, checksum)
            : rmd::repeat_weight_tile_gather_for_test(
                  args, block_count, local_k.data(), local_k.size(), 0, args.J,
                  iterations, checksum, metrics);
        const auto stop = Clock::now();
        if (status != rmd::RmdStatus::success) {
            return -1.0;
        }
        return std::chrono::duration<double, std::nano>(stop - start).count();
    };

    size_t iterations = 1024;
    for (;;) {
        uint64_t scalar_checksum = 0;
        uint64_t candidate_checksum = 0;
        rmd::RmdExecutionMetrics metrics{};
        const double scalar_ns = run(true, iterations, scalar_checksum, metrics);
        const double candidate_ns = run(false, iterations, candidate_checksum, metrics);
        if (scalar_ns >= minimum_batch_ns && candidate_ns >= minimum_batch_ns) {
            break;
        }
        iterations *= 2;
    }

    std::vector<double> scalar_batches;
    std::vector<double> candidate_batches;
    uint64_t scalar_checksum = 0;
    uint64_t candidate_checksum = 0;
    rmd::RmdExecutionMetrics candidate_metrics{};
    for (size_t batch = 0; batch < 7; ++batch) {
        uint64_t batch_scalar_checksum = 0;
        uint64_t batch_candidate_checksum = 0;
        rmd::RmdExecutionMetrics batch_metrics{};
        const double scalar_ns = run(true, iterations, batch_scalar_checksum, batch_metrics);
        const double candidate_ns = run(false, iterations, batch_candidate_checksum, batch_metrics);
        scalar_batches.push_back(scalar_ns);
        candidate_batches.push_back(candidate_ns);
        scalar_checksum = batch_scalar_checksum;
        candidate_checksum = batch_candidate_checksum;
        candidate_metrics = batch_metrics;
    }

    const double tiles_per_batch = static_cast<double>(iterations) * block_count;
    const double scalar_median = median(scalar_batches) / tiles_per_batch;
    const double candidate_median = median(candidate_batches) / tiles_per_batch;
    GatherBenchResult result;
    result.layout = layout;
    result.iterations = iterations;
    result.scalar_checksum = scalar_checksum;
    result.candidate_checksum = candidate_checksum;
    result.scalar_median_ns_per_tile = scalar_median;
    result.candidate_median_ns_per_tile = candidate_median;
    result.candidate_ratio = candidate_median / scalar_median;
    result.min_batch_ms = std::min(*std::min_element(scalar_batches.begin(), scalar_batches.end()),
                                   *std::min_element(candidate_batches.begin(), candidate_batches.end())) /
        1000000.0;
    result.baseline_resolutions_per_tile =
        candidate_metrics.weight_baseline_address_resolutions /
        (iterations * block_count);
    result.candidate_resolutions_per_tile =
        candidate_metrics.weight_address_resolutions /
        (iterations * block_count);
    result.checksum_match = scalar_checksum == candidate_checksum;
    return result;
}

bool run_rmd_gather_benchmark(const std::filesystem::path & json_path,
                              double max_h1_ratio) {
    constexpr size_t columns = 16;
    constexpr size_t logical_k = 64;
    constexpr uint32_t block_count = 2;
    constexpr std::array<uint16_t, 10> local_k = { 0, 3, 7, 9, 12, 16, 21, 24, 27, 31 };

    std::vector<block_q8_h1> h1(columns * block_count);
    for (size_t j = 0; j < columns; ++j) {
        for (size_t block = 0; block < block_count; ++block) {
            for (size_t k = 0; k < QK8_0; ++k) {
                h1[j * block_count + block].qs[k] =
                    static_cast<int8_t>(j * 19 + block * 7 + k - 96);
            }
        }
    }
    ggml_gemmini_args_t args{};
    args.J = columns;
    args.K = logical_k;
    args.weight_format = ggml_gemmini_args_t::im2p_weight_format_t::q8_h1;
    args.q8_h1_blocks = h1.data();
    args.q8_h1_block_count = h1.size();
    args.q8_h1_rows = columns;
    args.blocks_per_row = block_count;
    args.native_weight_bytes = h1.size() * sizeof(block_q8_h1);
    args.block_size_k = QK8_0;

    std::vector<GatherBenchResult> results;
    results.push_back(benchmark_gather_layout("q8_h1", args, block_count, local_k));

    constexpr size_t jxk_stride = logical_k + 7;
    constexpr size_t kxj_stride = columns + 9;
    std::vector<elem_t> dense_jxk(columns * jxk_stride, 0);
    std::vector<elem_t> dense_kxj(logical_k * kxj_stride, 0);
    for (size_t j = 0; j < columns; ++j) {
        for (size_t k = 0; k < logical_k; ++k) {
            const int8_t value = static_cast<int8_t>(j * 19 + k - 96);
            dense_jxk[j * jxk_stride + k] = value;
            dense_kxj[k * kxj_stride + j] = value;
        }
    }
    args = {};
    args.J = columns;
    args.K = logical_k;
    args.B = dense_jxk.data();
    args.sB = jxk_stride;
    args.transpose_B = true;
    args.weight_i8_scale_active = true;
    args.weight_scale = 1.0f;
    results.push_back(benchmark_gather_layout("dense_jxk", args, block_count, local_k));
    args.B = dense_kxj.data();
    args.sB = kxj_stride;
    args.transpose_B = false;
    results.push_back(benchmark_gather_layout("dense_kxj", args, block_count, local_k));

    size_t baseline_resolutions = 0;
    size_t candidate_resolutions = 0;
    bool checksums_match = true;
    bool batches_long_enough = true;
    for (const GatherBenchResult & result : results) {
        baseline_resolutions += result.baseline_resolutions_per_tile;
        candidate_resolutions += result.candidate_resolutions_per_tile;
        checksums_match = checksums_match && result.checksum_match;
        batches_long_enough = batches_long_enough && result.min_batch_ms >= 100.0;
    }
    const double address_reduction = 1.0 -
        static_cast<double>(candidate_resolutions) / baseline_resolutions;
    const bool passed = checksums_match && batches_long_enough &&
        results.front().candidate_ratio <= max_h1_ratio && address_reduction >= 0.70;

    std::ostringstream json;
    json << std::fixed << std::setprecision(3)
         << "{\n  \"record_type\": \"RMD_GATHER_BENCHMARK\",\n"
         << "  \"fixture\": {\"selected_k_per_block\": 10, \"j\": 16, \"block_count\": 2},\n"
         << "  \"max_h1_ratio\": " << max_h1_ratio << ",\n"
         << "  \"address_reduction\": " << address_reduction << ",\n"
         << "  \"checksums_match\": " << (checksums_match ? "true" : "false") << ",\n"
         << "  \"batches_long_enough\": " << (batches_long_enough ? "true" : "false") << ",\n"
         << "  \"passed\": " << (passed ? "true" : "false") << ",\n"
         << "  \"failed_field\": \""
         << (!checksums_match ? "checksum" : !batches_long_enough ? "batch_duration" :
             results.front().candidate_ratio > max_h1_ratio ? "h1_ratio" :
             address_reduction < 0.70 ? "address_reduction" : "") << "\",\n"
         << "  \"layouts\": [\n";
    for (size_t i = 0; i < results.size(); ++i) {
        const GatherBenchResult & result = results[i];
        json << "    {\"layout\": \"" << result.layout
             << "\", \"iterations\": " << result.iterations
             << ", \"scalar_checksum\": " << result.scalar_checksum
             << ", \"candidate_checksum\": " << result.candidate_checksum
             << ", \"checksum_match\": " << (result.checksum_match ? "true" : "false")
             << ", \"scalar_median_ns_per_tile\": " << result.scalar_median_ns_per_tile
             << ", \"candidate_median_ns_per_tile\": " << result.candidate_median_ns_per_tile
             << ", \"candidate_ratio\": " << result.candidate_ratio
             << ", \"min_batch_ms\": " << result.min_batch_ms
             << ", \"baseline_address_resolutions_per_tile\": "
             << result.baseline_resolutions_per_tile
             << ", \"candidate_address_resolutions_per_tile\": "
             << result.candidate_resolutions_per_tile << "}"
             << (i + 1 == results.size() ? "\n" : ",\n");
    }
    json << "  ]\n}\n";

    if (!json_path.empty()) {
        std::ofstream output(json_path);
        if (!output) {
            std::fprintf(stderr, "failed to open benchmark JSON: %s\n", json_path.c_str());
            return false;
        }
        output << json.str();
    }
    std::printf("%s", json.str().c_str());
    return passed;
}
#endif

bool profile_output_routing(const std::filesystem::path & expected, bool invalid_parent) {
    if (invalid_parent) {
        std::ofstream("blocked") << "not a directory";
    }

    std::vector<float> source(64);
    for (size_t index = 0; index < source.size(); ++index) {
        source[index] = index % 4 == 0 ? -0.5f : 0.5f;
    }
    std::vector<elem_t> quantized(64);
    ggml_tensor tensor{};
    tensor.type = GGML_TYPE_F32;
    tensor.data = source.data();
    ggml_gemmini_args_t args{};
    args.I = 1;
    args.J = 1;
    args.K = source.size();
    args.A.allocate(args.I, args.K, GGML_GEMMINI_ACTIVATION_BITS);
        args.sA = args.K;

    const bool wrote = quants::quantize_activation(&tensor, args);
    if (invalid_parent) {
        return check(!wrote, "invalid ExSIA profile parent reports existing writer failure") &&
            check(!std::filesystem::exists(expected), "invalid ExSIA profile path is absent") &&
            check(!std::filesystem::exists("log/exsia-cycle-detail.jsonl"), "invalid ExSIA profile does not fall back to legacy log");
    }

    std::ifstream input(expected);
    std::string line;
    const bool parsed = std::getline(input, line) && !line.empty() && line.front() == '{' &&
        line.back() == '}' &&
        line.find("\"schema\":\"gemmini.cycle\"") != std::string::npos &&
        line.find("\"version\":2") != std::string::npos &&
        line.find("\"record_type\":\"TIMELINE\"") != std::string::npos &&
        line.find("\"op\":\"exsia.local\"") != std::string::npos &&
        line.find("\"layer\":null") != std::string::npos &&
        line.find("\"stripe_id\":0") != std::string::npos &&
        line.find("\"slot\":0") != std::string::npos &&
        line.find("\"worker_id\":null") != std::string::npos &&
        line.find("exsia.timeline.run.") == std::string::npos;
#if defined(GGML_GEMMINI_HAS_OPENMP) && GGML_GEMMINI_EXSIA_DEFAULT_MODE_VALUE > 0
    std::array<bool, GGML_GEMMINI_EXSIA_LOCAL_WORKERS> workers{};
    while (std::getline(input, line)) {
        if (line.find("\"op\":\"exsia.local_group\"") == std::string::npos ||
            line.find("\"slot\":0") == std::string::npos) {
            continue;
        }
        for (size_t worker = 0; worker < workers.size(); ++worker) {
            const std::string worker_id =
                "\"worker_id\":" + std::to_string(worker);
            workers[worker] = workers[worker] ||
                line.find(worker_id) != std::string::npos;
        }
    }
    const bool worker_rows =
        std::all_of(workers.begin(), workers.end(), [](bool seen) { return seen; });
#else
    const bool worker_rows = true;
#endif
    return check(wrote, "ExSIA profile writer succeeds") &&
        check(parsed, "ExSIA profile writer emits non-empty JSONL") &&
        check(worker_rows, "ExSIA profile writer identifies every local worker") &&
        check(!std::filesystem::exists("log/exsia-cycle-detail.jsonl"), "ExSIA profile writer does not create legacy log");
}

#ifndef GEMMINI_EXSIA_WRITER_TEST_ONLY
bool test_compiled_width_rmd_suite() {
#if GGML_GEMMINI_ACTIVATION_BITS == 8 && GGML_GEMMINI_WEIGHT_BITS == 8
    return test_rmd_cpu_ws_routes() &&
        test_q8_srmd_software_ws_routing() &&
        test_q8_hp1_srmd_software_ws_routing() &&
        test_rmd_cpu_direct_parity() &&
        test_rmd_lane_partition() &&
        test_rmd_weight_gather();
#else
    return true;
#endif
}
#endif

}

int main(int argc, char ** argv) {
    if (argc >= 3 && std::string(argv[1]) == "--profile-output") {
        const bool invalid_parent = argc == 4 && std::string(argv[3]) == "--invalid-parent";
        return profile_output_routing(argv[2], invalid_parent) ? 0 : 1;
    }
#ifdef GEMMINI_EXSIA_WRITER_TEST_ONLY
    return 1;
#else
    if (argc >= 2 && std::string(argv[1]) == "--bench-rmd-gather") {
        std::filesystem::path json_path;
        double max_h1_ratio = 0.80;
        for (int i = 2; i < argc; ++i) {
            if (std::string(argv[i]) == "--json" && i + 1 < argc) {
                json_path = argv[++i];
            } else if (std::string(argv[i]) == "--max-h1-ratio" && i + 1 < argc) {
                char * end = nullptr;
                max_h1_ratio = std::strtod(argv[++i], &end);
                if (end == argv[i] || *end != '\0') {
                    return 2;
                }
            } else {
                std::fprintf(stderr,
                             "usage: %s --bench-rmd-gather [--json path] [--max-h1-ratio value]\n",
                             argv[0]);
                return 2;
            }
        }
        return run_rmd_gather_benchmark(json_path, max_h1_ratio) ? 0 : 1;
    }

    std::string case_name = "all";
    if (argc == 2 && std::string(argv[1]).rfind("--case=", 0) == 0) {
        case_name = std::string(argv[1]).substr(7);
    } else if (argc == 3 && std::string(argv[1]) == "--case") {
        case_name = argv[2];
    } else if (argc != 1) {
        std::fprintf(stderr, "usage: %s [--case=<name>|--case <name>]\n", argv[0]);
        return 2;
    }

    const bool known = case_name == "all" || case_name == "baseline" ||
        case_name == "dispatch" || case_name == "rmd-routes" ||
        case_name == "q8-srmd-software-ws" || case_name == "rmd-ws-contract-probe" ||
        case_name == "hp1-srmd-software-ws" ||
        case_name == "rmd-direct-parity" || case_name == "direct-executor" ||
        case_name == "rmd-gather" || case_name == "stripe-geometry" ||
        case_name == "meta-rho" || case_name == "non-outlier-clipping";
    if (!known) {
        std::fprintf(stderr, "unknown case: %s\n", case_name.c_str());
        return 2;
    }

    std::printf("TEST_CASE_BEGIN name=%s\n", case_name.c_str());
    const bool ok =
        (case_name == "all" && test_meta_rho_invariant() &&
         test_non_outlier_clipping_policy() &&
         test_exsia_baseline() && test_dispatch_modes() &&
         test_compiled_width_rmd_suite() && test_direct_cpu_executor() &&
         test_activation_stripe_geometry_contract() &&
         test_cpu_direct_residual_dequantization()) ||
        (case_name == "baseline" && test_exsia_baseline()) ||
        (case_name == "dispatch" && test_dispatch_modes()) ||
        (case_name == "rmd-routes" && test_rmd_cpu_ws_routes()) ||
        (case_name == "q8-srmd-software-ws" && test_q8_srmd_software_ws_routing()) ||
        (case_name == "hp1-srmd-software-ws" && test_q8_hp1_srmd_software_ws_routing()) ||
        (case_name == "rmd-ws-contract-probe" && test_rmd_ws_contract_probe()) ||
        (case_name == "rmd-direct-parity" && test_rmd_cpu_direct_parity() &&
         test_rmd_lane_partition()) ||
        (case_name == "direct-executor" && test_direct_cpu_executor()) ||
        (case_name == "stripe-geometry" && test_activation_stripe_geometry_contract() &&
         test_cpu_direct_residual_dequantization()) ||
        (case_name == "meta-rho" && test_meta_rho_invariant()) ||
        (case_name == "non-outlier-clipping" &&
         test_non_outlier_clipping_policy()) ||
        (case_name == "rmd-gather" && test_rmd_weight_gather());
    if (ok)
        std::printf("PASS: case=%s\n", case_name.c_str());
    return ok ? 0 : 1;
#endif
}
