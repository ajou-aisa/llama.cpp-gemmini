#include "../ggml/src/ggml-gemmini/ggml-gemmini-im2p.hpp"
#include "../ggml/src/ggml-gemmini/residual/rmd/rmd-builder.hpp"
#include "../ggml/src/ggml-gemmini/residual/rmd/rmd-compose.hpp"
#include "../ggml/src/ggml-gemmini/residual/rmd/rmd-executor.hpp"
#include "../ggml/src/ggml-gemmini/residual/rmd/rmd-im2p-executor.hpp"
#include "../ggml/src/ggml-gemmini/ggml-gemmini-args.h"
#include "../ggml/src/ggml-gemmini/quants/act/exsia/exsia.hpp"
#include "../ggml/src/ggml-gemmini/quants/common/weight_reader.hpp"

extern "C" im2p_sim_t * im2p_sim_create(void);
extern "C" void im2p_sim_destroy(im2p_sim_t * sim);

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <string_view>
#include <type_traits>
#include <vector>

namespace {

using namespace ggml::gemmini::rmd;
using ggml::gemmini::im2p_adapter::BuildIdentity;
using ggml::gemmini::im2p_adapter::Error;
using ggml::gemmini::im2p_adapter::ExsiaRouteRequest;
using ggml::gemmini::im2p_adapter::PublicMode;
using ggml::gemmini::im2p_adapter::ResidualBackend;
using ggml::gemmini::im2p_adapter::WeightFamily;

bool check(bool condition, const char * message) {
    if (!condition) std::fprintf(stderr, "FAIL: %s\n", message);
    return condition;
}

struct SimDeleter {
    void operator()(im2p_sim_t * sim) const { im2p_sim_destroy(sim); }
};
using Sim = std::unique_ptr<im2p_sim_t, SimDeleter>;

struct Fixture {
    static constexpr size_t rows = DIM + 1;
    static constexpr size_t columns = 3;
    static constexpr size_t logical_k = 2 * kBlockSize;

    ggml_gemmini_args_t args{};
#if GGML_GEMMINI_WEIGHT_BITS == 4
    std::vector<block_q4_h1> h1;
    std::vector<block_q4_hp1> hp1;
#elif GGML_GEMMINI_WEIGHT_BITS == 8
    std::vector<block_q8_h1> h1;
    std::vector<block_q8_hp1> hp1;
#else
    std::vector<block_q16_h1> h1;
    std::vector<block_q16_hp1> hp1;
#endif
    StripePacketHandle packet;

    void set_code(size_t block_index, size_t k, int32_t code) {
#if GGML_GEMMINI_WEIGHT_BITS == 4
        const size_t byte = k % (kBlockSize / 2);
        const uint8_t nibble = static_cast<uint8_t>(code + 8);
        auto set = [&](auto & block) {
            if (k < kBlockSize / 2) block.qs[byte] = static_cast<uint8_t>((block.qs[byte] & 0xf0u) | nibble);
            else block.qs[byte] = static_cast<uint8_t>((block.qs[byte] & 0x0fu) | (nibble << 4));
        };
        set(h1[block_index]);
        set(hp1[block_index]);
#else
        using H1Code = std::remove_reference_t<decltype(h1[block_index].qs[k])>;
        using Hp1Code = std::remove_reference_t<decltype(hp1[block_index].qs[k])>;
        h1[block_index].qs[k] = static_cast<H1Code>(code);
        hp1[block_index].qs[k] = static_cast<Hp1Code>(code);
#endif
    }

    explicit Fixture(bool use_hp1 = false, int16_t exponent = 2,
                     size_t row_count = rows, size_t column_count = columns,
                     size_t weight_seed = 0) : h1(column_count * 2), hp1(column_count * 2) {
        args.I = row_count;
        args.J = column_count;
        args.K = logical_k;
        args.block_size_k = kBlockSize;
        args.native_block_count = h1.size();
        args.native_blocks_per_row = 2;
        args.A.allocate(row_count, logical_k, GGML_GEMMINI_ACTIVATION_BITS);

        for (size_t block_index = 0; block_index < h1.size(); ++block_index) {
            for (size_t k = 0; k < kBlockSize; ++k) {
                const int32_t code = static_cast<int32_t>((block_index * 17 + k * 5 + weight_seed) % 7) - 3;
                set_code(block_index, k, code);
            }
            h1[block_index].c_b = 3;
            h1[block_index].R = static_cast<uint16_t>(block_index + 1);
            h1[block_index].s_rf = 0.25f;
            hp1[block_index].m = exponent;
            hp1[block_index].channel_scale = 0.25f;
        }
#if GGML_GEMMINI_WEIGHT_BITS == 4
        args.weight_format = use_hp1 ? ggml_gemmini_args_t::im2p_weight_format_t::q4_hp1 : ggml_gemmini_args_t::im2p_weight_format_t::q4_h1;
        args.q4_h1_blocks = use_hp1 ? nullptr : h1.data();
        args.q4_hp1_blocks = use_hp1 ? hp1.data() : nullptr;
#elif GGML_GEMMINI_WEIGHT_BITS == 8
        args.weight_format = use_hp1 ? ggml_gemmini_args_t::im2p_weight_format_t::q8_hp1 : ggml_gemmini_args_t::im2p_weight_format_t::q8_h1;
        args.q8_h1_blocks = use_hp1 ? nullptr : h1.data();
        args.q8_h1_block_count = use_hp1 ? 0 : h1.size();
        args.q8_h1_rows = use_hp1 ? 0 : column_count;
        args.blocks_per_row = use_hp1 ? 0 : 2;
        args.q8_hp1_blocks = use_hp1 ? hp1.data() : nullptr;
        args.q8_hp1_block_count = use_hp1 ? hp1.size() : 0;
        args.q8_hp1_blocks_per_row = use_hp1 ? 2 : 0;
#else
        args.weight_format = use_hp1 ? ggml_gemmini_args_t::im2p_weight_format_t::q16_hp1 : ggml_gemmini_args_t::im2p_weight_format_t::q16_h1;
        args.q16_h1_blocks = use_hp1 ? nullptr : h1.data();
        args.q16_hp1_blocks = use_hp1 ? hp1.data() : nullptr;
#endif
        args.native_weight_bytes = use_hp1 ? hp1.size() * sizeof(hp1.front()) : h1.size() * sizeof(h1.front());

        RmdStripeBuilder builder;
        builder.reset(19, 7, row_count, logical_k, column_count, GGML_GEMMINI_ACTIVATION_BITS);
        // Populate both K blocks so cancellation and checked K-accumulation
        // probes cross a real dot boundary without exceeding logical K.
        for (size_t k = 0; k < kBlockSize; ++k) {
            builder.add_residual(k % row_count, k, 1);
        }
        for (size_t k : {size_t{0}, size_t{1}, kBlockSize - 2, kBlockSize - 1}) {
            builder.add_residual((k + 3) % row_count, kBlockSize + k, -1);
        }
        packet = builder.finish();
    }
};

bool unchanged(const CompressedOutput & output, const RmdExecutionMetrics & metrics) {
    return output.j_padded == 91 && output.values == std::vector<OutputValue>({7, -11}) &&
           metrics.packet_call_count == 73 && metrics.im2p_dot_calls == 79 &&
           metrics.im2p_stats.work_total_cycles() == 0 &&
           metrics.ws_call_count == 0;
}

bool unchanged(const Correction & correction, const RmdExecutionMetrics & metrics) {
    const auto * values = std::get_if<PreScaledFloat64Correction>(&correction);
    return values != nullptr && values->values == std::vector<double>({7.25, -11.5}) &&
           metrics.packet_call_count == 73 && metrics.im2p_dot_calls == 79 &&
           metrics.im2p_stats.fields == RmdProviderStats{}.fields &&
           metrics.ws_call_count == 0;
}

bool run_success() {
    Fixture fixture;
    Sim sim(im2p_sim_create());
    CompressedOutput expected;
    CompressedOutput actual;
    RmdExecutionMetrics expected_metrics{};
    RmdExecutionMetrics actual_metrics{};
    RmdExecutionMetrics routed_metrics{};
    CompressedOutput routed;
    const RmdStatus oracle = fixture.packet ? execute_rmd_stripe_reference(
        fixture.args, *fixture.packet, expected, &expected_metrics) : RmdStatus::invalid_packet;
    ggml::gemmini::quants::wreader::test_reset_weight_reader_counters();
    const RmdStatus status = fixture.packet && sim ? execute_rmd_stripe_im2p(
        sim.get(), fixture.args, *fixture.packet, actual, &actual_metrics) : RmdStatus::execution_failed;
    const size_t code_reads =
        ggml::gemmini::quants::wreader::test_weight_reader_code_address_resolutions();
    const RmdStatus routed_status = fixture.packet ? execute_rmd_stripe_ws(
        fixture.args, *fixture.packet, routed, &routed_metrics) : RmdStatus::invalid_packet;
    Correction composed = PreScaledFloat64Correction{{7.25, -11.5}};
    Correction streamed = composed;
    RmdExecutionMetrics streaming_metrics{};
    const RmdStatus compose_status = fixture.packet ?
        compose_rmd_output(*fixture.packet, expected, composed) : RmdStatus::invalid_packet;
    const RmdStatus streaming_status = fixture.packet && sim ? execute_rmd_stripe_im2p(
        sim.get(), fixture.args, *fixture.packet, streamed, &streaming_metrics) :
        RmdStatus::execution_failed;
    const auto * composed_values = std::get_if<BlockScaledInt64Correction>(&composed);
    const auto * streamed_values = std::get_if<BlockScaledInt64Correction>(&streamed);
    const bool ok = check(oracle == RmdStatus::success && status == RmdStatus::success &&
                              routed_status == RmdStatus::success,
                          "provider, internal route, and checked oracle execute") &&
        check(actual.domain == expected.domain && actual.j_padded == expected.j_padded && actual.values == expected.values,
              "provider output equals checked oracle") &&
        check(actual_metrics.im2p_dot_calls > 0 &&
                  actual_metrics.im2p_stats.work_total_cycles() > 0 &&
                  actual_metrics.ws_call_count == 0,
              "active packet exposes nonzero independent IM2P provider stats") &&
        check(actual_metrics.matmul_call_count == actual_metrics.im2p_dot_calls,
              "each compact K call maps to one provider dot") &&
        check(code_reads > 0 && code_reads == actual_metrics.weight_address_resolutions,
              "native provider gathers weights without a redundant pre-dispatch read") &&
        check(routed.values == expected.values && routed_metrics.im2p_dot_calls > 0 &&
                  routed_metrics.im2p_stats.work_total_cycles() > 0 &&
                  routed_metrics.ws_call_count == 0,
              "public compact route propagates provider stats without fallback") &&
        check(compose_status == RmdStatus::success && streaming_status == RmdStatus::success &&
                  composed_values != nullptr && streamed_values != nullptr &&
                  streamed_values->values == composed_values->values,
              "typed provider streaming equals checked compact output plus compose") &&
        check(streaming_metrics.im2p_dot_calls == actual_metrics.im2p_dot_calls &&
                  streaming_metrics.im2p_stats.work_total_cycles() > 0 &&
                  streaming_metrics.ws_call_count == 0 &&
                  streaming_metrics.compressed_output_values == 0,
              "streaming uses the typed provider without allocating compressed output");
    if (ok) std::printf("IM2P_PROVIDER success width=%d dot_calls=%zu rmd_cycles=%llu "
                        "output_writes=%llu ws_calls=%zu values=%zu first=%lld\n",
                        GGML_GEMMINI_ACTIVATION_BITS, actual_metrics.im2p_dot_calls,
                        static_cast<unsigned long long>(
                            actual_metrics.im2p_stats.work_total_cycles()),
                        static_cast<unsigned long long>(
                            actual_metrics.im2p_stats.output_write_requests()),
                        actual_metrics.ws_call_count, actual.values.size(),
                        static_cast<long long>(actual.values.empty() ? 0 : actual.values.front()));
    return ok;
}

bool run_hp1_exp_62() {
    Fixture fixture(true, 62);
    for (size_t block_index = 0; block_index < fixture.hp1.size(); ++block_index) {
#if GGML_GEMMINI_WEIGHT_BITS == 4
        std::fill(std::begin(fixture.hp1[block_index].qs),
                  std::end(fixture.hp1[block_index].qs), uint8_t{0x88});
        fixture.hp1[block_index].qs[0] = uint8_t{0x89};
#else
        std::fill(std::begin(fixture.hp1[block_index].qs),
                  std::end(fixture.hp1[block_index].qs), 0);
        fixture.hp1[block_index].qs[0] = 1;
#endif
        fixture.hp1[block_index].m = block_index % 2 == 0
            ? int16_t{62} : std::numeric_limits<int16_t>::min();
    }
    Sim sim(im2p_sim_create());
    CompressedOutput expected;
    CompressedOutput actual;
    RmdExecutionMetrics metrics{};
    const RmdStatus oracle = fixture.packet ? execute_rmd_stripe_reference(
        fixture.args, *fixture.packet, expected) : RmdStatus::invalid_packet;
    const RmdStatus status = fixture.packet && sim ? execute_rmd_stripe_im2p(
        sim.get(), fixture.args, *fixture.packet, actual, &metrics) : RmdStatus::execution_failed;
    const auto beyond_i32 = std::find_if(actual.values.begin(), actual.values.end(),
        [](int64_t value) { return value > std::numeric_limits<int32_t>::max() ||
                                  value < std::numeric_limits<int32_t>::min(); });
    const bool ok = check(oracle == RmdStatus::success && status == RmdStatus::success,
                          "HP1 exponent 62 executes") &&
        check(actual.values == expected.values, "HP1 exponent 62 matches oracle") &&
        check(beyond_i32 != actual.values.end(), "provider preserves output beyond int32") &&
        check(metrics.im2p_dot_calls > 0 && metrics.ws_call_count == 0,
              "HP1 exponent 62 uses IM2P only");
    if (ok) std::printf("IM2P_PROVIDER hp1-exp-62 status=success dot_calls=%zu beyond_i32=%lld ws_calls=0\n",
                        metrics.im2p_dot_calls, static_cast<long long>(*beyond_i32));
    return ok;
}

bool run_malformed_packet() {
    Fixture fixture;
    StripePacket malformed = *fixture.packet;
    malformed.blocks.front().k_index_offset = static_cast<uint32_t>(malformed.k_indices.size());
    Sim sim(im2p_sim_create());
    CompressedOutput output{CompressedOutput::Domain::block_scaled_int64, 91, {7, -11}};
    RmdExecutionMetrics metrics{};
    metrics.packet_call_count = 73;
    metrics.im2p_dot_calls = 79;
    const RmdStatus status = execute_rmd_stripe_im2p(
        sim.get(), fixture.args, malformed, output, &metrics);
    const bool ok = check(status == RmdStatus::invalid_packet,
                          "malformed packet rejected") &&
        check(unchanged(output, metrics), "malformed packet is transactional");
    if (ok) std::puts("IM2P_PROVIDER malformed-packet status=invalid_packet before_execute=1 sentinels=unchanged ws_calls=0");
    return ok;
}

bool run_shared_preparation() {
    namespace wreader = ggml::gemmini::quants::wreader;
    namespace exsia = ggml::gemmini::quants::act::exsia;
    constexpr size_t stripes = 2;
    Sim sim(im2p_sim_create());
    if (!check(sim != nullptr, "shared preparation simulator exists")) return false;
    for (const bool use_hp1 : {false, true}) {
        Fixture fixture(use_hp1, 2, stripes, 1);
        auto & args = fixture.args;
        args.act_quant.storage().emplace<exsia::Meta>().theta = {-1};
        std::array<StripePacketHandle, stripes> packets;
        for (size_t row = 0; row < stripes; ++row) {
            RmdStripeBuilder builder;
            builder.reset(47, row, 1, args.K, args.J, GGML_GEMMINI_ACTIVATION_BITS);
            if (!check(builder.add_residual(0, kBlockSize + 1, row == 0 ? 1 : -1),
                       "shared preparation residual accepted")) return false;
            packets[row] = builder.finish();
            if (!check(packets[row] != nullptr, "shared preparation packet exists")) return false;
        }
        const std::array<OutputValue, stripes> expected_raw = use_hp1
            ? std::array<OutputValue, stripes>{-8, 8}
            : std::array<OutputValue, stripes>{-10, 10};
        const std::array<float, stripes> expected_output = use_hp1
            ? std::array<float, stripes>{6.0f, 8.0f}
            : std::array<float, stripes>{5.75f, 8.25f};
        detail::RmdWeightPreparation shared;
        std::array<Correction, stripes> corrections;
        for (const bool reuse : {false, true}) {
            std::array<float, stripes> output{7.0f, 7.0f};
            wreader::test_reset_weight_reader_counters();
            for (size_t row = 0; row < stripes; ++row) {
                detail::RmdWeightPreparation fresh;
                auto & weights = reuse ? shared : fresh;
                RmdExecutionMetrics metrics{};
                if (!check(detail::execute_rmd_stripe_im2p_with_weights(
                               sim.get(), args, *packets[row], corrections[row], weights,
                               &metrics) == RmdStatus::success,
                           "shared and fresh preparation execute through IM2P")) return false;
                const auto * integer = std::get_if<BlockScaledInt64Correction>(&corrections[row]);
                if (!check(integer != nullptr && integer->values ==
                               std::vector<OutputValue>{expected_raw[row]} &&
                               metrics.im2p_dot_calls == 1 && metrics.ws_call_count == 0,
                           "shared and fresh provider corrections equal literal weighted residuals")) return false;
                size_t nonzero_count = 0;
                if (!check(detail::merge_rmd_correction_with_weights(
                               args, output.data(), *packets[row], corrections[row], weights,
                               &nonzero_count) == RmdStatus::success && nonzero_count == 1,
                           "shared and fresh corrections merge")) return false;
                if (!check(weights.column_preparations() == 1 &&
                               weights.selected_block_preparations() == 1,
                           "column and selected block preparation occur once per context")) return false;
            }
            const size_t validations = wreader::test_weight_reader_storage_validations();
            if (!check(validations == (reuse ? 1 : stripes) && output == expected_output,
                       "shared stripes validate once and preserve exact FP32 output")) return false;
            std::array<float, stripes> original_output{7.0f, 7.0f};
            for (size_t row = 0; row < stripes; ++row) {
                if (!check(merge_rmd_correction_to(args, original_output.data(), *packets[row],
                                                  corrections[row]) == RmdStatus::success,
                           "original packet merge accepts provider corrections")) return false;
            }
            if (!check(original_output == output, "shared preparation preserves original FP32 merge")) return false;
            std::printf("IM2P_PROVIDER shared-preparation route=%s reuse=%d stripes=%zu validations=%zu\n",
                        use_hp1 ? "HP1" : "H1", reuse, stripes, validations);
        }

        StripePacket malformed = *packets.front();
        ++malformed.version;
        Correction rejected = PreScaledFloat64Correction{{7.25, -11.5}};
        RmdExecutionMetrics metrics{};
        metrics.packet_call_count = 73;
        metrics.im2p_dot_calls = 79;
        if (!check(detail::execute_rmd_stripe_im2p_with_weights(
                       sim.get(), args, malformed, rejected, shared, &metrics) == RmdStatus::invalid_packet &&
                       unchanged(rejected, metrics),
                   "shared preparation malformed execution preserves correction and metrics")) return false;
        std::array<float, stripes> output{7.0f, 7.0f};
        const auto sentinel = output;
        size_t nonzero_count = 91;
        StripePacket mismatched = *packets.front();
        ++mismatched.logical_j;
        if (!check(detail::merge_rmd_correction_with_weights(
                       args, output.data(), mismatched, corrections.front(), shared, &nonzero_count) ==
                       RmdStatus::invalid_arguments && output == sentinel && nonzero_count == 91,
                   "shared preparation shape mismatch preserves output and count")) return false;

        Fixture invalid(use_hp1, 2, stripes, 1);
        invalid.h1[1].s_rf = invalid.hp1[1].channel_scale = 0.5f;
        detail::RmdWeightPreparation invalid_weights;
        if (!check(detail::merge_rmd_correction_with_weights(
                       invalid.args, output.data(), *packets.front(), corrections.front(),
                       invalid_weights, &nonzero_count) == RmdStatus::unsupported_route &&
                       merge_rmd_correction_to(invalid.args, output.data(), *packets.front(),
                                               corrections.front(), &nonzero_count) == RmdStatus::unsupported_route &&
                       output == sentinel && nonzero_count == 91,
                   "shared and original merge reject selected scale mismatch transactionally")) return false;
        if (use_hp1) {
            invalid.hp1[1].m = 63;
            detail::RmdWeightPreparation overflow_weights;
            if (!check(detail::execute_rmd_stripe_im2p_with_weights(
                           sim.get(), invalid.args, *packets.front(), rejected, overflow_weights,
                           &metrics) == RmdStatus::overflow && unchanged(rejected, metrics),
                       "shared preparation invalid block scale preserves correction and metrics")) return false;
        }
    }
    return true;
}

bool run_packet_merge_contract() {
    namespace adapter = ggml::gemmini::im2p_adapter;
    namespace exsia = ggml::gemmini::quants::act::exsia;
    for (const bool pipeline : {false, true}) {
        for (const bool use_hp1 : {false, true}) {
            for (size_t probe = 0; probe < 5; ++probe) {
                Fixture fixture(use_hp1);
                auto & args = fixture.args;
                args.I = 1;
                args.tile_I = args.tile_J = args.tile_K = 1;
                args.activation_rows_per_stripe = DIM;
                args.sA = args.K;
                args.sB = args.J;
                args.residual_route = ggml::gemmini::residual::ResidualRoute::ws_packet;
                std::array<float, Fixture::columns> output;
                output.fill(9876.0f);
                const auto sentinel = output;
                args.f_out = output.data();
                auto & metadata = args.act_quant.storage().emplace<exsia::Meta>();
                metadata.run_id = 41;
                metadata.e_s = 0;
                metadata.theta = {0};
                if (!check(args.A.allocate(1, args.K, GGML_GEMMINI_ACTIVATION_BITS),
                           "packet merge activation allocation")) return false;
                args.A.zero_fill();
                for (size_t j = 0; j < args.J; ++j) {
                    fixture.h1[j * 2 + 1].s_rf = 0.5f;
                    fixture.hp1[j * 2 + 1].channel_scale = 0.5f;
                }
                RmdStripeBuilder builder;
                builder.reset(0, 0, 1, args.K, args.J, GGML_GEMMINI_ACTIVATION_BITS);
                if (!check(builder.add_residual(0, probe == 1 ? kBlockSize + 1 : 1, 1),
                           "packet merge residual accepted")) return false;
                auto packet = builder.finish();
                if (!check(packet != nullptr, "packet merge packet exists")) return false;
                if (probe >= 2) {
                    auto malformed = std::make_shared<StripePacket>(*packet);
                    if (probe == 2) malformed->row_begin = 1;
                    if (probe == 3) malformed->row_count = std::numeric_limits<size_t>::max();
                    if (probe == 4) malformed->row_begin = std::numeric_limits<size_t>::max();
                    packet = std::move(malformed);
                }
                exsia::StripeReadyEvent event{};
                event.run_id = 41;
                event.row_end = 1;
                event.rmd_packet = packet;
                event.activation_metadata = exsia::StripeMetadataSnapshot{
                    metadata.e_s, metadata.rho, metadata.sigma, 0};
                adapter::test_reset();
                adapter::Completion completion;
                const auto publish = [&] {
                    const auto * sink = args.exsia_stripe_ready_sink;
                    return sink != nullptr && sink->on_ready(sink->user_data, event);
                };
                if (pipeline) {
                    auto started = adapter::start_exsia_stripe_pipeline(args);
                    if (!check(started.result.ok() && started.pipeline->install_sink().ok(),
                               "packet merge PIPELINE starts")) return false;
                    completion = started.pipeline->finish(publish());
                } else {
                    auto started = adapter::start_exsia_full_execution(args);
                    if (!check(started.result.ok() && started.execution->install_sink().ok(),
                               "packet merge FULL starts")) return false;
                    completion = started.execution->finish(publish());
                }
                const auto counters = adapter::test_counters();
                if (probe == 0) {
                    std::array<float, Fixture::columns> expected{};
                    for (size_t j = 0; j < args.J; ++j) {
                        const int code = static_cast<int>((j * 2 * 17 + 5) % 7) - 3;
                        expected[j] = code * (use_hp1 ? 4 : 4 + static_cast<int>(j * 2)) * 0.25f;
                    }
                    if (!check(completion.result.ok() && output == expected &&
                                   counters.commit == 1 && counters.rmd_dot_calls > 0,
                               "untouched block scale mismatch allows packet merge")) {
                        std::fprintf(stderr, "mode=%s route=%s status=%s commit=%llu dots=%llu output=%g,%g,%g expected=%g,%g,%g\n",
                                     pipeline ? "PIPELINE" : "FULL", use_hp1 ? "HP1" : "H1",
                                     completion.result.message,
                                     static_cast<unsigned long long>(counters.commit),
                                     static_cast<unsigned long long>(counters.rmd_dot_calls),
                                     output[0], output[1], output[2], expected[0], expected[1], expected[2]);
                        return false;
                    }
                } else if (!check(!completion.result.ok() && output == sentinel &&
                                      counters.commit == 0 &&
                                      (probe == 1 ? completion.result.error == Error::unsupported_route
                                                  : completion.result.error == Error::invalid_contract &&
                                                        counters.residual_executions == 0 &&
                                                        counters.provider_dot_attempts == 0),
                                  "touched scale or packet/event range mismatch preserves output")) {
                    std::fprintf(stderr, "mode=%s route=%s probe=%zu status=%s\n",
                                 pipeline ? "PIPELINE" : "FULL", use_hp1 ? "HP1" : "H1",
                                 probe, completion.result.message);
                    return false;
                }
            }
        }
    }
    std::puts("IM2P_PROVIDER packet-merge-contract modes=FULL,PIPELINE routes=H1,HP1 untouched=accepted touched=rejected rows=checked output=transactional");
    return true;
}

bool run_int32_residuals() {
    struct Event { size_t row; size_t k; int32_t value; };
    const std::array<Event, 4> events{{
        {0, 0, std::numeric_limits<int32_t>::max()},
        {0, Fixture::logical_k - 1, std::numeric_limits<int32_t>::min()},
        {DIM, 1, int32_t{1} << 20},
        {DIM, kBlockSize, -(int32_t{1} << 20) - 1},
    }};
    for (const bool use_hp1 : {false, true}) {
        Fixture fixture(use_hp1);
        RmdStripeBuilder builder;
        builder.reset(29, 0, Fixture::rows, Fixture::logical_k, Fixture::columns,
                      GGML_GEMMINI_ACTIVATION_BITS);
        for (const Event & event : events) {
            if (!check(builder.add_residual(event.row, event.k, event.value),
                       "provider packet accepts INT32 residuals")) return false;
        }
        const auto packet = builder.finish();
        if (!check(packet != nullptr, "INT32 provider packet is valid")) return false;
        const auto top_lane = balanced_radix_contract(GGML_GEMMINI_ACTIVATION_BITS).lane_capacity - 1;
        if (!check((packet->blocks.front().active_lane_mask & (uint16_t{1} << top_lane)) != 0,
                   "INT32_MAX retains its top carry lane")) return false;

        std::vector<OutputValue> expected(Fixture::rows * Fixture::columns, 0);
        for (const Event & event : events) {
            for (size_t j = 0; j < Fixture::columns; ++j) {
                const size_t block = j * 2 + event.k / kBlockSize;
                const int64_t code = static_cast<int64_t>((block * 17 + (event.k % kBlockSize) * 5) % 7) - 3;
                const int64_t scale = use_hp1 ? 4 : 3 + static_cast<int64_t>(block + 1);
                expected[event.row * Fixture::columns + j] +=
                    static_cast<int64_t>(event.value) * code * scale;
            }
        }
        Sim sim(im2p_sim_create());
        if (!check(sim != nullptr, "INT32 provider simulator exists")) return false;
        Correction actual;
        RmdExecutionMetrics metrics{};
        const RmdStatus status = execute_rmd_stripe_im2p(
            sim.get(), fixture.args, *packet, actual, &metrics);
        const auto * integer = std::get_if<BlockScaledInt64Correction>(&actual);
        if (!check(status == RmdStatus::success && integer != nullptr &&
                       integer->values == expected && metrics.im2p_dot_calls != 0,
                   "INT32 provider result matches direct weighted residuals")) {
            std::fprintf(stderr, "route=%s status=%s dots=%zu\n",
                         use_hp1 ? "HP1" : "H1", rmd_status_message(status), metrics.im2p_dot_calls);
            if (integer != nullptr && integer->values.size() == expected.size()) {
                for (size_t index = 0; index < expected.size(); ++index) {
                    if (integer->values[index] == expected[index]) continue;
                    std::fprintf(stderr, "index=%zu actual=%lld expected=%lld\n", index,
                                 static_cast<long long>(integer->values[index]),
                                 static_cast<long long>(expected[index]));
                    break;
                }
            }
            return false;
        }
    }
    std::puts("IM2P_PROVIDER int32-residuals status=success routes=H1,HP1 carry_lane=retained");
    return true;
}

bool run_group_rows() {
    struct Event { size_t row; size_t k; int32_t value; };
    const size_t columns = DIM + 1;
    Sim sim(im2p_sim_create());
    if (!check(sim != nullptr, "row-boundary provider simulator exists")) return false;
    for (const size_t rows : {size_t{1}, size_t{DIM - 1}, size_t{DIM}, size_t{DIM + 1}}) {
        for (const bool use_hp1 : {false, true}) {
            for (const size_t seed : {size_t{0}, size_t{1}}) {
                Fixture fixture(use_hp1, 2, rows, columns, seed);
                std::vector<Event> events;
                for (size_t row = 0; row < rows; ++row) {
                    events.push_back({row, 2, 65537});
                    events.push_back({row, 5, -2});
                    events.push_back({row, kBlockSize + 2, -65536});
                    events.push_back({row, kBlockSize + 7, 3});
                }
                events.push_back({0, 7, std::numeric_limits<int32_t>::max()});
                events.push_back({rows - 1, kBlockSize + 3, std::numeric_limits<int32_t>::min()});
                RmdStripeBuilder builder;
                builder.reset(31, 0, rows, Fixture::logical_k, columns,
                              GGML_GEMMINI_ACTIVATION_BITS);
                std::vector<OutputValue> expected(rows * columns, 0);
                for (const Event & event : events) {
                    if (!check(builder.add_residual(event.row, event.k, event.value),
                               "row-boundary residual accepted")) return false;
                    for (size_t j = 0; j < columns; ++j) {
                        const size_t block = j * 2 + event.k / kBlockSize;
                        const int64_t code = static_cast<int64_t>(
                            (block * 17 + (event.k % kBlockSize) * 5 + seed) % 7) - 3;
                        const int64_t scale = use_hp1 ? 4 : 4 + static_cast<int64_t>(block);
                        expected[event.row * columns + j] +=
                            static_cast<int64_t>(event.value) * code * scale;
                    }
                }
                const auto packet = builder.finish();
                if (!check(packet != nullptr, "row-boundary packet is valid")) return false;
                size_t expected_values = 0;
                size_t expected_tiles = 0;
                for (const auto & block : packet->blocks) {
                    if (GGML_GEMMINI_ACTIVATION_BITS < 16 &&
                        !check((block.active_lane_mask & uint16_t{2}) == 0,
                               "row-boundary packet retains sparse lane IDs")) return false;
                    for (const auto & group : block.groups) {
                        const size_t group_rows = align_up(group.row_ids.size(), kArrayDim);
                        expected_values += group_rows * group.padded_k_count;
                        const size_t kj_tiles = (group.padded_k_count / kArrayDim) *
                            (packet->j_padded / kArrayDim);
                        expected_tiles += (group_rows / kArrayDim) * kj_tiles;
                    }
                }
                if (!check(packet->activation_value_count == expected_values,
                           "physical group rows have padding only at the group tail")) return false;
                Correction streamed;
                RmdExecutionMetrics metrics{};
                const auto status = execute_rmd_stripe_im2p(
                    sim.get(), fixture.args, *packet, streamed, &metrics);
                const auto * values = std::get_if<BlockScaledInt64Correction>(&streamed);
                if (!check(status == RmdStatus::success && values != nullptr &&
                               values->values == expected,
                           "group rows match direct INT32 weighted residuals") ||
                    !check(metrics.stacked_i_tile_count == expected_tiles,
                           "provider executes the compact group row tile count")) {
                    std::fprintf(stderr, "rows=%zu route=%s seed=%zu status=%s\n",
                                 rows, use_hp1 ? "HP1" : "H1", seed, rmd_status_message(status));
                    return false;
                }
                CompressedOutput compressed;
                Correction composed;
                if (!check(execute_rmd_stripe_im2p(sim.get(), fixture.args, *packet, compressed) ==
                               RmdStatus::success &&
                               compose_rmd_output(*packet, compressed, composed) == RmdStatus::success,
                           "group rows also execute through compressed output")) return false;
                const auto * composed_values = std::get_if<BlockScaledInt64Correction>(&composed);
                if (!check(composed_values != nullptr && composed_values->values == expected,
                           "compressed group rows compose to direct INT32 weighted residuals")) return false;
                for (const auto & block : packet->blocks) {
                    for (size_t lane = 0; lane < block.active_lane_count; ++lane) {
                        for (size_t row = 0; row < block.rows_padded; ++row) {
                            for (size_t j = 0; j < packet->j_padded; ++j) {
                                if (row < rows && j < columns) continue;
                                const size_t index = block.output_value_offset + lane * block.lane_stride_values +
                                    row * packet->j_padded + j;
                                if (!check(compressed.values[index] == 0,
                                           "compressed row and J padding remains zero")) return false;
                            }
                        }
                    }
                }
                std::printf("IM2P_PROVIDER group-rows rows=%zu route=%s seed=%zu payload_values=%zu "
                            "actual_tiles=%zu dots=%zu cycles=%llu\n",
                            rows, use_hp1 ? "HP1" : "H1", seed, packet->activation_value_count,
                            metrics.stacked_i_tile_count, metrics.im2p_dot_calls,
                            static_cast<unsigned long long>(metrics.im2p_stats.work_total_cycles()));
            }
        }
    }
    return true;
}

bool run_native_code_edges() {
    constexpr size_t rows = 3;
    constexpr int32_t minimum = -(int32_t{1} << (GGML_GEMMINI_WEIGHT_BITS - 1));
    constexpr int32_t maximum = -minimum - 1;
    const auto weight_code = [](size_t column, size_t block, size_t k) -> int32_t {
        if (column == 0) return block == 0 ? minimum : maximum;
        const size_t pattern = (k + block + column) % 3;
        return pattern == 0 ? minimum : pattern == 1 ? maximum : 0;
    };
    const auto residual = [](size_t row, size_t k) -> int32_t {
        if (row == 0) return std::numeric_limits<int32_t>::min();
        if (row == 1) return std::numeric_limits<int32_t>::max();
        return k % 2 == 0 ? minimum : maximum;
    };
    Sim sim(im2p_sim_create());
    if (!check(sim != nullptr, "native-code edge simulator exists")) return false;
    for (const bool use_hp1 : {false, true}) {
        Fixture fixture(use_hp1, 2, rows);
        for (size_t j = 0; j < Fixture::columns; ++j) {
            for (size_t block = 0; block < 2; ++block) {
                for (size_t k = 0; k < kBlockSize; ++k) {
                    fixture.set_code(j * 2 + block, k, weight_code(j, block, k));
                }
            }
        }
        RmdStripeBuilder builder;
        builder.reset(37, 0, rows, Fixture::logical_k, Fixture::columns,
                      GGML_GEMMINI_ACTIVATION_BITS);
        std::array<__int128, rows * Fixture::columns> sums{};
        for (size_t row = 0; row < rows; ++row) {
            for (size_t k = 0; k < Fixture::logical_k; ++k) {
                const int32_t value = residual(row, k);
                if (!check(builder.add_residual(row, k, value),
                           "native-code edge residual accepted")) return false;
                for (size_t j = 0; j < Fixture::columns; ++j) {
                    const size_t block = k / kBlockSize;
                    const int64_t scale = use_hp1 ? 4 : 4 + static_cast<int64_t>(j * 2 + block);
                    sums[row * Fixture::columns + j] += static_cast<__int128>(value) *
                        weight_code(j, block, k % kBlockSize) * scale;
                }
            }
        }
        std::vector<OutputValue> expected;
        for (const __int128 value : sums) {
            if (!check(value >= std::numeric_limits<int64_t>::min() &&
                           value <= std::numeric_limits<int64_t>::max(),
                       "native-code edge oracle fits INT64")) return false;
            expected.push_back(static_cast<int64_t>(value));
        }
        const auto packet = builder.finish();
        if (!check(packet != nullptr, "native-code edge packet is valid")) return false;
        Correction streamed;
        RmdExecutionMetrics metrics{};
        if (!check(execute_rmd_stripe_im2p(sim.get(), fixture.args, *packet, streamed, &metrics) ==
                       RmdStatus::success,
                   "native-code edge streaming executes")) return false;
        const auto * streamed_values = std::get_if<BlockScaledInt64Correction>(&streamed);
        if (!check(streamed_values != nullptr && streamed_values->values == expected &&
                       metrics.im2p_dot_calls > 0 && metrics.im2p_stats.work_total_cycles() > 0 &&
                       metrics.ws_call_count == 0,
                   "native-code edge streaming equals independent raw-residual oracle")) return false;
        CompressedOutput compressed;
        Correction composed;
        if (!check(execute_rmd_stripe_im2p(sim.get(), fixture.args, *packet, compressed) ==
                       RmdStatus::success &&
                       compose_rmd_output(*packet, compressed, composed) == RmdStatus::success,
                   "native-code edge compressed output executes")) return false;
        const auto * composed_values = std::get_if<BlockScaledInt64Correction>(&composed);
        if (!check(composed_values != nullptr && composed_values->values == expected,
                   "native-code edge compressed output equals independent raw-residual oracle")) return false;
        const auto & block = packet->blocks.front();
        const uint8_t sign_lane = 32 / GGML_GEMMINI_ACTIVATION_BITS - 1;
        const auto lane = std::find(block.lane_ids.begin(),
            block.lane_ids.begin() + block.active_lane_count, sign_lane);
        if (!check(lane != block.lane_ids.begin() + block.active_lane_count,
                   "INT32_MIN sign lane is present")) return false;
        const size_t lane_position = static_cast<size_t>(lane - block.lane_ids.begin());
        const int64_t raw_dot = static_cast<int64_t>(kBlockSize) * minimum * minimum;
        if (!check(compressed.values[block.output_value_offset + lane_position * block.lane_stride_values] ==
                       raw_dot * 4,
                   "native signed edge dot is retained before radix composition")) return false;
        if (GGML_GEMMINI_ACTIVATION_BITS == 16 &&
            !check(raw_dot > std::numeric_limits<int32_t>::max(),
                   "A16 edge dot exceeds INT32 without saturation")) return false;
        std::printf("IM2P_PROVIDER native-code-edges width=%d route=%s range=%d..%d "
                    "raw_edge_dot=%lld dots=%zu cycles=%llu status=success\n",
                    GGML_GEMMINI_ACTIVATION_BITS, use_hp1 ? "HP1" : "H1", minimum, maximum,
                    static_cast<long long>(raw_dot), metrics.im2p_dot_calls,
                    static_cast<unsigned long long>(metrics.im2p_stats.work_total_cycles()));
    }
    return true;
}

bool run_fault(Im2pProviderTestFault fault, RmdStatus expected, const char * name,
               size_t minimum_attempts = 1,
               size_t maximum_attempts = std::numeric_limits<size_t>::max()) {
    Fixture fixture;
    Sim sim(im2p_sim_create());
    reset_im2p_provider_dot_attempts_for_test();
    CompressedOutput output{CompressedOutput::Domain::block_scaled_int64, 91, {7, -11}};
    RmdExecutionMetrics metrics{};
    metrics.packet_call_count = 73;
    metrics.im2p_dot_calls = 79;
    const RmdStatus status = fixture.packet && sim ? execute_rmd_stripe_im2p_for_test(
        sim.get(), fixture.args, *fixture.packet, output, &metrics, fault) : RmdStatus::execution_failed;
    const size_t attempts = im2p_provider_dot_attempts_for_test();
    Sim streaming_sim(im2p_sim_create());
    Correction correction = PreScaledFloat64Correction{{7.25, -11.5}};
    RmdExecutionMetrics streaming_metrics{};
    streaming_metrics.packet_call_count = 73;
    streaming_metrics.im2p_dot_calls = 79;
    reset_im2p_provider_dot_attempts_for_test();
    const RmdStatus streaming_status = fixture.packet && streaming_sim ?
        execute_rmd_stripe_im2p_for_test(streaming_sim.get(), fixture.args, *fixture.packet,
                                        correction, &streaming_metrics, fault) :
        RmdStatus::execution_failed;
    const size_t streaming_attempts = im2p_provider_dot_attempts_for_test();
    const bool ok = check(status == expected, name) &&
        check(unchanged(output, metrics), "provider failure is transactional") &&
        check(attempts >= minimum_attempts && attempts <= maximum_attempts,
              "provider failure stops at its deterministic compact-call boundary") &&
        check(streaming_status == expected && unchanged(correction, streaming_metrics),
              "streaming provider failure preserves correction and metrics") &&
        check(streaming_attempts >= minimum_attempts && streaming_attempts <= maximum_attempts,
              "streaming provider failure stops at its deterministic compact-call boundary");
    if (ok) std::printf("IM2P_PROVIDER %s status=%s attempts=%zu output_sentinel=unchanged metrics_sentinel=unchanged ws_calls=%zu\n",
                        name, rmd_status_message(status), attempts,
                        metrics.ws_call_count);
    return ok;
}

bool run_hp1_exp_63() {
    Fixture fixture(true, 2);
    fixture.hp1.back().m = 63;
    Sim sim(im2p_sim_create());
    CompressedOutput output{CompressedOutput::Domain::block_scaled_int64, 91, {7, -11}};
    RmdExecutionMetrics metrics{};
    metrics.packet_call_count = 73;
    metrics.im2p_dot_calls = 79;
    ggml::gemmini::quants::wreader::test_reset_weight_reader_counters();
    const RmdStatus status = fixture.packet && sim ? execute_rmd_stripe_im2p(
        sim.get(), fixture.args, *fixture.packet, output, &metrics) : RmdStatus::execution_failed;
    const bool ok = check(status == RmdStatus::overflow, "HP1 exponent 63 is typed overflow") &&
        check(ggml::gemmini::quants::wreader::test_weight_reader_code_address_resolutions() == 0,
              "invalid final weight block scale rejects before any code gather") &&
        check(unchanged(output, metrics), "HP1 overflow is transactional");
    if (ok) std::printf("IM2P_PROVIDER hp1-exp-63 status=overflow dot_calls=0 output_sentinel=unchanged metrics_sentinel=unchanged ws_calls=%zu\n",
                        metrics.ws_call_count);
    return ok;
}

ExsiaRouteRequest route(WeightFamily family, ResidualBackend backend) {
    return {true, GGML_GEMMINI_ACTIVATION_BITS, GGML_GEMMINI_WEIGHT_BITS,
            GGML_GEMMINI_ACTIVATION_BITS, GGML_GEMMINI_WEIGHT_BITS, true,
            PublicMode::full, family, backend, BuildIdentity::im2p_sim_ws};
}

bool run_route(std::string_view selected) {
    if (selected == "route-matched") {
        const auto result = ggml::gemmini::im2p_adapter::gate_route(
            route(WeightFamily::h1, ResidualBackend::compact_ws));
        const bool ok = check(result.ok(), "matched IM2P H1 compact route accepted");
        if (ok) std::puts("IM2P_ROUTE matched compact_ws=accepted backend=IM2P_SIM");
        return ok;
    }
    if (selected == "route-mismatch") {
        auto request = route(WeightFamily::hp1, ResidualBackend::compact_ws);
        request.artifact_activation_bits = request.activation_bits == 4 ? 8 : 4;
        const auto result = ggml::gemmini::im2p_adapter::gate_route(request);
        const bool ok = check(result.error == Error::invalid_contract,
                              "artifact mismatch rejected");
        if (ok) std::puts("IM2P_ROUTE mismatch=rejected before_execute=1");
        return ok;
    }
    const auto result = ggml::gemmini::im2p_adapter::gate_route(
        route(WeightFamily::h0, ResidualBackend::compact_ws));
    const bool ok = check(result.error == Error::unsupported_route,
                          "H0 compact route rejected");
    if (ok) std::puts("IM2P_ROUTE h0-compact=rejected before_execute=1");
    return ok;
}

} // namespace

int main(int argc, char ** argv) {
    std::string_view selected = "all";
    if (argc == 3 && std::strcmp(argv[1], "--case") == 0) selected = argv[2];
    else if (argc == 2 && std::strncmp(argv[1], "--case=", 7) == 0) selected = argv[1] + 7;
    else if (argc != 1) {
        std::fprintf(stderr, "usage: test-gemmini-rmd-im2p-provider [--case CASE]\n");
        return 2;
    }

    bool ok = true;
    if (selected == "all" || selected == "success") ok = run_success() && ok;
    if (selected == "all" || selected == "provider-read-failure") ok = run_fault(Im2pProviderTestFault::read_failure, RmdStatus::execution_failed, "provider-read-failure", 1, 1) && ok;
    if (selected == "all" || selected == "provider-write-failure") ok = run_fault(Im2pProviderTestFault::write_failure, RmdStatus::execution_failed, "provider-write-failure", 1, 1) && ok;
    if (selected == "all" || selected == "provider-watchdog") ok = run_fault(Im2pProviderTestFault::watchdog, RmdStatus::execution_failed, "provider-watchdog", 1, 1) && ok;
    if (selected == "all" || selected == "k-accumulation-overflow") ok = run_fault(Im2pProviderTestFault::k_accumulation_overflow, RmdStatus::overflow, "k-accumulation-overflow", 1) && ok;
    if (selected == "all" || selected == "block-scale-overflow") ok = run_fault(Im2pProviderTestFault::block_scale_overflow, RmdStatus::overflow, "block-scale-overflow", 1) && ok;
    if (selected == "all" || selected == "cancel-between-dots") ok = run_fault(Im2pProviderTestFault::cancel_after_first_dot, RmdStatus::execution_failed, "cancel-between-dots", 1, 1) && ok;
    if (selected == "all" || selected == "duplicate-output") ok = run_fault(Im2pProviderTestFault::duplicate_output, RmdStatus::execution_failed, "duplicate-output", 1, 1) && ok;
    if (selected == "all" || selected == "missing-output") ok = run_fault(Im2pProviderTestFault::missing_output, RmdStatus::invalid_packet, "missing-output") && ok;
    if (selected == "all" || selected == "output-index") ok = run_fault(Im2pProviderTestFault::output_index, RmdStatus::execution_failed, "output-index") && ok;
    if (selected == "all" || selected == "stats-overflow") ok = run_fault(Im2pProviderTestFault::stats_overflow, RmdStatus::overflow, "stats-overflow") && ok;
    if (selected == "all" || selected == "hp1-exp-62") ok = run_hp1_exp_62() && ok;
    if (selected == "all" || selected == "hp1-exp-63") ok = run_hp1_exp_63() && ok;
    if (selected == "all" || selected == "malformed-packet") ok = run_malformed_packet() && ok;
    if (selected == "all" || selected == "shared-preparation") ok = run_shared_preparation() && ok;
    if (selected == "all" || selected == "packet-merge-contract") ok = run_packet_merge_contract() && ok;
    if (selected == "all" || selected == "int32-residuals") ok = run_int32_residuals() && ok;
    if (selected == "all" || selected == "group-rows") ok = run_group_rows() && ok;
    if (selected == "all" || selected == "native-code-edges") ok = run_native_code_edges() && ok;
    if (selected == "all" || selected == "route-matched" || selected == "route-mismatch" || selected == "h0-compact-rejection") ok = run_route(selected == "all" ? "route-matched" : selected) && ok;

    constexpr std::array<std::string_view, 23> valid{{"all", "success", "provider-read-failure", "provider-write-failure", "provider-watchdog", "k-accumulation-overflow", "block-scale-overflow", "cancel-between-dots", "duplicate-output", "missing-output", "output-index", "stats-overflow", "hp1-exp-62", "hp1-exp-63", "malformed-packet", "shared-preparation", "packet-merge-contract", "int32-residuals", "group-rows", "native-code-edges", "route-matched", "route-mismatch", "h0-compact-rejection"}};
    const bool is_valid = std::find(valid.begin(), valid.end(), selected) != valid.end() || selected == "h0-compact-rejection";
    if (!is_valid) {
        std::fprintf(stderr, "unsupported test case: %.*s\n", static_cast<int>(selected.size()), selected.data());
        return 2;
    }
    return ok ? 0 : 1;
}
