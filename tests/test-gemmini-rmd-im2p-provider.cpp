#include "../ggml/src/ggml-gemmini/ggml-gemmini-im2p.hpp"
#include "../ggml/src/ggml-gemmini/residual/rmd/rmd-builder.hpp"
#include "../ggml/src/ggml-gemmini/residual/rmd/rmd-executor.hpp"
#include "../ggml/src/ggml-gemmini/residual/rmd/rmd-im2p-executor.hpp"
#include "../ggml/src/ggml-gemmini/ggml-gemmini-args.h"

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
    static constexpr size_t block_count = columns * 2;

    ggml_gemmini_args_t args{};
#if GGML_GEMMINI_WEIGHT_BITS == 4
    std::array<block_q4_h1, block_count> h1{};
    std::array<block_q4_hp1, block_count> hp1{};
#elif GGML_GEMMINI_WEIGHT_BITS == 8
    std::array<block_q8_h1, block_count> h1{};
    std::array<block_q8_hp1, block_count> hp1{};
#else
    std::array<block_q16_h1, block_count> h1{};
    std::array<block_q16_hp1, block_count> hp1{};
#endif
    StripePacketHandle packet;

    explicit Fixture(bool use_hp1 = false, int16_t exponent = 2) {
        args.I = rows;
        args.J = columns;
        args.K = logical_k;
        args.block_size_k = kBlockSize;
        args.native_block_count = block_count;
        args.native_blocks_per_row = 2;
        args.A.allocate(rows, logical_k, GGML_GEMMINI_ACTIVATION_BITS);

        for (size_t block_index = 0; block_index < block_count; ++block_index) {
            for (size_t k = 0; k < kBlockSize; ++k) {
                const int32_t code = static_cast<int32_t>((block_index * 17 + k * 5) % 7) - 3;
#if GGML_GEMMINI_WEIGHT_BITS == 4
                const size_t byte = k % (kBlockSize / 2);
                const uint8_t nibble = static_cast<uint8_t>(code + 8);
                auto set_code = [&](auto & block) {
                    if (k < kBlockSize / 2) block.qs[byte] = static_cast<uint8_t>((block.qs[byte] & 0xf0u) | nibble);
                    else block.qs[byte] = static_cast<uint8_t>((block.qs[byte] & 0x0fu) | (nibble << 4));
                };
                set_code(h1[block_index]);
                set_code(hp1[block_index]);
#else
                using H1Code = std::remove_reference_t<decltype(h1[block_index].qs[k])>;
                using Hp1Code = std::remove_reference_t<decltype(hp1[block_index].qs[k])>;
                h1[block_index].qs[k] = static_cast<H1Code>(code);
                hp1[block_index].qs[k] = static_cast<Hp1Code>(code);
#endif
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
        args.q8_h1_rows = use_hp1 ? 0 : columns;
        args.blocks_per_row = use_hp1 ? 0 : 2;
        args.q8_hp1_blocks = use_hp1 ? hp1.data() : nullptr;
        args.q8_hp1_block_count = use_hp1 ? hp1.size() : 0;
        args.q8_hp1_blocks_per_row = use_hp1 ? 2 : 0;
#else
        args.weight_format = use_hp1 ? ggml_gemmini_args_t::im2p_weight_format_t::q16_hp1 : ggml_gemmini_args_t::im2p_weight_format_t::q16_h1;
        args.q16_h1_blocks = use_hp1 ? nullptr : h1.data();
        args.q16_hp1_blocks = use_hp1 ? hp1.data() : nullptr;
#endif
        args.native_weight_bytes = use_hp1 ? sizeof(hp1) : sizeof(h1);

        RmdStripeBuilder builder;
        builder.reset(19, 7, rows, logical_k, columns, GGML_GEMMINI_ACTIVATION_BITS);
        // Keep more than DIM compact positions in one radix lane so the
        // cancellation and checked K-accumulation probes cross a real dot boundary.
        for (size_t k = 0; k < DIM + 3; ++k) {
            builder.add_residual(k % rows, k, 1);
        }
        for (size_t k : {size_t{0}, size_t{1}, size_t{DIM}, size_t{31}}) {
            builder.add_residual((k + 3) % rows, kBlockSize + k, -1);
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
    const RmdStatus status = fixture.packet && sim ? execute_rmd_stripe_im2p(
        sim.get(), fixture.args, *fixture.packet, actual, &actual_metrics) : RmdStatus::execution_failed;
    const RmdStatus routed_status = fixture.packet ? execute_rmd_stripe_ws(
        fixture.args, *fixture.packet, routed, &routed_metrics) : RmdStatus::invalid_packet;
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
        check(routed.values == expected.values && routed_metrics.im2p_dot_calls > 0 &&
                  routed_metrics.im2p_stats.work_total_cycles() > 0 &&
                  routed_metrics.ws_call_count == 0,
              "public compact route propagates provider stats without fallback");
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

bool run_zero_residual() {
    RmdStripeBuilder builder;
    builder.reset(23, 0, 1, kBlockSize, 1, GGML_GEMMINI_ACTIVATION_BITS);
    const StripePacketHandle packet = builder.finish();
    const bool ok = check(packet == nullptr && builder.status() == RmdStatus::success,
                          "zero residual creates no provider work");
    if (ok) std::puts("IM2P_PROVIDER zero-residual packet=empty dot_calls=0 ws_calls=0");
    return ok;
}

bool run_signed21_overflow() {
    RmdStripeBuilder builder;
    builder.reset(29, 0, 1, kBlockSize, 1,
                  GGML_GEMMINI_ACTIVATION_BITS);
    reset_im2p_provider_dot_attempts_for_test();
    CompressedOutput output{CompressedOutput::Domain::block_scaled_int64,
                            91, {7, -11}};
    RmdExecutionMetrics metrics{};
    metrics.packet_call_count = 73;
    metrics.im2p_dot_calls = 79;
    const bool negative_rejected =
        !builder.add_residual(0, 0, kSigned21Min - 1) &&
        builder.status() == RmdStatus::residual_too_wide &&
        builder.finish() == nullptr;

    RmdStripeBuilder positive;
    positive.reset(30, 0, 1, kBlockSize, 1,
                   GGML_GEMMINI_ACTIVATION_BITS);
    const bool positive_rejected =
        !positive.add_residual(0, 0, kSigned21Max + 1) &&
        positive.status() == RmdStatus::residual_too_wide &&
        positive.finish() == nullptr;
    const bool ok = check(negative_rejected && positive_rejected,
                          "signed-21 overflow rejects both extrema") &&
        check(im2p_provider_dot_attempts_for_test() == 0 &&
                  unchanged(output, metrics),
              "signed-21 overflow is rejected before provider or mutation");
    if (ok) {
        std::puts("IM2P_PROVIDER signed21-overflow status=residual_too_wide "
                  "bounds=-1048576:1048575 attempts=0 dot_calls=0 "
                  "output_sentinel=unchanged metrics_sentinel=unchanged "
                  "ws_calls=0");
    }
    return ok;
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
    const bool ok = check(status == expected, name) &&
        check(unchanged(output, metrics), "provider failure is transactional") &&
        check(attempts >= minimum_attempts && attempts <= maximum_attempts,
              "provider failure stops at its deterministic compact-call boundary");
    if (ok) std::printf("IM2P_PROVIDER %s status=%s attempts=%zu output_sentinel=unchanged metrics_sentinel=unchanged ws_calls=%zu\n",
                        name, rmd_status_message(status), attempts,
                        metrics.ws_call_count);
    return ok;
}

bool run_hp1_exp_63() {
    Fixture fixture(true, 63);
    Sim sim(im2p_sim_create());
    CompressedOutput output{CompressedOutput::Domain::block_scaled_int64, 91, {7, -11}};
    RmdExecutionMetrics metrics{};
    metrics.packet_call_count = 73;
    metrics.im2p_dot_calls = 79;
    const RmdStatus status = fixture.packet && sim ? execute_rmd_stripe_im2p(
        sim.get(), fixture.args, *fixture.packet, output, &metrics) : RmdStatus::execution_failed;
    const bool ok = check(status == RmdStatus::overflow, "HP1 exponent 63 is typed overflow") &&
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
    if (selected == "all" || selected == "k-accumulation-overflow") ok = run_fault(Im2pProviderTestFault::k_accumulation_overflow, RmdStatus::overflow, "k-accumulation-overflow", 2) && ok;
    if (selected == "all" || selected == "block-scale-overflow") ok = run_fault(Im2pProviderTestFault::block_scale_overflow, RmdStatus::overflow, "block-scale-overflow", 1) && ok;
    if (selected == "all" || selected == "cancel-between-dots") ok = run_fault(Im2pProviderTestFault::cancel_after_first_dot, RmdStatus::execution_failed, "cancel-between-dots", 1, 1) && ok;
    if (selected == "all" || selected == "duplicate-output") ok = run_fault(Im2pProviderTestFault::duplicate_output, RmdStatus::execution_failed, "duplicate-output", 1, 1) && ok;
    if (selected == "all" || selected == "missing-output") ok = run_fault(Im2pProviderTestFault::missing_output, RmdStatus::invalid_packet, "missing-output") && ok;
    if (selected == "all" || selected == "output-index") ok = run_fault(Im2pProviderTestFault::output_index, RmdStatus::execution_failed, "output-index") && ok;
    if (selected == "all" || selected == "stats-overflow") ok = run_fault(Im2pProviderTestFault::stats_overflow, RmdStatus::overflow, "stats-overflow") && ok;
    if (selected == "all" || selected == "hp1-exp-62") ok = run_hp1_exp_62() && ok;
    if (selected == "all" || selected == "hp1-exp-63") ok = run_hp1_exp_63() && ok;
    if (selected == "all" || selected == "malformed-packet") ok = run_malformed_packet() && ok;
    if (selected == "all" || selected == "zero-residual") ok = run_zero_residual() && ok;
    if (selected == "all" || selected == "signed21-overflow") ok = run_signed21_overflow() && ok;
    if (selected == "all" || selected == "route-matched" || selected == "route-mismatch" || selected == "h0-compact-rejection") ok = run_route(selected == "all" ? "route-matched" : selected) && ok;

    constexpr std::array<std::string_view, 20> valid{{"all", "success", "provider-read-failure", "provider-write-failure", "provider-watchdog", "k-accumulation-overflow", "block-scale-overflow", "cancel-between-dots", "duplicate-output", "missing-output", "output-index", "stats-overflow", "hp1-exp-62", "hp1-exp-63", "malformed-packet", "zero-residual", "signed21-overflow", "route-matched", "route-mismatch", "h0-compact-rejection"}};
    const bool is_valid = std::find(valid.begin(), valid.end(), selected) != valid.end() || selected == "h0-compact-rejection";
    if (!is_valid) {
        std::fprintf(stderr, "unsupported test case: %.*s\n", static_cast<int>(selected.size()), selected.data());
        return 2;
    }
    return ok ? 0 : 1;
}
