#include "../ggml/src/ggml-gemmini/residual/rmd/rmd-builder.hpp"
#include "../ggml/src/ggml-gemmini/residual/residual-capture.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <type_traits>

namespace {

using namespace ggml::gemmini::rmd;

bool check(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
    }
    return condition;
}

bool test_balanced_radix_decomposition() {
    constexpr std::array<int32_t, 13> values = {
        std::numeric_limits<int32_t>::min(), -16777217, -129, -128, -1, 0, 1,
        127, 128, 255, 256, 16777216, 2139062143,
    };
    for (const int32_t value : values) {
        BalancedDigits digits{};
        if (!check(decompose_balanced_radix256(value, digits), "balanced value decomposes") ||
            !check(compose_balanced_radix256(digits) == value, "balanced digits round-trip")) {
            return false;
        }
    }
    return true;
}

bool test_q4_nonzero_fails_explicitly() {
    BalancedDigits digits{};
    digits.digits.fill(1);
    digits.lane_mask = 0x0f;
    const bool decomposed = decompose_balanced_radix256(std::numeric_limits<int32_t>::max(), digits);

    RmdStripeBuilder builder;
    builder.reset(0, 0, 1, 1, 1);
    const bool added = builder.add_residual(0, 0, std::numeric_limits<int32_t>::max());
    return check(!decomposed, "q4 != 0 rejects decomposition") &&
        check(digits.lane_mask == 0, "failed decomposition clears output") &&
        check(!added && builder.status() == RmdStatus::residual_too_wide,
              "builder exposes q4 failure status");
}

bool test_EXPLICIT_BLOCK_ID_DIM_PADDING() {
    RmdStripeBuilder builder;
    builder.reset(7, 9, 3, 64, 17);
    if (!builder.add_residual(0, 31, 1) ||
        !builder.add_residual(1, 32, 65536) ||
        !builder.add_residual(2, 33, 1)) {
        return check(false, "boundary residuals accepted");
    }
    const StripePacketHandle packet = builder.finish();
    if (!check(packet != nullptr, "boundary packet built") ||
        !check(packet->blocks.size() == 2, "block boundary stays explicit")) {
        return false;
    }

    const BlockDescriptor & first = packet->blocks[0];
    const BlockDescriptor & second = packet->blocks[1];
    bool ok = true;
    ok = check(first.block_id == 0 && first.global_k_begin == 0, "first block id") && ok;
    ok = check(second.block_id == 1 && second.global_k_begin == kBlockSize, "second block id") && ok;
    ok = check(second.active_lane_mask == 0x05 && second.active_lane_count == 2,
               "lane gap remains sparse") && ok;
    ok = check(second.lane_ids[0] == 0 && second.lane_ids[1] == 2,
               "lane ids preserve radix places") && ok;
    ok = check(first.rows_padded == kArrayDim && second.rows_padded == kArrayDim,
               "rows use DIM padding") && ok;
    ok = check(first.padded_k_count == kArrayDim && second.padded_k_count == kArrayDim,
               "each block uses independent DIM padding") && ok;
    ok = check(packet->j_padded == 2 * kArrayDim, "output columns use DIM padding") && ok;
    ok = check(validate_packet(*packet) == RmdStatus::success, "built packet validates") && ok;

    StripePacket malformed = *packet;
    malformed.stacked_activation[first.activation_offset + first.compact_k_count] = 1;
    ok = check(validate_packet(malformed) == RmdStatus::invalid_packet,
               "nonzero K padding rejected") && ok;
    malformed = *packet;
    malformed.stacked_activation[first.activation_offset +
        packet->row_count * first.padded_k_count] = 1;
    return check(validate_packet(malformed) == RmdStatus::invalid_packet,
                 "nonzero row padding rejected") && ok;
}

bool test_empty_residual_is_empty_success() {
    RmdStripeBuilder builder;
    builder.reset(0, 0, 1, 1, 1);
    return check(builder.add_residual(0, 0, 0), "zero residual accepted") &&
        check(builder.empty(), "zero residual emits no entries") &&
        check(builder.finish() == nullptr, "empty residual emits no packet") &&
        check(builder.status() == RmdStatus::success, "empty residual is successful");
}

bool test_padding_overflow_fails() {
    RmdStripeBuilder builder;
    builder.reset(0, 0, 1, 1, std::numeric_limits<size_t>::max());
    if (!builder.add_residual(0, 0, 1)) {
        return check(false, "overflow fixture residual accepted");
    }
    return check(builder.finish() == nullptr, "padding overflow emits no packet") &&
        check(builder.status() == RmdStatus::overflow, "padding overflow is explicit");
}

}

bool test_cpu_capture_is_canonical_and_packet_free() {
    using namespace ggml::gemmini::residual;
    TimedResidualCapture capture(ResidualRoute::cpu_direct);
    capture.reset(3, 7, 2, 64, 17);
    if (!capture.add_residual(1, 33, -129) ||
        !capture.add_residual(0, 31, 256) ||
        !capture.add_residual(0, 2, 1)) {
        return check(false, "CPU residual events accepted");
    }
    const ResidualStripePayload payload = capture.finish();
    if (!check(payload.direct != nullptr, "CPU creates a direct payload") ||
        !check(payload.packet == nullptr, "CPU creates no packet") ||
        !check(payload.direct->events.size() == 3, "CPU retains every original event")) {
        return false;
    }
    const auto &events = payload.direct->events;
    return check(events[0] == ResidualEvent{0, 2, 1}, "CPU event order row 0 k 2") &&
        check(events[1] == ResidualEvent{0, 31, 256}, "CPU event order row 0 k 31") &&
        check(events[2] == ResidualEvent{1, 33, -129}, "CPU event order row 1 k 33");
}

bool test_ws_capture_preserves_packet_contract() {
    using namespace ggml::gemmini::residual;
    RmdStripeBuilder legacy;
    legacy.reset(3, 7, 2, 64, 17);
    legacy.add_residual(1, 33, -129);
    legacy.add_residual(0, 31, 256);
    legacy.add_residual(0, 2, 1);
    const StripePacketHandle expected = legacy.finish();

    TimedResidualCapture capture(ResidualRoute::ws_packet);
    capture.reset(3, 7, 2, 64, 17);
    capture.add_residual(1, 33, -129);
    capture.add_residual(0, 31, 256);
    capture.add_residual(0, 2, 1);
    const ResidualStripePayload payload = capture.finish();
    if (!check(payload.direct == nullptr, "WS creates no event payload") ||
        !check(payload.packet != nullptr, "WS creates a packet") ||
        !check(expected != nullptr, "legacy packet fixture built")) {
        return false;
    }
    return check(payload.packet->k_indices == expected->k_indices,
                 "WS K-index packet bytes preserved") &&
        check(payload.packet->stacked_activation == expected->stacked_activation,
              "WS activation packet bytes preserved");
}

bool test_empty_capture_and_single_sink_selection() {
    using namespace ggml::gemmini::residual;
    TimedResidualCapture cpu(ResidualRoute::cpu_direct);
    cpu.reset(0, 0, 1, 1, 1);
    const ResidualStripePayload cpu_empty = cpu.finish();
    TimedResidualCapture ws(ResidualRoute::ws_packet);
    ws.reset(0, 0, 1, 1, 1);
    const ResidualStripePayload ws_empty = ws.finish();
    return check(cpu.holds_cpu_sink() && !cpu.holds_ws_sink(),
                 "CPU selection instantiates only CPU sink") &&
        check(ws.holds_ws_sink() && !ws.holds_cpu_sink(),
              "WS selection instantiates only WS sink") &&
        check(cpu_empty.empty() && ws_empty.empty(), "empty stripes produce no work") &&
        check(cpu_empty.capture_ns == 0 && ws_empty.capture_ns == 0,
              "empty stripes skip timed finish work");
}

bool test_direct_payload_slicing_and_ownership() {
    using namespace ggml::gemmini::residual;
    static_assert(std::is_same_v<DirectStripePayloadHandle,
                                std::shared_ptr<const DirectStripePayload>>,
                  "direct payload ownership must be immutable");
    TimedResidualCapture first(ResidualRoute::cpu_direct);
    first.reset(4, 10, 2, 64, 17);
    first.add_residual(0, 4, 11);
    first.add_residual(1, 5, 12);
    TimedResidualCapture second(ResidualRoute::cpu_direct);
    second.reset(5, 12, 2, 64, 17);
    second.add_residual(0, 6, 13);
    second.add_residual(1, 7, 14);
    std::vector<DirectStripePayloadHandle> inputs{
        first.finish().direct, second.finish().direct};

    RmdStatus status = RmdStatus::success;
    DirectStripePayloadHandle identity = slice_direct_payloads(inputs, 10, 12, 4, status);
    if (!check(status == RmdStatus::success && identity == inputs[0],
               "exact direct slice preserves payload identity")) {
        return false;
    }
    DirectStripePayloadHandle slice = slice_direct_payloads(inputs, 11, 13, 9, status);
    if (!check(status == RmdStatus::success && slice != nullptr,
               "direct payload slice succeeds") ||
        !check(slice->row_begin == 11 && slice->row_count == 2,
               "direct payload slice identity")) {
        return false;
    }
    return check(slice->events.size() == 2, "slice keeps matching events") &&
        check(slice->events[0] == ResidualEvent{0, 5, 12}, "slice renormalizes first local row") &&
        check(slice->events[1] == ResidualEvent{1, 6, 13}, "slice renormalizes second local row") &&
        check(inputs[0]->events[1] == ResidualEvent{1, 5, 12},
              "slice does not mutate source ownership");
}


bool test_direct_payload_validation_rejects_malformed_contracts() {
    using namespace ggml::gemmini::residual;
    auto valid = [] {
        DirectStripePayload payload;
        payload.stripe_id = 1;
        payload.row_begin = 4;
        payload.row_count = 2;
        payload.logical_k = 8;
        payload.logical_j = 3;
        payload.events = {{0, 1, 7}, {1, 2, -9}};
        return payload;
    };

    DirectStripePayload malformed = valid();
    malformed.row_count = 0;
    bool ok = check(validate_direct_payload(malformed) == RmdStatus::invalid_packet,
                    "direct validator rejects zero dimensions");
    malformed = valid();
    malformed.row_begin = std::numeric_limits<size_t>::max();
    ok = check(validate_direct_payload(malformed) == RmdStatus::invalid_packet,
               "direct validator rejects row interval overflow") && ok;
    malformed = valid();
    malformed.events[0].local_row = malformed.row_count;
    ok = check(validate_direct_payload(malformed) == RmdStatus::invalid_packet,
               "direct validator rejects out-of-range row") && ok;
    malformed = valid();
    malformed.events[0].original_k = malformed.logical_k;
    ok = check(validate_direct_payload(malformed) == RmdStatus::invalid_packet,
               "direct validator rejects out-of-range K") && ok;
    malformed = valid();
    malformed.events[0].residual = 0;
    ok = check(validate_direct_payload(malformed) == RmdStatus::invalid_packet,
               "direct validator rejects zero residual") && ok;
    malformed = valid();
    std::swap(malformed.events[0], malformed.events[1]);
    ok = check(validate_direct_payload(malformed) == RmdStatus::invalid_packet,
               "direct validator rejects unsorted keys") && ok;
    malformed = valid();
    malformed.events[1] = malformed.events[0];
    return check(validate_direct_payload(malformed) == RmdStatus::invalid_packet,
                 "direct validator rejects duplicate keys") && ok;
}

bool test_exact_slice_validates_payload_and_dimensions() {
    using namespace ggml::gemmini::residual;
    auto malformed = std::make_shared<DirectStripePayload>();
    malformed->stripe_id = 7;
    malformed->row_begin = 10;
    malformed->row_count = 2;
    malformed->logical_k = 8;
    malformed->logical_j = 3;
    malformed->events = {{1, 2, 4}, {0, 1, 3}};
    RmdStatus status = RmdStatus::success;
    const auto exact = slice_direct_payloads({malformed}, 10, 12, 7, status);
    if (!check(exact == nullptr && status == RmdStatus::invalid_packet,
               "exact slice validates before returning identity")) {
        return false;
    }

    auto first = std::make_shared<DirectStripePayload>();
    first->stripe_id = 1;
    first->row_begin = 0;
    first->row_count = 2;
    first->logical_k = 8;
    first->logical_j = 3;
    first->events = {{0, 1, 2}};
    auto second = std::make_shared<DirectStripePayload>();
    second->stripe_id = 2;
    second->row_begin = 2;
    second->row_count = 2;
    second->logical_k = 16;
    second->logical_j = 3;
    second->events = {{0, 2, 3}};
    const auto mixed = slice_direct_payloads({first, second}, 1, 3, 9, status);
    return check(mixed == nullptr && status == RmdStatus::invalid_packet,
                 "slice rejects overlapping payloads with mixed dimensions");
}

bool test_direct_builder_rejects_row_interval_overflow() {
    using namespace ggml::gemmini::residual;
    DirectStripeBuilder builder;
    builder.reset(0, std::numeric_limits<size_t>::max(), 2, 8, 3);
    return check(builder.status() == RmdStatus::overflow,
                 "direct builder reset rejects row interval overflow") &&
        check(builder.finish() == nullptr, "overflow builder cannot finish");
}

int main() {
    const bool ok = test_balanced_radix_decomposition() &&
        test_q4_nonzero_fails_explicitly() &&
        test_EXPLICIT_BLOCK_ID_DIM_PADDING() &&
        test_empty_residual_is_empty_success() &&
        test_padding_overflow_fails() &&
        test_cpu_capture_is_canonical_and_packet_free() &&
        test_ws_capture_preserves_packet_contract() &&
        test_empty_capture_and_single_sink_selection() &&
        test_direct_payload_slicing_and_ownership() &&
        test_direct_payload_validation_rejects_malformed_contracts() &&
        test_exact_slice_validates_payload_and_dimensions() &&
        test_direct_builder_rejects_row_interval_overflow();
    if (ok) {
        std::puts("PASS: balanced radix, q4 failure, block boundaries, lane gaps, padding, empty residual, overflow");
    }
    return ok ? 0 : 1;
}
