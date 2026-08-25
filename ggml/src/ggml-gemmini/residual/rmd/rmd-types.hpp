#pragma once

// Width-native Residual Matrix Decomposition (RMD).
//
// Residuals use balanced radix 16, 256, or 65536 for native signed INT4, INT8,
// or INT16 digits. A digit index is a "lane" and contributes radix^lane to the
// reconstruction. Packets retain only lanes that contain a nonzero digit and
// group their payload by original weight block.

#include "../../ggml-gemmini-config.hpp"

#include <gemmini_params.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <variant>
#include <vector>

namespace ggml::gemmini::rmd {

constexpr size_t kArrayDim = DIM;
constexpr size_t kNativeWeightScaleGroup = 32;
constexpr size_t kBlockSize = kNativeWeightScaleGroup;
constexpr size_t kMaxLanes = 4; // legacy radix-256 API and Q8 execution contract
constexpr size_t kLegacyRadix256Lanes = kMaxLanes;
constexpr size_t kMaxNativeRadixLanes = 8;

// Residual capture intentionally supports the signed 21-bit outlier envelope.
// Lane capacities describe the fixed transport budget, not permission to admit
// wider synthetic values.
constexpr int32_t kSigned21Min = -(int32_t{1} << 20);
constexpr int32_t kSigned21Max = (int32_t{1} << 20) - 1;

static_assert(kArrayDim > 0, "Gemmini DIM must be positive");
static_assert(kBlockSize > 0, "native weight scale group must be positive");
static_assert(kBlockSize % kArrayDim == 0 || kArrayDim % kBlockSize == 0,
              "Gemmini DIM and native weight scale group must divide one another");

constexpr uint32_t kPacketVersion = 2;

// The compact RMD packet stores adjacent logical Q4 digits low nibble first as
// signed two's-complement INT4. IM2P model weights remain GGUF split-half,
// offset-binary Q4 and are decoded by the frontend before scalar simulation.
enum class DigitStorage : uint8_t {
    invalid = 0,
    packed_signed_int4 = 1,
    signed_int8 = 2,
    signed_int16 = 3,
};

enum class Int4Packing : uint8_t {
    none = 0,
    adjacent_low_nibble_first = 1,
};

constexpr DigitStorage digit_storage_for_bits(uint8_t digit_bits) {
    return digit_bits == 4 ? DigitStorage::packed_signed_int4 :
        digit_bits == 8 ? DigitStorage::signed_int8 :
        digit_bits == 16 ? DigitStorage::signed_int16 : DigitStorage::invalid;
}

using OutputValue = int64_t;

// CPU-direct H1/HP1 corrections have consumed the integer block factor but not
// the per-column floating factor. H0 corrections have consumed their arbitrary
// floating block factors in double precision. The variant keeps those domains
// distinct until the common composition path handles them.
struct BlockScaledInt64Correction {
    std::vector<OutputValue> values;
};

struct PreScaledFloat64Correction {
    std::vector<double> values;
};

// The correction remains tagged through execution, radix composition, and the
// final destination merge. Integer-block-scaled and pre-scaled floating values
// are intentionally not implicitly convertible to one another.
using Correction = std::variant<BlockScaledInt64Correction, PreScaledFloat64Correction>;
using DirectOutput = Correction;

inline size_t correction_size(const Correction & correction) {
    return std::visit([](const auto & typed) { return typed.values.size(); }, correction);
}

inline bool correction_empty(const Correction & correction) {
    return correction_size(correction) == 0;
}

inline size_t align_up(size_t value, size_t alignment) {
    if (alignment == 0 || value == 0)
        return value;
    const size_t remainder = value % alignment;
    if (remainder == 0)
        return value;
    const size_t padding = alignment - remainder;
    return value > std::numeric_limits<size_t>::max() - padding ? 0 : value + padding;
}

// One original weight block that carries at least one residual digit in this stripe.
struct BlockDescriptor {
    uint32_t block_id = 0;        // original weight block index (K / kBlockSize)
    uint32_t global_k_begin = 0;  // block_id * kBlockSize

    uint16_t compact_k_count = 0; // selected K within this block
    uint16_t padded_k_count = 0;  // compact_k_count aligned to kArrayDim

    uint8_t active_lane_mask = 0; // bit l set when lane l carries a nonzero digit
    uint8_t active_lane_count = 0;
    std::array<uint8_t, kMaxNativeRadixLanes> lane_ids{}; // position -> lane id

    uint32_t k_index_offset = 0;    // into StripePacket::k_indices (block-local K)
    uint32_t activation_offset = 0; // logical digit offset before native storage
    uint32_t activation_byte_offset = 0; // byte offset into the selected payload
    uint32_t activation_byte_count = 0;  // byte extent owned by this block

    uint32_t output_value_offset = 0; // into CompressedOutput::values
    uint16_t rows_padded = 0;         // row_count aligned to kArrayDim
    uint32_t lane_stride_values = 0;  // rows_padded * j_padded
};

// Exactly one member is populated according to StripePacket::digit_storage.
// INT16 values live in a typed vector so their alignment is guaranteed. The
struct ActivationPayload {
    std::vector<uint8_t> packed_int4;
    std::vector<int8_t> signed_int8;
    std::vector<int16_t> signed_int16;

    friend bool operator==(const ActivationPayload & left,
                           const ActivationPayload & right) {
        return left.packed_int4 == right.packed_int4 &&
            left.signed_int8 == right.signed_int8 &&
            left.signed_int16 == right.signed_int16;
    }

    friend bool operator!=(const ActivationPayload & left,
                           const ActivationPayload & right) {
        return !(left == right);
    }
};

// Immutable, self-contained description of one stripe's residual work.
// The packet never borrows ExSIA slot memory: it owns every buffer it exposes.
struct StripePacket {
    uint32_t version = kPacketVersion;
    uint8_t digit_bits = 8;
    uint8_t lane_capacity = 4;
    DigitStorage digit_storage = DigitStorage::signed_int8;
    Int4Packing int4_packing = Int4Packing::none;

    size_t stripe_id = 0;
    size_t row_begin = 0;
    size_t row_count = 0;
    size_t logical_k = 0;
    size_t logical_j = 0;
    size_t j_padded = 0;

    size_t block_size = kBlockSize;
    size_t array_dim = kArrayDim;

    std::vector<BlockDescriptor> blocks;
    std::vector<uint16_t> k_indices; // block-local K, ascending inside each block
    // block / lane position / padded row / padded K in selected native storage
    ActivationPayload stacked_activation;
    size_t activation_value_count = 0; // decoded values, including DIM padding
    size_t residual_event_count = 0;   // nonzero source residuals before radix expansion

    size_t total_output_values = 0;
};

using StripePacketHandle = std::shared_ptr<const StripePacket>;

struct CompressedOutput {
    enum class Domain : uint8_t {
        block_scaled_int64,
    };

    Domain domain = Domain::block_scaled_int64;
    size_t j_padded = 0;
    std::vector<OutputValue> values;
};

enum class RmdStatus : uint8_t {
    success,
    invalid_arguments,
    invalid_packet,
    residual_too_wide,   // outside the supported width-native residual envelope
    unsupported_route,   // route cannot satisfy the exact result contract
    overflow,
    allocation_failure,
    execution_failed,
};

const char * rmd_status_message(RmdStatus status);

// Width-native numerical contract shared by packet construction and consumers.
struct BalancedRadixContract {
    uint32_t radix = 0;
    uint8_t lane_capacity = 0;
    int32_t digit_min = 0;
    int32_t digit_max = 0;
};

BalancedRadixContract balanced_radix_contract(uint8_t operand_bits);

struct NativeBalancedDigits {
    std::array<int32_t, kMaxNativeRadixLanes> digits{};
    uint32_t radix = 0;
    uint8_t lane_capacity = 0;
    uint8_t active_lane_count = 0;

    bool operator==(const NativeBalancedDigits & other) const {
        return digits == other.digits && radix == other.radix &&
            lane_capacity == other.lane_capacity &&
            active_lane_count == other.active_lane_count;
    }
};

// Decomposes only values in the supported signed 21-bit envelope. Both calls
// are transactional: failure leaves the caller-provided output unchanged.
RmdStatus decompose_balanced_radix(int32_t residual,
                                   uint8_t operand_bits,
                                   NativeBalancedDigits & out);
RmdStatus compose_balanced_radix(const NativeBalancedDigits & digits,
                                 int64_t & out);

// Legacy radix-256 packet digits. `digits` is always kMaxLanes wide; unused
// high lanes are zero.
struct BalancedDigits {
    std::array<int8_t, kMaxLanes> digits{};
    uint8_t lane_mask = 0;
};

// Returns false when the value needs a fifth digit (explicit failure, never a
// silent truncation).
bool decompose_balanced_radix256(int32_t residual, BalancedDigits & out);

// Reconstructs a residual from its digits; used by tests and the reference path.
int64_t compose_balanced_radix256(const BalancedDigits & digits);

}
