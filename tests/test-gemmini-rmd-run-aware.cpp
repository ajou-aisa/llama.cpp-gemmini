#include "../ggml/src/ggml-gemmini/residual/rmd/rmd-builder.hpp"
#include "../ggml/src/ggml-gemmini/residual/rmd/rmd-run-aware.hpp"
#include "../ggml/src/ggml-gemmini/ggml-gemmini-args.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <type_traits>
#include <vector>

using namespace ggml::gemmini::rmd;

namespace {

constexpr size_t kRows = 3;
constexpr size_t kColumns = 2;
constexpr size_t kBlocksPerRow = 4;
constexpr size_t kLogicalK = kBlocksPerRow * kBlockSize;

#if GGML_GEMMINI_WEIGHT_BITS == 4
using Hp1Block = block_q4_hp1;
#elif GGML_GEMMINI_WEIGHT_BITS == 8
using Hp1Block = block_q8_hp1;
#else
using Hp1Block = block_q16_hp1;
#endif

bool check(bool condition, const char *message) {
    if (!condition) std::fprintf(stderr, "FAIL: %s\n", message);
    return condition;
}

int32_t source_code(size_t column, size_t original_block, size_t local_k) {
#if GGML_GEMMINI_WEIGHT_BITS == 4
    return static_cast<int32_t>((column * 5 + original_block * 3 + local_k) % 15) - 7;
#elif GGML_GEMMINI_WEIGHT_BITS == 8
    return static_cast<int32_t>(column * 31 + original_block * 7 + local_k) - 40;
#else
    return static_cast<int32_t>(column * 1000 + original_block * 100 + local_k) - 700;
#endif
}

void set_code(Hp1Block &block, size_t local_k, int32_t code) {
#if GGML_GEMMINI_WEIGHT_BITS == 4
    const size_t byte = local_k % (kBlockSize / 2);
    const uint8_t nibble = static_cast<uint8_t>(code + 8);
    if (local_k < kBlockSize / 2)
        block.qs[byte] = static_cast<uint8_t>((block.qs[byte] & 0xf0u) | nibble);
    else
        block.qs[byte] = static_cast<uint8_t>((block.qs[byte] & 0x0fu) | (nibble << 4));
#else
    using Code = std::remove_reference_t<decltype(block.qs[local_k])>;
    block.qs[local_k] = static_cast<Code>(code);
#endif
}

} // namespace

int main() {
    ggml_gemmini_args_t args{};
    args.I = kRows;
    args.J = kColumns;
    args.K = kLogicalK;
    args.block_size_k = kBlockSize;
    args.native_blocks_per_row = kBlocksPerRow;

    std::vector<Hp1Block> source(kColumns * kBlocksPerRow);
    for (size_t column = 0; column < kColumns; ++column) {
        for (size_t block = 0; block < kBlocksPerRow; ++block) {
            Hp1Block &value = source[column * kBlocksPerRow + block];
            value.m = static_cast<int16_t>(1 + column * kBlocksPerRow + block);
            value.channel_scale = 0.125f * static_cast<float>(column + 1);
            for (size_t k = 0; k < kBlockSize; ++k)
                set_code(value, k, source_code(column, block, k));
        }
    }

#if GGML_GEMMINI_WEIGHT_BITS == 4
    args.weight_format = ggml_gemmini_args_t::im2p_weight_format_t::q4_hp1;
    args.q4_hp1_blocks = source.data();
#elif GGML_GEMMINI_WEIGHT_BITS == 8
    args.weight_format = ggml_gemmini_args_t::im2p_weight_format_t::q8_hp1;
    args.q8_hp1_blocks = source.data();
    args.q8_hp1_block_count = source.size();
    args.q8_hp1_blocks_per_row = kBlocksPerRow;
#else
    args.weight_format = ggml_gemmini_args_t::im2p_weight_format_t::q16_hp1;
    args.q16_hp1_blocks = source.data();
#endif
    args.native_block_count = source.size();
    args.native_weight_bytes = source.size() * sizeof(source.front());

    const auto radix = balanced_radix_contract(GGML_GEMMINI_ACTIVATION_BITS).radix;
    RmdStripeBuilder builder;
    builder.reset(23, 17, kRows, kLogicalK, kColumns,
                  GGML_GEMMINI_ACTIVATION_BITS);
    for (size_t k = 0; k < 11; ++k)
        if (!builder.add_residual(0, k, 1)) return 1;
    if (!builder.add_residual(1, 11, static_cast<int32_t>(radix))) return 1;
    for (size_t k = 0; k < 9; ++k)
        if (!builder.add_residual(1, 3 * kBlockSize + k, -1)) return 1;
    if (!builder.add_residual(2, 3 * kBlockSize + 9,
                              static_cast<int32_t>(radix))) return 1;
    const StripePacketHandle packet = builder.finish();

    RunAwareRequest request;
    const RmdStatus status = packet ? build_run_aware_request(args, *packet, request)
                                    : RmdStatus::invalid_packet;
    bool ok = check(status == RmdStatus::success, "request builds") &&
              check(request.runs.size() == 2, "two ordered runs") &&
              check(request.runs[0].original_block_id == 0 &&
                    request.runs[0].compact_k_begin == 0 &&
                    request.runs[0].compact_k_count == 12 &&
                    request.runs[1].original_block_id == 3 &&
                    request.runs[1].compact_k_begin == 12 &&
                    request.runs[1].compact_k_count == 10,
                    "original block IDs and compact K counts survive") &&
              check(request.original_k == kLogicalK && request.m == 4 &&
                    request.n == kColumns && request.k == 22,
                    "final request shape and original K") &&
              check(request.stripe_id == 23 && request.source_row_begin == 17 &&
                    request.source_row_count == kRows,
                    "stripe provenance survives") &&
              check(request.rows == std::vector<RunAwareRow>({{0, 0}, {0, 1},
                                                               {1, 1}, {1, 2}}),
                    "global rows are lane-major and source-row-minor");

    const auto at = [&](size_t row, size_t k) { return request.activations[row * request.k + k]; };
    ok = check(at(0, 0) == 1 && at(0, 10) == 1 && at(0, 11) == 0 &&
               at(0, 12) == 0,
               "block 0 lane 0 remains on its global row") && ok;
    ok = check(at(1, 0) == 0 && at(1, 12) == -1 && at(1, 20) == -1 &&
               at(1, 21) == 0,
               "block 3 lane 0 fills the shared global row") && ok;
    ok = check(at(2, 11) == 1 && at(2, 12) == 0 && at(3, 21) == 1,
               "missing run/lane cells are zero filled") && ok;

    for (size_t compact_k = 0; compact_k < request.k; ++compact_k) {
        const size_t block = compact_k < 12 ? 0 : 3;
        const size_t local_k = compact_k < 12 ? compact_k : compact_k - 12;
        for (size_t column = 0; column < kColumns; ++column)
            ok = check(request.weights[compact_k * kColumns + column] ==
                           source_code(column, block, local_k),
                       "weight gather uses original source coordinates") && ok;
    }
    ok = check(request.carriers == std::vector<uint32_t>({1, 5, 4, 8}),
               "run-major carriers use original block IDs") && ok;
    ok = check(request.tile_i > 0 && request.tile_j > 0 && request.tile_k > 0,
               "final geometry was selected") && ok;

    RmdStripeBuilder long_builder;
    long_builder.reset(24, 0, kRows, kLogicalK, kColumns,
                       GGML_GEMMINI_ACTIVATION_BITS);
    for (size_t k = 0; k < 17; ++k)
        if (!long_builder.add_residual(0, k, 1)) return 1;
    for (size_t k = 0; k < 20; ++k)
        if (!long_builder.add_residual(1, 3 * kBlockSize + k, -1)) return 1;
    const StripePacketHandle long_packet = long_builder.finish();
    RunAwareRequest long_request;
    ok = check(long_packet &&
                   build_run_aware_request(args, *long_packet, long_request) ==
                       RmdStatus::success &&
                   long_request.runs.size() == 2 && long_request.k == 37 &&
                   long_request.runs[0].compact_k_count == 17 &&
                   long_request.runs[1].compact_k_begin == 17 &&
                   long_request.runs[1].compact_k_count == 20 &&
                   long_request.activations[17] == 0 &&
                   long_request.activations[long_request.k + 17] == -1,
               "K>32 crosses original block boundary without a logical gap") && ok;

    RunAwareRequest sentinel = request;
    StripePacket malformed = *packet;
    malformed.blocks[1].block_id = 1;
    ok = check(build_run_aware_request(args, malformed, request) ==
                   RmdStatus::invalid_packet && request == sentinel,
               "malformed packet rejection is transactional") && ok;

    RunAwareRequest empty = sentinel;
    ok = check(build_run_aware_request(args, StripePacketHandle{}, empty) ==
                   RmdStatus::success && empty.empty(),
               "empty packet is successful zero work") && ok;

    if (ok) std::puts("RMD_RUN_AWARE_PASS blocks=0,3 compact_k=12,10 and 17,20 rows=4 columns=2");
    return ok ? 0 : 1;
}
