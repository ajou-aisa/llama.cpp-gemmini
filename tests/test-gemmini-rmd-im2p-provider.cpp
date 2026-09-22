#include "../ggml/src/ggml-gemmini/ggml-gemmini-im2p.hpp"
#include "../ggml/src/ggml-gemmini/residual/rmd/rmd-builder.hpp"
#include "../ggml/src/ggml-gemmini/residual/rmd/rmd-compose.hpp"
#include "../ggml/src/ggml-gemmini/residual/rmd/rmd-executor.hpp"
#include "../ggml/src/ggml-gemmini/residual/rmd/rmd-im2p-executor.hpp"
#include "../ggml/src/ggml-gemmini/residual/rmd/rmd-run-aware.hpp"
#include "../ggml/src/ggml-gemmini/ggml-gemmini-args.h"
#include "../ggml/src/ggml-gemmini/quants/act/exsia/exsia.hpp"
#include "../ggml/src/ggml-gemmini/quants/act/block/types.hpp"
#include "../ggml/src/ggml-gemmini/quants/common/weight_reader.hpp"

#include <im2p_sim.h>
#include <im2p_cpu_functional.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <filesystem>
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

bool hp1_backend() {
    return std::strcmp(im2p_sim_implementation(), "gemmini-hp1-integrated-v1") == 0;
}

bool run_aware_backend() {
    return hp1_backend() ||
        std::strcmp(im2p_sim_implementation(), "CPU_FUNCTIONAL") == 0;
}

struct ParityObservation {
    im2p_matmul_desc_t descriptor{};
    im2p_production_geometry_v1_t geometry{};
    im2p_compact_runs_t run_view{};
    std::vector<im2p_compact_run_t> runs;
    std::vector<int8_t> activations;
    std::vector<int8_t> weights;
    std::vector<uint32_t> carriers;
    std::vector<int64_t> output;
    size_t calls = 0;
    bool failed = false;
};

void capture_parity(ParityObservation &observation, const im2p_matmul_desc_t &descriptor,
                    const im2p_production_geometry_v1_t &geometry,
                    const im2p_compact_runs_t *run_view) {
    if (!run_view || !descriptor.activations || !descriptor.provider.read_weight_i8 ||
        !descriptor.provider.read_scale || descriptor.activation_storage_bytes != 1 ||
        descriptor.weight_storage_bytes != 1) {
        observation.failed = true;
        return;
    }
    observation.descriptor = descriptor;
    observation.geometry = geometry;
    observation.run_view = *run_view;
    observation.runs.assign(run_view->runs, run_view->runs + run_view->run_count);
    observation.activations.resize(descriptor.m * descriptor.k);
    observation.weights.resize(descriptor.k * descriptor.n);
    observation.carriers.resize(run_view->run_count * descriptor.n);
    const auto *activation_bytes = static_cast<const int8_t *>(descriptor.activations);
    for (size_t row = 0; row < descriptor.m; ++row)
        std::copy_n(activation_bytes + row * descriptor.activation_row_stride_bytes,
                    descriptor.k, observation.activations.data() + row * descriptor.k);
    for (size_t k = 0; k < descriptor.k; ++k)
        if (descriptor.provider.read_weight_i8(descriptor.provider.context, k, 0,
                                               descriptor.n, observation.weights.data() + k * descriptor.n) != IM2P_OK)
            observation.failed = true;
    for (size_t run = 0; run < run_view->run_count; ++run)
        if (descriptor.provider.read_scale(descriptor.provider.context, run, 0,
                                           descriptor.n, observation.carriers.data() + run * descriptor.n) != IM2P_OK)
            observation.failed = true;
    ++observation.calls;
}

#if defined(IM2P_CPU_FUNCTIONAL_TEST_HOOKS)
void observe_parity(void *context, const im2p_matmul_desc_t &descriptor,
                    const im2p_production_geometry_v1_t &geometry,
                    const im2p_compact_runs_t *runs) noexcept {
    auto &observation = *static_cast<ParityObservation *>(context);
    try {
        if (runs) capture_parity(observation, descriptor, geometry, runs);
        else observation.failed = true;
    } catch (...) { observation.failed = true; }
}
#endif

struct ParityExecutorContext {
    im2p_sim_t *sim;
    ParityObservation *observation;
    bool fail;
};

struct ParityProviderContext {
    im2p_provider_t original;
    ParityObservation *observation;
};

int parity_read_weight(void *opaque, size_t row, size_t column, size_t count, int8_t *out) {
    const auto &provider = *static_cast<ParityProviderContext *>(opaque);
    return provider.original.read_weight_i8(provider.original.context, row, column, count, out);
}

int parity_read_scale(void *opaque, size_t row, size_t column, size_t count, uint32_t *out) {
    const auto &provider = *static_cast<ParityProviderContext *>(opaque);
    return provider.original.read_scale(provider.original.context, row, column, count, out);
}

int parity_write_output(void *opaque, size_t block, size_t row, size_t column,
                        size_t count, const int64_t *values, uint32_t domain) {
    auto &provider = *static_cast<ParityProviderContext *>(opaque);
    try {
        if (block != 0 || row != 0 || column != 0 ||
            count != provider.observation->descriptor.m * provider.observation->descriptor.n ||
            domain != IM2P_OUTPUT_SCU_FINAL || !provider.observation->output.empty()) {
            provider.observation->failed = true;
            return IM2P_ERROR;
        }
        provider.observation->output.assign(values, values + count);
    } catch (...) {
        provider.observation->failed = true;
        return IM2P_ERROR;
    }
    return provider.original.write_output(provider.original.context, block, row,
                                          column, count, values, domain);
}

int execute_parity_runs(void *opaque, const im2p_matmul_desc_t *descriptor,
                        const im2p_production_geometry_v1_t *geometry,
                        const im2p_compact_runs_t *runs,
                        im2p_work_stats_extended_t *stats) {
    auto &context = *static_cast<ParityExecutorContext *>(opaque);
    try { capture_parity(*context.observation, *descriptor, *geometry, runs); }
    catch (...) { context.observation->failed = true; }
    if (context.observation->failed) return IM2P_ERROR;
    if (context.fail) return IM2P_ERROR;
    ParityProviderContext provider{descriptor->provider, context.observation};
    im2p_matmul_desc_t wrapped = *descriptor;
    wrapped.provider = {&provider, parity_read_weight, nullptr,
                        parity_read_scale, parity_write_output};
    return im2p_execute_matmul_planned_runs(context.sim, &wrapped, geometry, runs, stats);
}

template <typename T>
bool write_fixture_values(FILE *file, const char *name, const std::vector<T> &values) {
    if (std::fprintf(file, "%s %zu\n", name, values.size()) < 0) return false;
    for (size_t i = 0; i < values.size(); ++i)
        if (std::fprintf(file, "%s%lld", i ? " " : "",
                         static_cast<long long>(values[i])) < 0) return false;
    return std::fputc('\n', file) != EOF;
}

std::vector<bool> provider_routes() {
    return hp1_backend() ? std::vector<bool>{true} : std::vector<bool>{false, true};
}

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

    explicit Fixture(bool use_hp1 = hp1_backend(), int16_t exponent = 2,
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

    StripePacket remap_hp1_second_block(uint32_t second_block) {
        const size_t blocks_per_row = static_cast<size_t>(second_block) + 1;
        h1.assign(args.J * blocks_per_row, {});
        hp1.assign(args.J * blocks_per_row, {});
        for (size_t block_index = 0; block_index < hp1.size(); ++block_index) {
            for (size_t k = 0; k < kBlockSize; ++k) {
                const int32_t code =
                    static_cast<int32_t>((block_index * 17 + k * 5) % 7) - 3;
                set_code(block_index, k, code);
            }
            h1[block_index].c_b = 3;
            h1[block_index].R = static_cast<uint16_t>(block_index + 1);
            h1[block_index].s_rf = 0.25f;
            hp1[block_index].m = static_cast<int16_t>(block_index + 1);
            hp1[block_index].channel_scale = 0.25f;
        }
        args.K = blocks_per_row * kBlockSize;
        args.native_block_count = hp1.size();
        args.native_blocks_per_row = blocks_per_row;
#if GGML_GEMMINI_WEIGHT_BITS == 4
        args.weight_format = ggml_gemmini_args_t::im2p_weight_format_t::q4_hp1;
        args.q4_h1_blocks = nullptr;
        args.q4_hp1_blocks = hp1.data();
#elif GGML_GEMMINI_WEIGHT_BITS == 8
        args.weight_format = ggml_gemmini_args_t::im2p_weight_format_t::q8_hp1;
        args.q8_h1_blocks = nullptr;
        args.q8_h1_block_count = 0;
        args.q8_h1_rows = 0;
        args.blocks_per_row = 0;
        args.q8_hp1_blocks = hp1.data();
        args.q8_hp1_block_count = hp1.size();
        args.q8_hp1_blocks_per_row = blocks_per_row;
#else
        args.weight_format = ggml_gemmini_args_t::im2p_weight_format_t::q16_hp1;
        args.q16_h1_blocks = nullptr;
        args.q16_hp1_blocks = hp1.data();
#endif
        args.native_weight_bytes = hp1.size() * sizeof(hp1.front());
        args.A.allocate(args.I, args.K, GGML_GEMMINI_ACTIVATION_BITS);
        StripePacket mapped = *packet;
        mapped.logical_k = args.K;
        mapped.blocks[1].block_id = second_block;
        mapped.blocks[1].global_k_begin = second_block * kBlockSize;
        return mapped;
    }
};

struct OriginalExpectedWork {
    uint8_t operand_bits = 0;
    size_t n = 0;
    size_t k = 0;
    size_t original_k = 0;
    std::vector<im2p_compact_run_t> runs;
    std::vector<std::vector<uint16_t>> local_k;
    std::vector<RunAwareRow> rows;
    std::vector<int8_t> activations;
    std::vector<int8_t> weights;
    std::vector<uint32_t> carriers;
    std::vector<int64_t> output;
};

bool original_packet_digit(const StripePacket &packet, const BlockDescriptor &block,
                           uint8_t lane_id, size_t source_row, uint16_t local_k,
                           int8_t &digit) {
    if (source_row >= packet.row_count || local_k >= 32) return false;
    digit = 0;
    bool found = false;
    for (size_t position = 0; position < block.active_lane_count; ++position) {
        if (block.lane_ids[position] != lane_id) continue;
        for (const auto &group : block.groups) {
            for (size_t group_lane = 0; group_lane < group.lane_positions.size(); ++group_lane) {
                if (group.lane_positions[group_lane] != position) continue;
                if (found) return false;
                found = true;
                const uint32_t bit = uint32_t{1} << local_k;
                if (!(group.k_mask & bit)) continue;
                const size_t rank = __builtin_popcount(group.k_mask & (bit - 1));
                const size_t offset =
                    (group_lane * packet.row_count + source_row) * group.padded_k_count + rank;
                if (packet.digit_storage == DigitStorage::packed_signed_int4) {
                    const size_t byte = group.activation_byte_offset + offset / 2;
                    if (byte >= packet.stacked_activation.packed_int4.size()) return false;
                    const uint8_t nibble =
                        (packet.stacked_activation.packed_int4[byte] >> (4 * (offset % 2))) & 15;
                    digit = static_cast<int8_t>(nibble < 8 ? nibble : int(nibble) - 16);
                } else if (packet.digit_storage == DigitStorage::signed_int8) {
                    const size_t index = group.activation_offset + offset;
                    if (index >= packet.stacked_activation.signed_int8.size()) return false;
                    digit = packet.stacked_activation.signed_int8[index];
                } else {
                    return false;
                }
            }
        }
        if (!found) return false;
    }
    return true;
}

bool original_expected_work(const Fixture &fixture, const StripePacket &packet,
                            OriginalExpectedWork &expected) {
    expected = {};
    expected.operand_bits = packet.digit_bits;
    expected.n = packet.logical_j;
    expected.original_k = packet.logical_k;
    for (const auto &block : packet.blocks) {
        uint32_t mask = 0;
        for (const auto &group : block.groups) mask |= group.k_mask;
        if (__builtin_popcount(mask) != block.compact_k_count ||
            block.global_k_begin != block.block_id * 32) return false;
        std::vector<uint16_t> local;
        for (uint16_t k = 0; k < 32; ++k)
            if (mask & (uint32_t{1} << k)) {
                if (size_t(block.global_k_begin) + k >= packet.logical_k) return false;
                local.push_back(k);
            }
        if (size_t(block.k_index_offset) + local.size() > packet.k_indices.size())
            return false;
        for (size_t k = 0; k < local.size(); ++k)
            if (packet.k_indices[block.k_index_offset + k] != local[k]) return false;
        expected.runs.push_back({block.block_id, mask,
                                 static_cast<uint32_t>(expected.k),
                                 static_cast<uint32_t>(local.size())});
        expected.k += local.size();
        expected.local_k.push_back(std::move(local));
    }
    for (uint8_t lane = 0; lane < packet.lane_capacity; ++lane) {
        for (size_t row = 0; row < packet.row_count; ++row) {
            bool active = false;
            for (size_t run = 0; run < packet.blocks.size(); ++run)
                for (uint16_t local : expected.local_k[run]) {
                    int8_t digit = 0;
                    if (!original_packet_digit(packet, packet.blocks[run], lane, row,
                                               local, digit)) return false;
                    active |= digit != 0;
                }
            if (active) expected.rows.push_back({lane, static_cast<uint32_t>(row)});
        }
    }
    expected.activations.resize(expected.rows.size() * expected.k);
    expected.weights.resize(expected.k * expected.n);
    expected.carriers.resize(expected.runs.size() * expected.n);
    for (size_t row = 0; row < expected.rows.size(); ++row)
        for (size_t run = 0; run < expected.runs.size(); ++run)
            for (size_t local = 0; local < expected.local_k[run].size(); ++local) {
                int8_t digit = 0;
                if (!original_packet_digit(packet, packet.blocks[run],
                                           expected.rows[row].original_lane_id,
                                           expected.rows[row].source_row,
                                           expected.local_k[run][local], digit)) return false;
                expected.activations[row * expected.k +
                                     expected.runs[run].compact_k_begin + local] = digit;
            }
    for (size_t run = 0; run < expected.runs.size(); ++run) {
        const size_t original_block = expected.runs[run].original_block_id;
        for (size_t column = 0; column < expected.n; ++column) {
            const size_t native = column * fixture.args.native_blocks_per_row + original_block;
            if (native >= fixture.hp1.size()) return false;
            const auto &block = fixture.hp1[native];
            if (block.m < 0 && block.m != INT16_MIN) return false;
            expected.carriers[run * expected.n + column] =
                block.m == INT16_MIN ? 0x80000000u : static_cast<uint32_t>(block.m);
            for (size_t local = 0; local < expected.local_k[run].size(); ++local) {
                const size_t original_local_k = expected.local_k[run][local];
#if GGML_GEMMINI_WEIGHT_BITS == 4
                const uint8_t packed = block.qs[original_local_k % (kBlockSize / 2)];
                const uint8_t nibble = (packed >> (original_local_k < kBlockSize / 2 ? 0 : 4)) & 15;
                const int8_t code = static_cast<int8_t>(int(nibble) - 8);
#else
                const int8_t code = static_cast<int8_t>(block.qs[original_local_k]);
#endif
                expected.weights[(expected.runs[run].compact_k_begin + local) *
                                 expected.n + column] = code;
            }
        }
    }
    const auto sat32 = [](int64_t value) {
        return static_cast<int32_t>(std::clamp<int64_t>(value, INT32_MIN, INT32_MAX));
    };
    expected.output.resize(expected.rows.size() * expected.n);
    for (size_t row = 0; row < expected.rows.size(); ++row)
        for (size_t column = 0; column < expected.n; ++column) {
            int32_t total = 0;
            for (size_t run = 0; run < expected.runs.size(); ++run) {
                int32_t fragment = 0;
                for (size_t local = 0; local < expected.local_k[run].size(); ++local) {
                    const size_t k = expected.runs[run].compact_k_begin + local;
                    fragment += int32_t(expected.activations[row * expected.k + k]) *
                                expected.weights[k * expected.n + column];
                    if ((local + 1) % DIM == 0 || local + 1 == expected.local_k[run].size()) {
                        const uint32_t carrier = expected.carriers[run * expected.n + column];
                        const int32_t scaled = carrier == 0x80000000u || fragment == 0 ? 0 :
                            carrier >= 32 ? (fragment < 0 ? INT32_MIN : INT32_MAX) :
                            sat32(int64_t(fragment) * (int64_t{1} << carrier));
                        total = sat32(int64_t(total) + scaled);
                        fragment = 0;
                    }
                }
            }
            expected.output[row * expected.n + column] = total;
        }
    return true;
}

bool captured_work_matches(const OriginalExpectedWork &expected,
                           const RunAwareRequest &request,
                           const ParityObservation &observation) {
    const auto &d = observation.descriptor;
    const auto &g = observation.geometry;
    if (request.m != expected.rows.size() || request.n != expected.n ||
        request.k != expected.k || request.original_k != expected.original_k ||
        request.operand_bits != expected.operand_bits ||
        request.rows.size() != expected.rows.size() ||
        request.runs.size() != expected.runs.size() ||
        d.m != expected.rows.size() || d.n != expected.n || d.k != expected.k ||
        d.activation_bits != expected.operand_bits ||
        d.weight_bits != expected.operand_bits || d.dim != DIM ||
        d.scale_total_k != expected.original_k ||
        observation.run_view.original_k != expected.original_k ||
        observation.run_view.run_count != expected.runs.size() ||
        observation.runs.size() != expected.runs.size() ||
        g.m != d.m || g.n != d.n || g.k != d.k || g.dim != DIM ||
        g.activation_bits != expected.operand_bits ||
        g.weight_bits != expected.operand_bits || g.scope != IM2P_GEOMETRY_FULL ||
        g.tile_i_count != request.tile_i || g.tile_j_count != request.tile_j ||
        g.tile_k_count != request.tile_k ||
        request.activations != expected.activations ||
        observation.activations != expected.activations ||
        request.weights.size() != expected.weights.size() ||
        observation.weights != expected.weights ||
        request.carriers != expected.carriers ||
        observation.carriers != expected.carriers ||
        observation.output != expected.output) return false;
    for (size_t row = 0; row < expected.rows.size(); ++row)
        if (request.rows[row].source_row != expected.rows[row].source_row ||
            request.rows[row].original_lane_id != expected.rows[row].original_lane_id)
            return false;
    for (size_t run = 0; run < expected.runs.size(); ++run) {
        const auto &want = expected.runs[run];
        const auto &actual = observation.runs[run];
        const auto &built = request.runs[run];
        if (actual.original_block_id != want.original_block_id ||
            actual.original_k_mask != want.original_k_mask ||
            actual.compact_k_begin != want.compact_k_begin ||
            actual.compact_k_count != want.compact_k_count ||
            built.original_block_id != want.original_block_id ||
            built.union_k_mask != want.original_k_mask ||
            built.compact_k_begin != want.compact_k_begin ||
            built.compact_k_count != want.compact_k_count ||
            built.original_local_k != expected.local_k[run]) return false;
    }
    for (size_t k = 0; k < expected.weights.size(); ++k)
        if (request.weights[k] != expected.weights[k]) return false;
    return true;
}

bool recomposed_output_matches(const StripePacket &packet,
                               const OriginalExpectedWork &expected,
                               const std::vector<OutputValue> &actual) {
    if (actual.size() != packet.row_count * expected.n) return false;
    std::vector<__int128> recomposed(actual.size());
    for (size_t row = 0; row < expected.rows.size(); ++row) {
        const auto &map = expected.rows[row];
        if (map.source_row >= packet.row_count || map.original_lane_id >= packet.lane_capacity)
            return false;
        __int128 place = 1;
        for (uint8_t lane = 0; lane < map.original_lane_id; ++lane)
            place *= int64_t{1} << packet.digit_bits;
        for (size_t column = 0; column < expected.n; ++column)
            recomposed[map.source_row * expected.n + column] +=
                static_cast<__int128>(expected.output[row * expected.n + column]) * place;
    }
    for (size_t i = 0; i < actual.size(); ++i)
        if (recomposed[i] != actual[i]) return false;
    return true;
}

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
    if (hp1_backend()) {
        Fixture h1(false);
        Sim sim(im2p_sim_create());
        CompressedOutput output{CompressedOutput::Domain::block_scaled_int64, 91, {7, -11}};
        RmdExecutionMetrics metrics{};
        metrics.packet_call_count = 73;
        metrics.im2p_dot_calls = 79;
        if (!check(execute_rmd_stripe_im2p(sim.get(), h1.args, *h1.packet, output, &metrics) ==
                       RmdStatus::unsupported_route && unchanged(output, metrics),
                   "HP1 backend rejects H1 without changing output or metrics")) return false;
    }
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
        check(streaming_metrics.im2p_dot_calls ==
                  (hp1_backend() ? 1 : actual_metrics.im2p_dot_calls) &&
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
        check(hp1_backend()
                  ? beyond_i32 == actual.values.end() &&
                        std::find(actual.values.begin(), actual.values.end(), INT32_MAX) != actual.values.end()
                  : beyond_i32 != actual.values.end(),
              "provider preserves its declared Sat32 or legacy INT64 domain") &&
        check(metrics.im2p_dot_calls > 0 && metrics.ws_call_count == 0,
              "HP1 exponent 62 uses IM2P only");
    if (ok) std::printf("IM2P_PROVIDER hp1-exp-62 status=success dot_calls=%zu sat32=%u ws_calls=0\n",
                        metrics.im2p_dot_calls, unsigned(hp1_backend()));
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
    for (const bool use_hp1 : provider_routes()) {
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
            const size_t expected_validations =
                use_hp1 && run_aware_backend()
                    ? stripes + (reuse ? 1 : stripes)
                    : (reuse ? 1 : stripes);
            if (!check(validations == expected_validations && output == expected_output,
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
            invalid.hp1[1].m = hp1_backend() ? -1 : 63;
            detail::RmdWeightPreparation overflow_weights;
            if (!check(detail::execute_rmd_stripe_im2p_with_weights(
                           sim.get(), invalid.args, *packets.front(), rejected, overflow_weights,
                           &metrics) == (hp1_backend() ? RmdStatus::unsupported_route : RmdStatus::overflow) &&
                           unchanged(rejected, metrics),
                       "shared preparation invalid block scale preserves correction and metrics")) return false;
        }
    }
    return true;
}

bool run_packet_merge_contract() {
    namespace adapter = ggml::gemmini::im2p_adapter;
    namespace exsia = ggml::gemmini::quants::act::exsia;
    for (const bool pipeline : {false, true}) {
        for (const bool use_hp1 : provider_routes()) {
            for (size_t probe = 0; probe < 6; ++probe) {
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
                if (probe == 1 || probe == 5)
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
                if (probe >= 2 && probe <= 4) {
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
                    if (!started.result.ok()) completion.result = started.result;
                    else {
                        if (!check(started.pipeline->install_sink().ok(),
                                   "packet merge PIPELINE installs sink")) return false;
                        completion = started.pipeline->finish(publish());
                    }
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
                               "consistent shared block scales allow packet merge")) {
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
                                      (completion.result.error == Error::invalid_contract &&
                                                        counters.residual_executions == 0 &&
                                                        counters.provider_dot_attempts == 0),
                                  "shared scale or packet/event range mismatch preserves output")) {
                    std::fprintf(stderr, "mode=%s route=%s probe=%zu status=%s\n",
                                 pipeline ? "PIPELINE" : "FULL", use_hp1 ? "HP1" : "H1",
                                 probe, completion.result.message);
                    return false;
                }
            }
        }
    }
    std::puts("IM2P_PROVIDER packet-merge-contract modes=FULL,PIPELINE shared-scale=consistent mismatch=rejected rows=checked output=transactional");
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
    for (const bool use_hp1 : provider_routes()) {
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
    std::puts("IM2P_PROVIDER int32-residuals status=success carry_lane=retained");
    return true;
}

bool run_group_rows() {
    struct Event { size_t row; size_t k; int32_t value; };
    const size_t columns = DIM + 1;
    Sim sim(im2p_sim_create());
    if (!check(sim != nullptr, "row-boundary provider simulator exists")) return false;
    for (const size_t rows : {size_t{1}, size_t{DIM - 1}, size_t{DIM}, size_t{DIM + 1}}) {
        for (const bool use_hp1 : provider_routes()) {
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
                        const size_t group_rows = align_up(group.lane_positions.size() * rows, kArrayDim);
                        expected_values += group_rows * group.padded_k_count;
                        const size_t kj_tiles = (hp1_backend() ? 1 : group.padded_k_count / kArrayDim) *
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
                    !check(use_hp1 && run_aware_backend()
                               ? metrics.stacked_i_tile_count > 0 &&
                                     metrics.im2p_dot_calls == 1
                               : metrics.stacked_i_tile_count == expected_tiles,
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
    for (const bool use_hp1 : provider_routes()) {
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
    if (hp1_backend()) {
        CompressedOutput expected;
        return check(status == RmdStatus::success &&
                         execute_rmd_stripe_reference(fixture.args, *fixture.packet, expected) == RmdStatus::success &&
                         output.values == expected.values &&
                         std::all_of(output.values.begin(), output.values.end(), [](int64_t value) {
                             return value >= INT32_MIN && value <= INT32_MAX;
                         }),
                     "HP1 exponent 63 retains its carrier and produces Sat32 output");
    }
    const bool ok = check(status == RmdStatus::overflow, "HP1 exponent 63 is typed overflow") &&
        check(ggml::gemmini::quants::wreader::test_weight_reader_code_address_resolutions() == 0,
              "invalid final weight block scale rejects before any code gather") &&
        check(unchanged(output, metrics), "HP1 overflow is transactional");
    if (ok) std::printf("IM2P_PROVIDER hp1-exp-63 status=overflow dot_calls=0 output_sentinel=unchanged metrics_sentinel=unchanged ws_calls=%zu\n",
                        metrics.ws_call_count);
    return ok;
}

bool run_cross_block_one_work() {
    if (!run_aware_backend()) return true;
    Fixture fixture(true);
    const StripePacket packet = fixture.remap_hp1_second_block(3);
    Sim sim(im2p_sim_create());
    Correction expected;
    Correction correction = PreScaledFloat64Correction{{7.25, -11.5}};
    RmdExecutionMetrics metrics{};
    reset_im2p_provider_dot_attempts_for_test();
    const RmdStatus oracle = execute_rmd_stripe_reference(
        fixture.args, packet, expected);
    const RmdStatus status = sim ? execute_rmd_stripe_im2p(
        sim.get(), fixture.args, packet, correction, &metrics) :
        RmdStatus::execution_failed;
    const auto *expected_values = std::get_if<BlockScaledInt64Correction>(&expected);
    const auto *actual_values = std::get_if<BlockScaledInt64Correction>(&correction);
    const bool ok = check(packet.blocks.size() == 2 &&
                              packet.blocks[0].block_id == 0 &&
                              packet.blocks[1].block_id == 3,
                          "fixture retains gapped original K32 blocks") &&
        check(oracle == RmdStatus::success && expected_values && actual_values &&
                  actual_values->values == expected_values->values,
              "run ordinals preserve original block carriers") &&
        check(status == RmdStatus::success, "cross-block residual execution succeeds") &&
        check(metrics.im2p_dot_calls == 1 && im2p_provider_dot_attempts_for_test() == 1,
              "one residual invocation selects one logical provider work");
    if (ok) std::puts("IM2P_PROVIDER cross-block-one-work logical_calls=1 original_blocks=0,3 PASS");
    return ok;
}

bool run_dispatch_parity(std::string_view case_name = "gap-0-3") {
    if (!run_aware_backend()) return true;
    const size_t rows = case_name == "one-run" || case_name == "radix-lane" ? 1 : Fixture::rows;
    const size_t columns = case_name == "tail-mn" ? DIM + 1 : Fixture::columns;
    Fixture fixture(true, 2, rows, columns);
    StripePacket packet = fixture.remap_hp1_second_block(
        case_name == "boundary-1-31" ? 31 : 3);
    if (case_name == "gap-0-3-k22" || case_name == "gap-0-3-k37" ||
        case_name == "one-run" || case_name == "radix-lane") {
        RmdStripeBuilder builder;
        builder.reset(19, 7, rows, fixture.args.K, columns,
                      GGML_GEMMINI_ACTIVATION_BITS);
        const size_t first_count = case_name == "gap-0-3-k22" ? 18 :
                                   case_name == "gap-0-3-k37" ? 32 : 1;
        for (size_t k = 0; k < first_count; ++k)
            if (!builder.add_residual(k % rows, k,
                                      case_name == "radix-lane"
                                          ? (1 << GGML_GEMMINI_ACTIVATION_BITS) + 2 : 1)) return false;
        if (case_name == "radix-lane" && !builder.add_residual(0, 1, 1)) return false;
        if (case_name != "one-run" && case_name != "radix-lane") {
            for (size_t k : {size_t{0}, size_t{1}, kBlockSize - 2, kBlockSize - 1})
                if (!builder.add_residual((k + 3) % rows, 3 * kBlockSize + k, -1)) return false;
            if (case_name == "gap-0-3-k37" &&
                !builder.add_residual(5 % rows, 3 * kBlockSize + 2, -1)) return false;
        }
        const auto built = builder.finish();
        if (!check(bool(built), "production fixture packet builds")) return false;
        packet = *built;
    }
    if (case_name == "boundary-1-31") {
        packet.blocks[0].block_id = 1;
        packet.blocks[0].global_k_begin = kBlockSize;
    }
    Sim sim(im2p_sim_create());
    if (!check(bool(sim), "parity simulator exists")) return false;
    ParityObservation observation;
    ParityExecutorContext context{sim.get(), &observation, false};
    Im2pFullExecutor executor{};
    executor.context = &context;
    executor.execute_planned_runs = execute_parity_runs;
    Correction correction;
    RmdExecutionMetrics metrics{};
#if defined(IM2P_CPU_FUNCTIONAL_TEST_HOOKS)
    if (!hp1_backend()) im2p::cpu_functional::set_dispatch_observer(observe_parity, &observation);
#endif
    const RmdStatus status = execute_rmd_stripe_im2p(
        sim.get(), fixture.args, packet, correction, &metrics,
        hp1_backend() ? &executor : nullptr);
#if defined(IM2P_CPU_FUNCTIONAL_TEST_HOOKS)
    if (!hp1_backend()) im2p::cpu_functional::set_dispatch_observer(nullptr, nullptr);
#endif
    const auto *values = std::get_if<BlockScaledInt64Correction>(&correction);
    Correction expected;
    const RmdStatus reference_status = execute_rmd_stripe_reference(fixture.args, packet, expected);
    const auto *reference_values = std::get_if<BlockScaledInt64Correction>(&expected);
    const bool ok = check(status == RmdStatus::success &&
                              reference_status == RmdStatus::success && values && reference_values &&
                              (case_name != "gap-0-3" ||
                               values->values == reference_values->values) &&
                              metrics.im2p_dot_calls == 1,
                          "parity dispatch matches reference") &&
        check(!observation.failed && observation.calls == 1 &&
                  observation.run_view.version == IM2P_COMPACT_RUNS_VERSION &&
                  observation.runs.size() == packet.blocks.size(),
              "one production run-aware dispatch retains original block owners");
    if (!ok) return false;
    for (size_t run = 0; run < observation.runs.size(); ++run)
        if (!check(observation.runs[run].original_block_id == packet.blocks[run].block_id,
                   "captured run owns the original packet block")) return false;

    const char *log_dir = std::getenv("GEMMINI_LOG_DIR");
    if (!check(log_dir && *log_dir, "GEMMINI_LOG_DIR is set")) return false;
    const std::filesystem::path base(log_dir);
    std::error_code error;
    std::filesystem::create_directories(base, error);
    if (!check(!error, "parity output directory exists")) return false;
    const auto &d = observation.descriptor;
    const auto &g = observation.geometry;
    const auto &v = observation.run_view;
    const std::string suffix = case_name == "gap-0-3" ? "" : "-" + std::string(case_name);
    const std::string scalar_path = (base / ("rmd-im2p-dispatch" + suffix + ".txt")).string();
    FILE *scalar = std::fopen(scalar_path.c_str(), "wb");
    if (!check(scalar != nullptr, "parity scalar file opens")) return false;
    std::fprintf(scalar, "descriptor %u %u %u %u %u %u %zu %zu %zu %zu %zu %zu %zu %zu %zu %zu %zu %zu %zu %zu %u %u %llu\n",
                 d.abi_version, d.activation_bits, d.activation_storage_bytes,
                 d.weight_bits, d.weight_storage_bytes, d.dim, d.m, d.n, d.k,
                 d.activation_row_stride_bytes, d.weight_row_stride_bytes,
                 d.output_row_stride, d.tile_i_rows, d.tile_j_columns,
                 d.block_size, d.scale_total_k, d.scale_row_stride,
                 d.scale_column_offset, d.scale_valid_columns, d.scale_values_len,
                 unsigned(d.vector_op), unsigned(d.output_domain),
                 static_cast<unsigned long long>(d.work_context));
    std::fprintf(scalar, "geometry %u %u %u %u %u %u %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu\n",
                 g.version, g.struct_size, g.activation_bits, g.weight_bits, g.dim, g.scope,
                 static_cast<unsigned long long>(g.m), static_cast<unsigned long long>(g.n),
                 static_cast<unsigned long long>(g.k), static_cast<unsigned long long>(g.tile_i_count),
                 static_cast<unsigned long long>(g.tile_j_count), static_cast<unsigned long long>(g.tile_k_count),
                 static_cast<unsigned long long>(g.stripe_rows), static_cast<unsigned long long>(g.row_begin),
                 static_cast<unsigned long long>(g.row_count), static_cast<unsigned long long>(g.stripe_id));
    std::fprintf(scalar, "runs %u %u %u %zu\n", v.version, v.struct_size, v.original_k, v.run_count);
    for (const auto &run : observation.runs)
        std::fprintf(scalar, "run %u %u %u %u\n", run.original_block_id,
                     run.original_k_mask, run.compact_k_begin, run.compact_k_count);
    const bool scalar_ok = std::fclose(scalar) == 0;
    const std::string binary_path = (base / ("rmd-im2p-output" + suffix + ".bin")).string();
    FILE *binary = std::fopen(binary_path.c_str(), "wb");
    if (!check(binary != nullptr, "parity binary file opens")) return false;
    const bool write_ok = std::fwrite(values->values.data(), sizeof(OutputValue),
                                     values->values.size(), binary) == values->values.size();
    const bool binary_ok = std::fclose(binary) == 0 && write_ok;
    if (!check(scalar_ok && binary_ok, "parity files are complete")) return false;
    if (!hp1_backend()) return true;

    OriginalExpectedWork original;
    RunAwareRequest request;
    if (!check(original_expected_work(fixture, packet, original) &&
                   build_run_aware_request(fixture.args, packet, request) == RmdStatus::success &&
                   captured_work_matches(original, request, observation),
               "captured request matches original packet and native HP1 blocks")) return false;
    if (case_name == "radix-lane") {
        RunAwareRequest swapped = request;
        std::swap(swapped.rows[0], swapped.rows[1]);
        ParityObservation wrong_k = observation;
        RunAwareRequest wrong_k_request = request;
        std::swap(wrong_k_request.runs[0].original_local_k[0],
                  wrong_k_request.runs[0].original_local_k[1]);
        for (size_t row = 0; row < original.rows.size(); ++row) {
            std::swap(wrong_k.activations[row * original.k],
                      wrong_k.activations[row * original.k + 1]);
            std::swap(wrong_k_request.activations[row * original.k],
                      wrong_k_request.activations[row * original.k + 1]);
        }
        for (size_t column = 0; column < original.n; ++column) {
            std::swap(wrong_k.weights[column], wrong_k.weights[original.n + column]);
            std::swap(wrong_k_request.weights[column],
                      wrong_k_request.weights[original.n + column]);
        }
        ParityObservation wrong_carrier = observation;
        RunAwareRequest wrong_carrier_request = request;
        std::swap(wrong_carrier.carriers[0], wrong_carrier.carriers[1]);
        std::swap(wrong_carrier_request.carriers[0], wrong_carrier_request.carriers[1]);
        if (!check(original.rows.size() == 2 &&
                       original.rows[0].original_lane_id != original.rows[1].original_lane_id &&
                       original.activations[0] != original.activations[original.k] &&
                       original.output[0] != original.output[original.n] &&
                       !captured_work_matches(original, swapped, observation),
                   "oracle rejects a swapped radix row map")) return false;
        if (!check(!captured_work_matches(original, wrong_k_request, wrong_k),
                   "oracle rejects a swapped original K mapping")) return false;
        if (!check(!captured_work_matches(original, wrong_carrier_request, wrong_carrier),
                   "oracle rejects a swapped carrier mapping")) return false;
    }
    if (!check(recomposed_output_matches(packet, original, values->values),
               "raw output and original radix lanes recompose final correction")) return false;

    FILE *metadata = std::fopen(scalar_path.c_str(), "rb");
    const std::string fixture_path = (base / ("rmd-run-work-" + std::string(case_name) + ".txt")).string();
    FILE *work_file = std::fopen(fixture_path.c_str(), "wb");
    if (!check(metadata && work_file, "run-work fixture files open")) {
        if (metadata) std::fclose(metadata);
        if (work_file) std::fclose(work_file);
        return false;
    }
    bool fixture_ok = std::fputs("RMD_RUN_WORK_V1\n", work_file) >= 0;
    char buffer[4096];
    size_t count;
    while (fixture_ok && (count = std::fread(buffer, 1, sizeof(buffer), metadata)) != 0)
        fixture_ok = std::fwrite(buffer, 1, count, work_file) == count;
    fixture_ok = fixture_ok && !std::ferror(metadata);
    fixture_ok = std::fclose(metadata) == 0 && fixture_ok;
    fixture_ok = std::fprintf(work_file, "ROW_MAP %zu\n", original.rows.size()) >= 0 && fixture_ok;
    for (const auto &row : original.rows)
        fixture_ok = std::fprintf(work_file, "%u %u\n", row.source_row,
                                  unsigned(row.original_lane_id)) >= 0 && fixture_ok;
    fixture_ok = write_fixture_values(work_file, "A", observation.activations) && fixture_ok;
    fixture_ok = write_fixture_values(work_file, "B", observation.weights) && fixture_ok;
    fixture_ok = write_fixture_values(work_file, "CARRIERS", observation.carriers) && fixture_ok;
    fixture_ok = write_fixture_values(work_file, "OUTPUT", observation.output) && fixture_ok;
    fixture_ok = std::fclose(work_file) == 0 && fixture_ok;
    return check(fixture_ok, "run-work fixture is complete");
}

bool run_dispatch_parity_callback_failure() {
    if (!hp1_backend()) return true;
    Fixture fixture(true);
    const StripePacket packet = fixture.remap_hp1_second_block(3);
    Sim sim(im2p_sim_create());
    ParityObservation observation;
    ParityExecutorContext context{sim.get(), &observation, true};
    Im2pFullExecutor executor{};
    executor.context = &context;
    executor.execute_planned_runs = execute_parity_runs;
    Correction correction = PreScaledFloat64Correction{{7.25, -11.5}};
    RmdExecutionMetrics metrics{};
    metrics.packet_call_count = 73;
    metrics.im2p_dot_calls = 79;
    const RmdStatus status = sim ? execute_rmd_stripe_im2p(
        sim.get(), fixture.args, packet, correction, &metrics, &executor) :
        RmdStatus::execution_failed;
    return check(status == RmdStatus::execution_failed && observation.calls == 1 &&
                     unchanged(correction, metrics),
                 "callback failure leaves correction and metrics unchanged");
}

bool run_sat32_cross_run() {
    if (!run_aware_backend()) return true;
    const auto sat32 = [](int64_t value) {
        return static_cast<int32_t>(std::clamp<int64_t>(value, INT32_MIN, INT32_MAX));
    };
    for (const int sign : {-1, 1}) {
        Fixture fixture(true, 31, 1, 3);
        for (size_t column = 0; column < 3; ++column) {
            fixture.set_code(column * 2, 0, sign);
            fixture.set_code(column * 2 + 1, 0, sign);
        }
        RmdStripeBuilder builder;
        builder.reset(31, 7, 1, Fixture::logical_k, 3,
                      GGML_GEMMINI_ACTIVATION_BITS);
        if (!builder.add_residual(0, 0, 1) ||
            !builder.add_residual(0, kBlockSize, 1)) return false;
        const auto packet = builder.finish();
        Sim sim(im2p_sim_create());
        Correction correction;
        RmdExecutionMetrics metrics{};
        reset_im2p_provider_dot_attempts_for_test();
        const RmdStatus status = packet && sim ? execute_rmd_stripe_im2p(
            sim.get(), fixture.args, *packet, correction, &metrics) :
            RmdStatus::execution_failed;
        const int32_t fragment = sat32(int64_t(sign) * (int64_t{1} << 31));
        const int32_t expected = sat32(int64_t(fragment) + fragment);
        const auto *values = std::get_if<BlockScaledInt64Correction>(&correction);
        if (!check(packet && packet->blocks.size() == 2 &&
                       packet->blocks[0].block_id == 0 &&
                       packet->blocks[1].block_id == 1,
                   "original events make two distinct run owners") ||
            !check(expected != int64_t(fragment) + fragment,
                   "oracle distinguishes host sum from Sat32 accumulation") ||
            !check(status == RmdStatus::success && values &&
                       values->values == std::vector<OutputValue>(3, expected) &&
                       metrics.im2p_dot_calls == 1 &&
                       im2p_provider_dot_attempts_for_test() == 1,
                   "cross-run fragment saturation uses one signed32 accumulator"))
            return false;
    }
    std::puts("IM2P_PROVIDER sat32-cross-run signs=-1,+1 logical_calls=1 PASS");
    return true;
}

bool run_run_aware_fail_closed() {
    if (!run_aware_backend()) return true;
    Fixture fixture(true);
    Correction correction = PreScaledFloat64Correction{{7.25, -11.5}};
    RmdExecutionMetrics metrics{};
    metrics.packet_call_count = 73;
    metrics.im2p_dot_calls = 79;
    reset_im2p_provider_dot_attempts_for_test();
    const RmdStatus capability_status = fixture.packet ?
        execute_rmd_stripe_im2p_missing_runs_for_test(
            fixture.args, *fixture.packet, correction, &metrics) :
        RmdStatus::execution_failed;
    if (!check(capability_status == RmdStatus::unsupported_route &&
                   unchanged(correction, metrics) &&
                   im2p_provider_dot_attempts_for_test() == 0,
               "missing planned-runs capability fails closed")) return false;

    Sim sim(im2p_sim_create());
    fixture.args.optrace_context =
        std::make_shared<ggml::gemmini::optrace::Context>();
    reset_im2p_provider_dot_attempts_for_test();
    const RmdStatus trace_status = fixture.packet && sim ? execute_rmd_stripe_im2p(
        sim.get(), fixture.args, *fixture.packet, correction, &metrics) :
        RmdStatus::execution_failed;
    const bool ok = check(trace_status == RmdStatus::unsupported_route &&
                              unchanged(correction, metrics) &&
                              im2p_provider_dot_attempts_for_test() == 0,
                          "unrepresentable run-aware trace fails closed");
    if (ok) std::puts("IM2P_PROVIDER run-aware-fail-closed capability=missing trace=unsupported PASS");
    return ok;
}

bool run_activation_block_gate() {
    Fixture fixture(true);
    auto &metadata = fixture.args.act_quant.storage()
                         .emplace<ggml::gemmini::quants::act::block::Meta>();
    metadata.rows = fixture.args.activation_row_offset +
                    fixture.packet->row_begin + fixture.packet->row_count;
    metadata.cols = fixture.args.K;
    metadata.scales.assign(metadata.rows *
                               ((metadata.cols + kBlockSize - 1) / kBlockSize),
                           1.0f);
    Sim sim(im2p_sim_create());
    Correction expected;
    Correction actual;
    const RmdStatus oracle = fixture.packet ? execute_rmd_stripe_reference(
        fixture.args, *fixture.packet, expected) : RmdStatus::invalid_packet;
    reset_im2p_provider_dot_attempts_for_test();
    RmdExecutionMetrics metrics{};
    const RmdStatus status = fixture.packet && sim ? execute_rmd_stripe_im2p(
        sim.get(), fixture.args, *fixture.packet, actual, &metrics) :
        RmdStatus::execution_failed;
    const auto *expected_values =
        std::get_if<FullyScaledFloat64Correction>(&expected);
    const auto *actual_values =
        std::get_if<FullyScaledFloat64Correction>(&actual);
    if (oracle != RmdStatus::success || status != RmdStatus::success ||
        !expected_values || !actual_values ||
        actual_values->values != expected_values->values) {
        std::fprintf(stderr,
                     "activation-block detail oracle=%s status=%s expected-domain=%zu actual-domain=%zu expected-size=%zu actual-size=%zu\n",
                     rmd_status_message(oracle), rmd_status_message(status),
                     expected.index(), actual.index(),
                     expected_values ? expected_values->values.size() : 0,
                     actual_values ? actual_values->values.size() : 0);
    }
    const bool ok = check(oracle == RmdStatus::success &&
                              status == RmdStatus::success &&
                              expected_values && actual_values &&
                              actual_values->values == expected_values->values,
                          "activation block metadata keeps exact block-local correction") &&
        check(metrics.im2p_dot_calls > 1 &&
                  im2p_provider_dot_attempts_for_test() == metrics.im2p_dot_calls,
              "activation block metadata bypasses packet-wide dispatch");
    if (ok) std::puts("IM2P_PROVIDER activation-block-gate route=block-local PASS");
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
    if (selected == "all" || selected == "cross-block-one-work") ok = run_cross_block_one_work() && ok;
    if (selected == "dispatch-parity") ok = run_dispatch_parity() && ok;
    if (selected == "production-oracle-mutations") ok = run_dispatch_parity("radix-lane") && ok;
    if (selected == "production-fixtures") {
        for (std::string_view case_name : {"gap-0-3", "gap-0-3-k22", "gap-0-3-k37",
                                           "one-run", "boundary-1-31", "tail-mn",
                                           "radix-lane"})
            ok = run_dispatch_parity(case_name) && ok;
    }
    if (selected == "dispatch-parity-callback-failure") ok = run_dispatch_parity_callback_failure() && ok;
    if (selected == "all" || selected == "sat32-cross-run") ok = run_sat32_cross_run() && ok;
    if (selected == "all" || selected == "run-aware-fail-closed") ok = run_run_aware_fail_closed() && ok;
    if (selected == "all" || selected == "activation-block-gate") ok = run_activation_block_gate() && ok;
    if (selected == "all" || selected == "malformed-packet") ok = run_malformed_packet() && ok;
    if (selected == "all" || selected == "shared-preparation") ok = run_shared_preparation() && ok;
    if (selected == "all" || selected == "packet-merge-contract") ok = run_packet_merge_contract() && ok;
    if (selected == "all" || selected == "int32-residuals") ok = run_int32_residuals() && ok;
    if (selected == "all" || selected == "group-rows") ok = run_group_rows() && ok;
    if (selected == "all" || selected == "native-code-edges") ok = run_native_code_edges() && ok;
    if (selected == "all" || selected == "route-matched" || selected == "route-mismatch" || selected == "h0-compact-rejection") ok = run_route(selected == "all" ? "route-matched" : selected) && ok;

    constexpr std::array<std::string_view, 31> valid{{"all", "success", "provider-read-failure", "provider-write-failure", "provider-watchdog", "k-accumulation-overflow", "block-scale-overflow", "cancel-between-dots", "duplicate-output", "missing-output", "output-index", "stats-overflow", "hp1-exp-62", "hp1-exp-63", "cross-block-one-work", "dispatch-parity", "production-fixtures", "production-oracle-mutations", "dispatch-parity-callback-failure", "sat32-cross-run", "run-aware-fail-closed", "activation-block-gate", "malformed-packet", "shared-preparation", "packet-merge-contract", "int32-residuals", "group-rows", "native-code-edges", "route-matched", "route-mismatch", "h0-compact-rejection"}};
    const bool is_valid = std::find(valid.begin(), valid.end(), selected) != valid.end() || selected == "h0-compact-rejection";
    if (!is_valid) {
        std::fprintf(stderr, "unsupported test case: %.*s\n", static_cast<int>(selected.size()), selected.data());
        return 2;
    }
    return ok ? 0 : 1;
}
