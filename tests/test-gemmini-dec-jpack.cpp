#include "../ggml/src/ggml-gemmini/quants/dec/dec_h1_jpack.hpp"
#include "../ggml/src/ggml-gemmini/quants/dec/dec_kernel.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

namespace {

using ggml::gemmini::quants::dec::DecH1JPack;
using ggml::gemmini::quants::dec::DecH1JPackAccounting;
using ggml::gemmini::quants::dec::DecH1JPackCounters;
using ggml::gemmini::quants::dec::DecH1JPackEntry;
using ggml::gemmini::quants::dec::DecH1JPackSafety;

constexpr size_t kGroupSize = ggml::gemmini::quants::dec::kDecH1JPackGroupSize;
constexpr size_t kNr = ggml::gemmini::quants::dec::kDecH1JPackNr;
constexpr size_t kWarmups = 7;
constexpr size_t kSamples = 21;
constexpr size_t kRepeats = 128;

bool check(bool condition, const char *message)
{
    if (!condition)
        std::fprintf(stderr, "FAIL: %s\n", message);
    return condition;
}

uint32_t float_bits(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

size_t group_count(size_t depth)
{
    return depth / kGroupSize + (depth % kGroupSize != 0);
}

std::vector<block_q8_h1> make_source(size_t columns, size_t depth)
{
    std::vector<block_q8_h1> source(columns * group_count(depth));
    for (size_t j = 0; j < columns; ++j)
        for (size_t group = 0; group < group_count(depth); ++group)
        {
            block_q8_h1 &block = source[j * group_count(depth) + group];
            for (size_t offset = 0; offset < kGroupSize; ++offset)
                block.qs[offset] = static_cast<int8_t>(
                    static_cast<int>(j * 17 + group * 13 + offset * 7) % 255 - 127);
            block.c_b = static_cast<uint8_t>(11 + (j + group) % 193);
            block.s_rf = 0.00032747327350080013f + static_cast<float>(j % 7) * 0.0001f;
            block.R = static_cast<uint16_t>(4095 - (j + group) % 23);
        }
    return source;
}

bool same_ints(const std::vector<int64_t> &lhs, const std::vector<int64_t> &rhs)
{
    return lhs.size() == rhs.size() &&
        std::memcmp(lhs.data(), rhs.data(), lhs.size() * sizeof(int64_t)) == 0;
}

bool same_floats(const std::vector<float> &lhs, const std::vector<float> &rhs)
{
    return lhs.size() == rhs.size() &&
        std::memcmp(lhs.data(), rhs.data(), lhs.size() * sizeof(float)) == 0;
}

std::array<DecH1JPackEntry, 4> safe_entries()
{
    return {{{1, 1000}, {7, -900}, {14, 800}, {23, -700}}};
}

std::array<DecH1JPackEntry, 4> benchmark_entries()
{
    return {{{321, 1000}, {327, -900}, {334, 800}, {343, -700}}};
}

bool test_pack_layout_and_accounting()
{
    constexpr size_t columns = 9;
    constexpr size_t depth = 65;
    std::vector<block_q8_h1> source = make_source(columns, depth);
    DecH1JPack pack(source.data(), columns, depth);
    const DecH1JPackAccounting &accounting = pack.accounting();
    bool ok = check(pack.valid(), "J pack allocates") &&
        check(reinterpret_cast<uintptr_t>(pack.code_pointer(0, 0)) % 64 == 0,
              "J pack base is 64-byte aligned") &&
        check(accounting.packed_bytes == 3 * 2 * kGroupSize * kNr &&
                  accounting.allocated_bytes == accounting.packed_bytes &&
                  accounting.source_code_bytes_read == columns * depth &&
                  accounting.packed_bytes_written == accounting.packed_bytes &&
                  accounting.panel_count == 2 && accounting.group_count == 3 &&
                  accounting.tail_panel_count == 1,
              "J pack accounting is exact");
    for (size_t k_group = 0; k_group < pack.group_count(); ++k_group)
        for (size_t panel = 0; panel < pack.panel_count(); ++panel)
            for (size_t k_offset = 0; k_offset < kGroupSize; ++k_offset)
                for (size_t lane = 0; lane < kNr; ++lane)
                {
                    const size_t j = panel * kNr + lane;
                    const size_t k = k_group * kGroupSize + k_offset;
                    const int8_t expected = j < columns && k < depth ?
                        source[j * group_count(depth) + k_group].qs[k_offset] : 0;
                    ok = check(pack.packed_code(k_group, panel, k_offset, lane) == expected,
                               "J pack order and zero tails") && ok;
                }
    const int8_t retained = pack.packed_code(0, 0, 0, 0);
    source[0].qs[0] = static_cast<int8_t>(retained + 1);
    return check(pack.packed_code(0, 0, 0, 0) == retained,
                 "J pack stays immutable after source mutation") && ok;
}

bool test_safety_routes()
{
    std::vector<block_q8_h1> source = make_source(8, 65);
    DecH1JPack pack(source.data(), 8, 65);
    const auto entries = safe_entries();
    const DecH1JPackEntry int64_required[] = {{0, std::numeric_limits<int32_t>::min()}};
    const DecH1JPackEntry invalid_k[] = {{65, 1}};
    const DecH1JPackEntry zero_residual[] = {{0, 0}};
    const DecH1JPackEntry too_many[] = {{0, 1}, {1, 1}, {2, 1}, {3, 1}, {4, 1}};
    return check(ggml::gemmini::quants::dec::dec_h1_jpack_classify(
                     pack, entries.data(), entries.size()) == DecH1JPackSafety::INT32_SAFE,
                 "INT32 safety bound accepts S1-S4") &&
        check(ggml::gemmini::quants::dec::dec_h1_jpack_classify(
                  pack, int64_required, 1) == DecH1JPackSafety::INT64_REQUIRED,
              "INT32_MIN is overflow-safe and selects INT64") &&
        check(ggml::gemmini::quants::dec::dec_h1_jpack_classify(
                  pack, nullptr, 0) == DecH1JPackSafety::INVALID,
              "empty group is invalid") &&
        check(ggml::gemmini::quants::dec::dec_h1_jpack_classify(
                  pack, invalid_k, 1) == DecH1JPackSafety::INVALID,
              "out-of-range K is invalid") &&
        check(ggml::gemmini::quants::dec::dec_h1_jpack_classify(
                  pack, zero_residual, 1) == DecH1JPackSafety::INVALID,
              "zero residual is malformed") &&
        check(ggml::gemmini::quants::dec::dec_h1_jpack_classify(
                  pack, too_many, 5) == DecH1JPackSafety::INVALID,
              "more than four entries is invalid for NR8");
}

bool test_integer_parity_for_shape(size_t columns, size_t depth)
{
    std::vector<block_q8_h1> source = make_source(columns, depth);
    DecH1JPack pack(source.data(), columns, depth);
    bool ok = true;
    for (size_t count = 1; count <= 4; ++count)
    {
        const auto all_entries = safe_entries();
        std::vector<int64_t> source_output(columns);
        std::vector<int64_t> packed_output(columns);
        DecH1JPackCounters counters;
        ok = check(ggml::gemmini::quants::dec::dec_h1_jpack_execute_source_scalar(
                       source.data(), columns, depth, all_entries.data(), count, source_output.data()),
                   "source scalar executes") && ok;
        ok = check(ggml::gemmini::quants::dec::dec_h1_jpack_execute_packed(
                       pack, all_entries.data(), count, packed_output.data(), counters),
                   "packed integer executes") && ok;
        ok = check(same_ints(source_output, packed_output),
                   "source and packed integer results are bitwise equal") && ok;
        ok = check(counters.processed_macs == columns * count,
                   "processed MAC counter follows execution") && ok;
        if (columns >= kNr)
            ok = check(counters.scalar_fallback_loads == (columns % kNr) * count,
                       "only J tail uses scalar fallback") && ok;
    }

    const DecH1JPackEntry int64_required[] = {{0, std::numeric_limits<int32_t>::min()}};
    std::vector<int64_t> source_output(columns);
    std::vector<int64_t> packed_output(columns);
    DecH1JPackCounters int64_counters;
    ok = check(ggml::gemmini::quants::dec::dec_h1_jpack_execute_source_scalar(
                   source.data(), columns, depth, int64_required, 1, source_output.data()) &&
                   ggml::gemmini::quants::dec::dec_h1_jpack_execute_packed(
                       pack, int64_required, 1, packed_output.data(), int64_counters) &&
                   same_ints(source_output, packed_output) && int64_counters.vector_loads == 0 &&
                   int64_counters.scalar_fallback_loads == columns,
               "INT64 route is bitwise scalar fallback") && ok;

    const DecH1JPackEntry five_entries[] = {{0, 1}, {1, -2}, {2, 3}, {3, -4}, {4, 5}};
    DecH1JPackCounters wide_counters;
    ok = check(ggml::gemmini::quants::dec::dec_h1_jpack_execute_source_scalar(
                   source.data(), columns, depth, five_entries, 5, source_output.data()) &&
                   ggml::gemmini::quants::dec::dec_h1_jpack_execute_packed(
                       pack, five_entries, 5, packed_output.data(), wide_counters) &&
                   same_ints(source_output, packed_output) && wide_counters.vector_loads == 0 &&
                   wide_counters.scalar_fallback_loads == columns * 5,
               "nnz above four uses scalar reference") && ok;
    return ok;
}

bool test_integer_parity()
{
    bool ok = true;
    for (size_t columns : {size_t {1}, size_t {7}, size_t {8}, size_t {9}, size_t {2304}, size_t {3072}})
        ok = test_integer_parity_for_shape(columns, 65) && ok;
    return ok;
}

bool test_full_h1_bits()
{
    constexpr size_t columns = 9;
    constexpr size_t depth = 65;
    std::vector<block_q8_h1> source = make_source(columns, depth);
    DecH1JPack pack(source.data(), columns, depth);
    const auto entries = safe_entries();
    std::vector<int64_t> source_integer(columns);
    std::vector<int64_t> packed_integer(columns);
    DecH1JPackCounters counters;
    bool ok = check(ggml::gemmini::quants::dec::dec_h1_jpack_execute_source_scalar(
                        source.data(), columns, depth, entries.data(), entries.size(),
                        source_integer.data()) &&
                    ggml::gemmini::quants::dec::dec_h1_jpack_execute_packed(
                        pack, entries.data(), entries.size(), packed_integer.data(), counters),
                    "full H1 integer stage executes");
    std::vector<float> reference(columns);
    for (size_t j = 0; j < columns; ++j)
        reference[j] = -0.25f + static_cast<float>(j) * 0.03125f;
    const std::vector<float> initial = reference;
    std::vector<float> packed = reference;
    for (size_t j = 0; j < columns; ++j)
    {
        const block_q8_h1 &block = source[j * group_count(depth)];
        const uint64_t c_eff = static_cast<uint64_t>(block.c_b) + block.R;
        reference[j] += ggml::gemmini::quants::dec::apply_h1_scale_ordered(
            source_integer[j], c_eff, block.s_rf, 0.5f);
        packed[j] += ggml::gemmini::quants::dec::apply_h1_scale_ordered(
            packed_integer[j], c_eff, block.s_rf, 0.5f);
    }
    bool contribution_changed_output = false;
    for (size_t j = 0; j < columns; ++j)
        contribution_changed_output = contribution_changed_output ||
            float_bits(reference[j]) != float_bits(initial[j]);
    return check(same_floats(reference, packed) && contribution_changed_output,
                  "full H1 keeps ordered scale and nonzero Y_com bits") && ok;
}

struct Timing
{
    double median_ns = 0.0;
    double p90_ns = 0.0;
};

template <typename Prepare, typename Execute, typename Observe>
Timing measure(Prepare prepare, Execute execute, Observe observe, uint64_t &sink)
{
    for (size_t warmup = 0; warmup < kWarmups; ++warmup)
    {
        prepare();
        execute();
        sink += observe() ^ static_cast<uint64_t>(warmup + 1);
    }
    std::array<uint64_t, kSamples> samples{};
    for (size_t sample = 0; sample < kSamples; ++sample)
    {
        prepare();
        const auto start = std::chrono::steady_clock::now();
        execute();
        const auto end = std::chrono::steady_clock::now();
        samples[sample] = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
        sink += observe() ^ samples[sample] ^ static_cast<uint64_t>(sample + 1);
    }
    std::sort(samples.begin(), samples.end());
    return {
        static_cast<double>(samples[kSamples / 2]) / kRepeats,
        static_cast<double>(samples[(kSamples * 9 + 9) / 10 - 1]) / kRepeats,
    };
}

double gmac_per_second(size_t columns, size_t nnz, double ns)
{
    return static_cast<double>(columns * nnz) / ns;
}

struct IntegerBench
{
    Timing scalar;
    Timing packed;
    DecH1JPackCounters counters;
    bool parity = false;
};

IntegerBench benchmark_integer(
    const std::vector<block_q8_h1> &source, const DecH1JPack &pack,
    const DecH1JPackEntry *entries, size_t entry_count)
{
    std::vector<int64_t> scalar_output(pack.columns());
    std::vector<int64_t> packed_output(pack.columns());
    uint64_t sink = 0;
    const Timing scalar = measure(
        [] {},
        [&] {
            for (size_t repeat = 0; repeat < kRepeats; ++repeat)
                ggml::gemmini::quants::dec::dec_h1_jpack_execute_source_scalar(
                    source.data(), pack.columns(), pack.depth(), entries, entry_count,
                    scalar_output.data());
        }, [&] { return static_cast<uint64_t>(scalar_output[sink % scalar_output.size()]); }, sink);
    const Timing packed = measure(
        [] {},
        [&] {
            for (size_t repeat = 0; repeat < kRepeats; ++repeat)
            {
                DecH1JPackCounters ignored;
                ggml::gemmini::quants::dec::dec_h1_jpack_execute_packed(
                    pack, entries, entry_count, packed_output.data(), ignored);
            }
        }, [&] { return static_cast<uint64_t>(packed_output[sink % packed_output.size()]); }, sink);
    DecH1JPackCounters counters;
    ggml::gemmini::quants::dec::dec_h1_jpack_execute_packed(
        pack, entries, entry_count, packed_output.data(), counters);
    ggml::gemmini::quants::dec::dec_h1_jpack_execute_source_scalar(
        source.data(), pack.columns(), pack.depth(), entries, entry_count, scalar_output.data());
    std::printf("integer sink=%llu\n", static_cast<unsigned long long>(sink));
    return {scalar, packed, counters, same_ints(scalar_output, packed_output)};
}

struct FullH1Bench
{
    Timing scalar;
    Timing packed;
    bool parity = false;
};

FullH1Bench benchmark_full_h1(
    const std::vector<block_q8_h1> &source, const DecH1JPack &pack,
    const DecH1JPackEntry *entries, size_t entry_count)
{
    std::vector<int64_t> scalar_integer(pack.columns());
    std::vector<int64_t> packed_integer(pack.columns());
    std::vector<float> initial(pack.columns());
    std::vector<float> scalar_ycom(pack.columns());
    std::vector<float> packed_ycom(pack.columns());
    std::vector<float> scalar_batch(kRepeats * pack.columns());
    std::vector<float> packed_batch(kRepeats * pack.columns());
    for (size_t j = 0; j < pack.columns(); ++j)
        initial[j] = -0.25f + static_cast<float>(j) * 0.03125f;
    const auto apply = [&](const std::vector<int64_t> &integer, float *ycom)
    {
        for (size_t j = 0; j < pack.columns(); ++j)
        {
            const size_t k_group = entries[0].k / kGroupSize;
            const block_q8_h1 &block = source[j * group_count(pack.depth()) + k_group];
            ycom[j] += ggml::gemmini::quants::dec::apply_h1_scale_ordered(
                integer[j], static_cast<uint64_t>(block.c_b) + block.R, block.s_rf, 0.5f);
        }
    };
    uint64_t sink = 0;
    const Timing scalar = measure(
        [&] {
            for (size_t repeat = 0; repeat < kRepeats; ++repeat)
                std::copy(initial.begin(), initial.end(),
                          scalar_batch.begin() + repeat * pack.columns());
        },
        [&] {
            for (size_t repeat = 0; repeat < kRepeats; ++repeat)
            {
                ggml::gemmini::quants::dec::dec_h1_jpack_execute_source_scalar(
                    source.data(), pack.columns(), pack.depth(), entries, entry_count,
                    scalar_integer.data());
                apply(scalar_integer, scalar_batch.data() + repeat * pack.columns());
            }
        }, [&] { return float_bits(scalar_batch[sink % scalar_batch.size()]); }, sink);
    const Timing packed = measure(
        [&] {
            for (size_t repeat = 0; repeat < kRepeats; ++repeat)
                std::copy(initial.begin(), initial.end(),
                          packed_batch.begin() + repeat * pack.columns());
        },
        [&] {
            for (size_t repeat = 0; repeat < kRepeats; ++repeat)
            {
                DecH1JPackCounters ignored;
                ggml::gemmini::quants::dec::dec_h1_jpack_execute_packed(
                    pack, entries, entry_count, packed_integer.data(), ignored);
                apply(packed_integer, packed_batch.data() + repeat * pack.columns());
            }
        }, [&] { return float_bits(packed_batch[sink % packed_batch.size()]); }, sink);
    scalar_ycom = initial;
    packed_ycom = initial;
    ggml::gemmini::quants::dec::dec_h1_jpack_execute_source_scalar(
        source.data(), pack.columns(), pack.depth(), entries, entry_count, scalar_integer.data());
    DecH1JPackCounters counters;
    ggml::gemmini::quants::dec::dec_h1_jpack_execute_packed(
        pack, entries, entry_count, packed_integer.data(), counters);
    apply(scalar_integer, scalar_ycom.data());
    apply(packed_integer, packed_ycom.data());
    std::printf("full-h1 sink=%llu\n", static_cast<unsigned long long>(sink));
    return {scalar, packed, same_floats(scalar_ycom, packed_ycom)};
}

bool run_benchmarks()
{
    constexpr size_t depth = 768;
    bool gate_a = true;
    bool gate_b = true;
    for (size_t columns : {size_t {2304}, size_t {3072}})
    {
        std::vector<block_q8_h1> source = make_source(columns, depth);
        const auto first_start = std::chrono::steady_clock::now();
        DecH1JPack pack(source.data(), columns, depth);
        const auto entries = benchmark_entries();
        std::vector<int64_t> first_output(columns);
        DecH1JPackCounters first_counters;
        const bool first_ok = ggml::gemmini::quants::dec::dec_h1_jpack_execute_packed(
            pack, entries.data(), entries.size(), first_output.data(), first_counters);
        const auto first_end = std::chrono::steady_clock::now();
        const uint64_t first_use_ns = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(first_end - first_start).count());
        const DecH1JPackAccounting &accounting = pack.accounting();
        std::printf(
            "pack J=%zu K=%zu packed_bytes=%zu allocated_bytes=%zu pack_ns=%llu source_code_bytes=%zu packed_writes=%zu panels=%zu groups=%zu tail_panels=%zu first_use_pack_execute_ns=%llu\n",
            columns, depth, accounting.packed_bytes, accounting.allocated_bytes,
            static_cast<unsigned long long>(accounting.pack_time_ns), accounting.source_code_bytes_read,
            accounting.packed_bytes_written, accounting.panel_count, accounting.group_count,
            accounting.tail_panel_count, static_cast<unsigned long long>(first_use_ns));
        gate_a = gate_a && first_ok;

        for (size_t nnz = 1; nnz <= 4; ++nnz)
        {
            const IntegerBench bench = benchmark_integer(source, pack, entries.data(), nnz);
            const double speedup = bench.scalar.median_ns / bench.packed.median_ns;
            std::printf(
                "integer J=%zu S%zu samples=%zu warmup=%zu scalar_ns median=%.2f p90=%.2f packed_ns median=%.2f p90=%.2f scalar_gmac_s=%.3f packed_gmac_s=%.3f speedup=%.3fx counters vector_hits=%llu vector_loads=%llu scalar_loads=%llu macs=%llu parity=%s steady_execute_ns=%.2f\n",
                columns, nnz, kSamples, kWarmups, bench.scalar.median_ns, bench.scalar.p90_ns,
                bench.packed.median_ns, bench.packed.p90_ns,
                gmac_per_second(columns, nnz, bench.scalar.median_ns),
                gmac_per_second(columns, nnz, bench.packed.median_ns), speedup,
                static_cast<unsigned long long>(bench.counters.vector_kernel_hits),
                static_cast<unsigned long long>(bench.counters.vector_loads),
                static_cast<unsigned long long>(bench.counters.scalar_fallback_loads),
                static_cast<unsigned long long>(bench.counters.processed_macs),
                bench.parity ? "yes" : "no", bench.packed.median_ns);
            gate_a = gate_a && bench.parity && speedup >= 2.0 &&
                bench.counters.vector_loads > 0 && bench.counters.scalar_fallback_loads == 0;
        }

        const FullH1Bench full = benchmark_full_h1(source, pack, entries.data(), entries.size());
        const double full_speedup = full.scalar.median_ns / full.packed.median_ns;
        const double threshold = columns == 2304 ? 1.2 : 1.3;
        std::printf(
            "full-h1 J=%zu nnz=4 samples=%zu warmup=%zu scalar_ns median=%.2f p90=%.2f packed_ns median=%.2f p90=%.2f speedup=%.3fx parity=%s steady_execute_ns=%.2f\n",
            columns, kSamples, kWarmups, full.scalar.median_ns, full.scalar.p90_ns,
            full.packed.median_ns, full.packed.p90_ns, full_speedup,
            full.parity ? "yes" : "no", full.packed.median_ns);
        gate_b = gate_b && full.parity && full_speedup >= threshold &&
            full.packed.p90_ns <= full.scalar.p90_ns;
    }
    std::printf("Gate A: %s\n", gate_a ? "PASS" : "FAIL");
    std::printf("Gate B: %s\n", gate_b ? "PASS" : "FAIL");
    return gate_a && gate_b;
}
}

int main()
{
    const bool correctness = test_pack_layout_and_accounting() && test_safety_routes() &&
        test_integer_parity() && test_full_h1_bits();
    const bool performance_gates = run_benchmarks();
    std::printf("gemmini DEC H1 J-pack correctness: %s\n", correctness ? "PASS" : "FAIL");
    std::printf("gemmini DEC H1 J-pack performance gates: %s (informational)\n",
                performance_gates ? "PASS" : "FAIL");
    return correctness ? 0 : 1;
}
