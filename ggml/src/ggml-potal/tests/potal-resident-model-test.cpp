// E4A numerical regression: software-only, no resident lifecycle or command API.
// Serial uses its existing automatic tiling; the microtile and active H2 routes
// must retain I1/J9/K28. No timing, traffic, or hardware evidence is produced.
#include "support/paired-support.hpp"
#include "matmul/paired.hpp"
#include "matmul/paired-auto.hpp"
#include "support/paired-h2-emulator.hpp"

#include <algorithm>

using namespace ggml::potal;
using namespace potal_test;

namespace {

constexpr size_t M = 17, N = 145, K = 928;
static_assert(PD == 16, "E4A fixtures require DIM16");
using wide = __int128_t;

int32_t clamp32(wide value) {
    return value < INT32_MIN ? INT32_MIN : value > INT32_MAX ? INT32_MAX : int32_t(value);
}

// Independent threshold arithmetic, not the production or emulator SCU helper.
int32_t shifted(int64_t dot, uint32_t carrier) {
    if (carrier == POTAL_CARRIER_ZERO || dot == 0) return 0;
    if (carrier >= 32) return dot < 0 ? INT32_MIN : INT32_MAX;
    const wide factor = wide(1) << carrier;
    if (wide(dot) > wide(INT32_MAX) / factor) return INT32_MAX;
    if (wide(dot) < wide(INT32_MIN) / factor) return INT32_MIN;
    return int32_t(wide(dot) * factor);
}

// Packet decoding is shared, as in E3; arithmetic, DIM boundaries, accumulation,
// and wide radix composition are independent of all three execution routes.
bool independent(const paired_fixture & f, const paired_inputs & in, paired_output & out) {
    out.main_c.assign(M * N, 0);
    out.stripes.resize(in.packets.size());
    for (size_t m = 0; m < M; ++m) for (size_t n = 0; n < N; ++n) {
        int32_t acc = 0;
        for (size_t first = 0; first < K; first += PD) {
            int64_t dot = 0;
            for (size_t k = first; k < first + PD; ++k)
                dot += int64_t(f.a.get()[m * f.lda + k]) * f.code_at(k, n);
            acc = clamp32(wide(acc) + shifted(dot, f.carrier_at(first / 32, n)));
        }
        out.main_c[m * N + n] = acc;
    }
    for (size_t s = 0; s < in.packets.size(); ++s) {
        if (!in.packets[s]) continue;
        residual::decoded_packet packet;
        if (residual::decode_packet(*in.packets[s], f.weight, packet) != grmd::RmdStatus::success) return false;
        auto & got = out.stripes[s];
        got.rows = packet.rows;
        got.residual_c.assign(packet.rows.size() * N, 0);
        for (size_t row = 0; row < packet.rows.size(); ++row) for (size_t n = 0; n < N; ++n) {
            int32_t acc = 0;
            size_t compact = 0;
            for (const auto & run : packet.runs) {
                for (size_t first = 0; first < run.local_k.size(); first += PD) {
                    int64_t dot = 0;
                    for (size_t x = first; x < std::min(first + PD, run.local_k.size()); ++x) {
                        const size_t original = run.block_id * 32 + run.local_k[x];
                        dot += int64_t(packet.digits[row * packet.k_compact + compact + x]) *
                               f.code_at(original, n);
                    }
                    acc = clamp32(wide(acc) + shifted(dot, f.carrier_at(run.block_id, n)));
                }
                compact += run.local_k.size();
            }
            got.residual_c[row * N + n] = acc;
        }
        const size_t rows = std::min(PD, M - s * PD);
        std::vector<wide> correction(rows * N, 0);
        for (size_t row = 0; row < got.rows.size(); ++row) {
            wide place = 1;
            for (size_t lane = 0; lane < got.rows[row].lane; ++lane) place *= 256;
            for (size_t n = 0; n < N; ++n)
                correction[got.rows[row].source_row * N + n] += wide(got.residual_c[row * N + n]) * place;
        }
        for (wide value : correction) {
            if (value < INT64_MIN || value > INT64_MAX) return false;
            got.correction.push_back(int64_t(value));
        }
    }
    return true;
}

bool equal_raw(const paired_output & expected, const paired_output & actual) {
    if (!same_bits(expected.main_c, actual.main_c) || expected.stripes.size() != actual.stripes.size()) return false;
    for (size_t s = 0; s < expected.stripes.size(); ++s) {
        const auto & a = expected.stripes[s];
        const auto & b = actual.stripes[s];
        if (a.rows.size() != b.rows.size() || !same_bits(a.residual_c, b.residual_c) ||
            !same_bits(a.correction, b.correction)) return false;
        for (size_t row = 0; row < a.rows.size(); ++row)
            if (a.rows[row].lane != b.rows[row].lane || a.rows[row].source_row != b.rows[row].source_row)
                return false;
    }
    return true;
}

bool active_h2(const paired_fixture & f, const paired_inputs & in, paired_output & out) {
    potal_tile_plan main;
    if (potal_frozen_plan(M, N, K, 1, 9, 28, 2, &main) != POTAL_OK ||
        main.tile_I != 1 || main.tile_J != 9 || main.tile_K != 28) return false;
    out.main_c.resize(M * N);
    out.stripes.resize(in.packets.size());
    for (size_t s = 0; s < in.packets.size(); ++s) {
        const size_t begin = s * PD, rows = std::min(PD, M - begin);
        residual::decoded_packet decoded;
        if (in.packets[s] &&
            residual::decode_packet(*in.packets[s], f.weight, decoded) != grmd::RmdStatus::success) return false;
        const auto * packet = in.packets[s] ? &decoded : nullptr;
        potal_paired_tile_plan tile;
        potal_paired_execution_plan plan;
        std::string why;
        if (potal_paired_plan_stripe(main, rows, N, K, packet, tile, plan, &why) != potal_paired_status::ok ||
            tile.tile_I != 1 || tile.tile_J != 9 || tile.tile_K != 28 ||
            tile.k_panels != 3 || tile.j_groups != 2) return false;
        const size_t ldc = align_up(N, POTAL_HW_MAX_BYTES / sizeof(int32_t));
        std::unique_ptr<int32_t, dma_deleter> mc(static_cast<int32_t *>(dma_alloc(align_up(rows, PD) * ldc * sizeof(int32_t))));
        std::unique_ptr<int32_t, dma_deleter> rc(static_cast<int32_t *>(
            dma_alloc(align_up(std::max(size_t(1), decoded.rows.size()), PD) * ldc * sizeof(int32_t))));
        if (!mc || !rc) return false;
        uint32_t generation = 254;
        potal_paired_request request;
        request.stripe_id = s; request.row_begin = begin; request.rows = rows; request.n = N; request.k = K;
        request.main_plan = &main; request.a = f.a.get() + begin * f.lda; request.lda = f.lda;
        request.weight = &f.weight; request.residual = packet;
        request.c_main = mc.get(); request.ldc_main = ldc; request.c_residual = rc.get(); request.ldc_residual = ldc;
        request.generation = &generation; request.plan = &plan;
        h2_emulator emulator;
        auto sink = emulator.sink();
        potal_paired_report report;
        if (potal_paired_run_session(request, &sink, report) != potal_paired_status::ok ||
            report.fatal || !emulator.errors.empty() || emulator.scale_miss || report.tile.tile_K != 28) return false;
        for (size_t row = 0; row < rows; ++row)
            std::copy_n(mc.get() + row * ldc, N, out.main_c.data() + (begin + row) * N);
        auto & got = out.stripes[s];
        got.rows = decoded.rows;
        got.residual_c.resize(decoded.rows.size() * N);
        for (size_t row = 0; row < decoded.rows.size(); ++row)
            std::copy_n(rc.get() + row * ldc, N, got.residual_c.data() + row * N);
        if (in.packets[s]) {
            grmd::BlockScaledInt64Correction correction;
            if (residual::reconstruct_rows(*in.packets[s], got.rows, rc.get(), ldc, N, correction) !=
                grmd::RmdStatus::success) return false;
            got.correction = std::move(correction.values);
        }
    }
    return true;
}

// The three audit witnesses use only lane0/source_row0. Assert constants at
// every column, both J groups and both stripes, including the one-row tail.
bool witness(const paired_output & out, int32_t main, int32_t residual) {
    if (out.main_c.size() != M * N || out.stripes.size() != 2 ||
        !std::all_of(out.main_c.begin(), out.main_c.end(), [=](int32_t x) { return x == main; })) return false;
    for (size_t s = 0; s < out.stripes.size(); ++s) {
        const auto & got = out.stripes[s];
        if (got.rows.size() != 1 || got.rows[0].lane != 0 || got.rows[0].source_row != 0 ||
            got.residual_c.size() != N || got.correction.size() != std::min(PD, M - s * PD) * N) return false;
        for (size_t n = 0; n < N; ++n) if (got.residual_c[n] != residual) return false;
        for (size_t x = 0; x < got.correction.size(); ++x)
            if (got.correction[x] != (x < N ? int64_t(residual) : 0)) return false;
    }
    return true;
}

enum class fixture_kind { zero, sparse, dense, cross_window, compact, fragments };

void run_case(const char * name, fixture_kind kind) {
    const bool manual = kind == fixture_kind::cross_window || kind == fixture_kind::compact ||
                        kind == fixture_kind::fragments;
    const std::string label = std::string("E4A software-only ") + name;
    paired_fixture f(M, N, K, 20261023,
        [=](size_t block, size_t n) -> int16_t {
            if (manual) return kind == fixture_kind::fragments ? 24 : 31;
            return (block + n) % 11 == 0 ? INT16_MIN : int16_t((block + n) % 4);
        },
        [=](size_t k, size_t n, std::mt19937 &) -> int8_t {
            if (kind == fixture_kind::zero) return 0;
            if (kind == fixture_kind::cross_window) return k == 0 || k == 416 ? 1 : k == 448 ? -1 : 0;
            if (kind == fixture_kind::compact) return k == 0 ? 1 : k == 16 ? -1 : 0;
            if (kind == fixture_kind::fragments) return k < 16 ? 127 : k == 16 ? -127 : 0;
            return int8_t(int((k * 3 + n) % 7) - 3);
        },
        [=](size_t row, size_t k, std::mt19937 &) -> int8_t {
            return manual ? 1 : int8_t(int((row + k) % 5) - 2);
        });
    paired_tile_plan frozen;
    paired_inputs in;
    const bool ready = f.ok && paired_frozen_plan(M, N, K, 1, 9, 28, frozen) == POTAL_OK &&
        frozen.frozen && frozen.tile_I == 1 && frozen.tile_J == 9 && frozen.tile_K == 28 &&
        frozen.requested_tile_K == 28 && frozen.stripe_rows == 16 && frozen.stripes == 2 &&
        frozen.j_groups == 2 && frozen.k_windows == 3 &&
        make_inputs(f, PD, [=](size_t stripe, size_t rows) {
            std::vector<res_entry> entries;
            if (kind == fixture_kind::zero) return entries;
            if (kind == fixture_kind::cross_window) return std::vector<res_entry>{{0, 0, 1}, {0, 416, 1}, {0, 448, 1}};
            if (kind == fixture_kind::compact) return std::vector<res_entry>{{0, 0, 1}, {0, 16, 1}};
            if (kind == fixture_kind::fragments) {
                for (size_t k = 0; k < 17; ++k) entries.push_back({0, k, 1});
                return entries;
            }
            const size_t counts[] = {1, 15, 16, 17, 31, 32};
            for (size_t block = 0; block < K / 32; ++block) {
                const size_t count = kind == fixture_kind::dense ? 32 : counts[block % 6];
                for (size_t x = 0; x < count; ++x) {
                    const int32_t sign = (block + x + stripe) % 2 ? -1 : 1;
                    if (kind == fixture_kind::dense) {
                        for (size_t row = 0; row < rows; ++row)
                            entries.push_back({row, block * 32 + x, sign * 0x10101});
                    } else {
                        entries.push_back({(block + x + stripe) % rows, block * 32 + x, sign * 7});
                    }
                }
            }
            return entries;
        }, in);
    check(label + " fixture retains I1/J9/K28 and M/N/K tails", ready);
    if (!ready) return;

    paired_output expected;
    const bool golden_ok = independent(f, in, expected);
    check(label + " independent wide arithmetic", golden_ok);
    if (!golden_ok) return;
    // Serial Main deliberately remains automatic; K28 is not imposed on it.
    const auto serial = serialized_of(f, PD, in);
    paired_output serial_raw;
    serial_raw.main_c = serial.main_c;
    serial_raw.stripes.resize(in.packets.size());
    for (size_t s = 0; s < in.packets.size(); ++s) {
        serial_raw.stripes[s].rows = serial.rows[s];
        serial_raw.stripes[s].residual_c = serial.residual_c[s];
        serial_raw.stripes[s].correction = serial.correction[s];
    }
    check(label + " Serial automatic tiling raw Main/Residual/INT64",
          serial.main_ok && serial.legacy_ok && equal_raw(expected, serial_raw));

    paired_output microtile;
    paired_execution_report report;
    const auto desc = desc_of(f, frozen, in);
    const bool micro_ok = paired_tiled_matmul_auto(desc, microtile, report) == POTAL_OK &&
        report.execution_backend == paired_execution_backend::paired_emulated &&
        report.plan.tile_I == 1 && report.plan.tile_J == 9 && report.plan.tile_K == 28;
    check(label + " microtile K28 raw Main/Residual/INT64", micro_ok && equal_raw(expected, microtile));
    paired_output active;
    const bool active_ok = active_h2(f, in, active);
    check(label + " active Paired H2 K28 raw Main/Residual/INT64", active_ok && equal_raw(expected, active));
    if (manual) {
        const int32_t main = kind == fixture_kind::fragments ? 16777215 : -1;
        const int32_t residual = kind == fixture_kind::compact ? 0 : main;
        check(label + " manually calculated constants on all routes",
              witness(expected, main, residual) && witness(serial_raw, main, residual) &&
              micro_ok && witness(microtile, main, residual) && active_ok && witness(active, main, residual));
    }
}

} // namespace

int main() {
    std::puts("E4A software-only numerical regression; no hardware, timing, or traffic evidence.");
    // A: MAX, MAX, MIN in original K order across the K28 boundary.
    check("E4A software-only A order versus reassociation",
          shifted(1, 31) == INT32_MAX && shifted(-1, 31) == INT32_MIN &&
          clamp32(wide(clamp32(wide(INT32_MAX) + INT32_MAX)) + INT32_MIN) == -1 &&
          clamp32(wide(clamp32(wide(INT32_MAX) + INT32_MIN)) + INT32_MAX) == 2147483646 &&
          shifted(1 + 1 - 1, 31) == INT32_MAX);
    // B: one compact dot cancels; two original-DIM partials do not.
    check("E4A software-only B compact versus zero-expanded",
          shifted(1 - 1, 31) == 0 && clamp32(wide(shifted(1, 31)) + shifted(-1, 31)) == -1);
    // C: carrier equality never permits merging two compact DIM partials.
    check("E4A software-only C fragment versus merged partial",
          shifted(2032, 24) == INT32_MAX && shifted(-127, 24) == -2130706432 &&
          clamp32(wide(shifted(2032, 24)) + shifted(-127, 24)) == 16777215 &&
          shifted(2032 - 127, 24) == INT32_MAX);
    run_case("zero", fixture_kind::zero);
    run_case("sparse compact tails", fixture_kind::sparse);
    run_case("dense three-lane", fixture_kind::dense);
    run_case("A cross-K28 accumulation", fixture_kind::cross_window);
    run_case("B compact partition", fixture_kind::compact);
    run_case("C separate fragments", fixture_kind::fragments);
    std::printf("E4A software-only failures=%d\n", g_failures);
    return g_failures ? 1 : 0;
}
