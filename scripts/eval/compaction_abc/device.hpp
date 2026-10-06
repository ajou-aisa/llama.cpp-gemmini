#pragma once
#include "types.hpp"
#include <im2p_cycle_model.h>
#include <cmath>
#include <memory>

namespace abc {
inline std::array<size_t, 3> tile_geometry(const Packed &p, size_t n, size_t dim) {
    // Existing gemmini_set_tile_ws policy with the resolved packed-width memory profile.
    const size_t row_bytes = dim * p.bits / 8;
    const size_t bank_rows = 262144 / (4 * row_bytes);
    const size_t acc_rows = 65536 / (dim * 4);
    const size_t sp_limit = 4 * bank_rows / 2, acc_limit = acc_rows / 2;
    const size_t square = std::sqrt((acc_rows / 2) / dim);
    const size_t kmax = ((4 * bank_rows / 4) / dim) / square;
    const size_t mi = (p.m + dim - 1) / dim, nj = (n + dim - 1) / dim, kk = (p.k + dim - 1) / dim;
    size_t i = std::min(mi, square), j = std::min(nj, square), k = std::min(kk, kmax);
    const auto fits = [&](size_t ti, size_t tj, size_t tk) {
        return (ti + tj) * tk * dim <= sp_limit && ti * tj * dim <= acc_limit;
    };
    while (true) {
        bool changed = false;
        if (j < nj && fits(i,j+1,k)) { ++j; changed = true; }
        if (i < mi && fits(i+1,j,k)) { ++i; changed = true; }
        if (k < kk && fits(i,j,k+1)) { ++k; changed = true; }
        if (!changed) return {i,j,k};
    }
}
struct DeviceResult { uint64_t cycles, fragments, loads, stores, scales; };
inline DeviceResult estimate(const Input &input, const Packed &packed, size_t dim) {
    im2p_cycle_model_config_t config{};
    im2p_cycle_model_config_init(&config);
    config.max_cycles = 1000000000;
    config.max_fragments = 10000000;
    if (const char *limit = std::getenv("ABC_MAX_CYCLES")) config.max_cycles = std::stoull(limit);
    const size_t row_bytes = dim * input.bits / 8;
    config.hardware = {static_cast<uint32_t>(input.bits),static_cast<uint32_t>(input.bits),static_cast<uint32_t>(dim),32,32,4,
        static_cast<uint32_t>(262144 / (4 * row_bytes)),static_cast<uint32_t>(65536 / (dim * 4)),
        static_cast<uint32_t>(row_bytes),static_cast<uint32_t>(dim * 4),4,2};
    std::unique_ptr<im2p_cycle_model_t, decltype(&im2p_cycle_model_destroy)> model(
        im2p_cycle_model_create(&config), im2p_cycle_model_destroy);
    if (!model) throw std::runtime_error("cycle model initialization failed");
    im2p_cycle_request_t request{};
    im2p_cycle_request_init(&request);
    request.m = packed.m; request.n = input.n; request.k = packed.k;
    const auto tiles = tile_geometry(packed, input.n, dim);
    request.tile_i = tiles[0]; request.tile_j = tiles[1]; request.tile_k = tiles[2];
    std::vector<im2p_compact_run_t> runs;
    for (const Run &run : packed.runs) runs.push_back({run.block,run.mask,run.begin,run.count});
    const im2p_compact_runs_t view{1,sizeof(im2p_compact_runs_t),static_cast<uint32_t>(input.k),runs.size(),runs.data()};
    im2p_cycle_result_t output{};
    if (im2p_cycle_estimate_runs(model.get(), &request, &view, &output) != 0)
        throw std::runtime_error(im2p_cycle_model_error(model.get()));
    return {output.total_cycles,output.fragment_count,output.load_request_count,output.store_request_count,output.scale_request_count};
}
}
