#pragma once
#include "../ggml-npu-common/types/geometry.hpp"

namespace ggml::gemmini {
using namespace ggml::npu;

constexpr NpuGeometryResult make_potal_geometry(NpuLogicalShape shape, unsigned bits) {
    if (bits != 4 && bits != 8)
        return {{}, NpuGeometryError::unsupported_operand_width};
    if (shape.i == 0 || shape.j == 0 || shape.k == 0)
        return {{}, NpuGeometryError::zero_dimension};
    // Original DIM32 quantization profile: 4 banks, 4096/2048 rows, 512 accumulator rows.
    constexpr size_t dim = 32, acc = 512 / 2;
    const size_t spad = 32768 / bits;
    const size_t ip = geometry_detail::ceil_div(shape.i, dim);
    const size_t jp = geometry_detail::ceil_div(shape.j, dim);
    const size_t kp = geometry_detail::ceil_div(shape.k, dim);
    NpuTileFactors tiles{std::min(ip, size_t{2}), std::min(jp, size_t{2}),
                             std::min(kp, spad / (4 * dim))};
    for (;;) {
        bool grew = false;
        if (tiles.j < jp && (tiles.i + tiles.j + 1) * tiles.k * dim <= spad &&
            tiles.i * (tiles.j + 1) * dim <= acc) {
            ++tiles.j;
            grew = true;
        }
        if (tiles.i < ip && (tiles.i + 1 + tiles.j) * tiles.k * dim <= spad &&
            (tiles.i + 1) * tiles.j * dim <= acc) {
            ++tiles.i;
            grew = true;
        }
        if (tiles.k < kp && (tiles.i + tiles.j) * (tiles.k + 1) * dim <= spad) {
            ++tiles.k;
            grew = true;
        }
        if (!grew)
            return make_npu_geometry({shape, tiles, dim});
    }
}

} // namespace ggml::gemmini
