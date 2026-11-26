#pragma once

#include <cstddef>
#include <limits>
#include <algorithm>

namespace ggml::npu {

struct NpuLogicalShape {
    size_t i;
    size_t j;
    size_t k;
};

struct NpuTileFactors {
    size_t i;
    size_t j;
    size_t k;
};

struct NpuOuterCounts {
    size_t i;
    size_t j;
    size_t k;
};

struct NpuGeometryInput {
    NpuLogicalShape shape;
    NpuTileFactors  tiles;
    size_t              array_dim;
};

struct NpuGeometry {
    NpuLogicalShape shape{};
    NpuTileFactors  tiles{};
    NpuOuterCounts  outer{};
    size_t              stripe_rows    = 0;
    size_t              stripe_count   = 0;
    size_t              final_rows     = 0;
    size_t              ws_inner_calls = 0;
};

enum class NpuGeometryError {
    none,
    zero_dimension,
    zero_array_dimension,
    zero_tile_factor,
    overflow,
    unsupported_operand_width,
};

struct NpuGeometryResult {
    NpuGeometry      geometry{};
    NpuGeometryError error = NpuGeometryError::none;

    constexpr bool ok() const {
        return error == NpuGeometryError::none;
    }
};

namespace geometry_detail {

constexpr size_t ceil_div(size_t value, size_t divisor) {
    return value / divisor + static_cast<size_t>(value % divisor != 0);
}

constexpr bool checked_multiply(size_t lhs, size_t rhs, size_t & result) {
    if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs) {
        return false;
    }
    result = lhs * rhs;
    return true;
}

} // namespace geometry_detail

constexpr NpuGeometryResult make_npu_geometry(NpuGeometryInput input) {
    if (input.shape.i == 0 || input.shape.j == 0 || input.shape.k == 0) {
        return {{}, NpuGeometryError::zero_dimension};
    }
    if (input.array_dim == 0) {
        return {{}, NpuGeometryError::zero_array_dimension};
    }
    if (input.tiles.i == 0 || input.tiles.j == 0 || input.tiles.k == 0) {
        return {{}, NpuGeometryError::zero_tile_factor};
    }

    size_t tile_i_rows = 0;
    size_t tile_j_rows = 0;
    size_t tile_k_rows = 0;
    if (!geometry_detail::checked_multiply(input.tiles.i, input.array_dim, tile_i_rows) ||
        !geometry_detail::checked_multiply(input.tiles.j, input.array_dim, tile_j_rows) ||
        !geometry_detail::checked_multiply(input.tiles.k, input.array_dim, tile_k_rows)) {
        return {{}, NpuGeometryError::overflow};
    }

    const NpuOuterCounts outer{
        geometry_detail::ceil_div(input.shape.i, tile_i_rows),
        geometry_detail::ceil_div(input.shape.j, tile_j_rows),
        geometry_detail::ceil_div(input.shape.k, tile_k_rows),
    };
    size_t ij_calls       = 0;
    size_t ws_inner_calls = 0;
    if (!geometry_detail::checked_multiply(outer.i, outer.j, ij_calls) ||
        !geometry_detail::checked_multiply(ij_calls, outer.k, ws_inner_calls)) {
        return {{}, NpuGeometryError::overflow};
    }

    const size_t remainder = input.shape.i % tile_i_rows;
    return {{input.shape,
             input.tiles,
             outer,
             tile_i_rows,
             outer.i,
             remainder == 0 ? tile_i_rows : remainder,
             ws_inner_calls},
            NpuGeometryError::none};
}

} // namespace ggml::npu
