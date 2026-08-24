#pragma once

#include <cstddef>
#include <limits>

namespace ggml::gemmini {

struct GemminiLogicalShape {
    size_t i;
    size_t j;
    size_t k;
};

struct GemminiTileFactors {
    size_t i;
    size_t j;
    size_t k;
};

struct GemminiOuterCounts {
    size_t i;
    size_t j;
    size_t k;
};

struct GemminiGeometryInput {
    GemminiLogicalShape shape;
    GemminiTileFactors tiles;
    size_t array_dim;
};

struct GemminiGeometry {
    GemminiLogicalShape shape{};
    GemminiTileFactors tiles{};
    GemminiOuterCounts outer{};
    size_t stripe_rows = 0;
    size_t stripe_count = 0;
    size_t final_rows = 0;
    size_t ws_inner_calls = 0;
};

enum class GemminiGeometryError {
    none,
    zero_dimension,
    zero_array_dimension,
    zero_tile_factor,
    overflow,
};

struct GemminiGeometryResult {
    GemminiGeometry geometry{};
    GemminiGeometryError error = GemminiGeometryError::none;

    constexpr bool ok() const { return error == GemminiGeometryError::none; }
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

constexpr GemminiGeometryResult make_gemmini_geometry(GemminiGeometryInput input) {
    if (input.shape.i == 0 || input.shape.j == 0 || input.shape.k == 0) {
        return {{}, GemminiGeometryError::zero_dimension};
    }
    if (input.array_dim == 0) {
        return {{}, GemminiGeometryError::zero_array_dimension};
    }
    if (input.tiles.i == 0 || input.tiles.j == 0 || input.tiles.k == 0) {
        return {{}, GemminiGeometryError::zero_tile_factor};
    }

    size_t tile_i_rows = 0;
    size_t tile_j_rows = 0;
    size_t tile_k_rows = 0;
    if (!geometry_detail::checked_multiply(input.tiles.i, input.array_dim, tile_i_rows) ||
        !geometry_detail::checked_multiply(input.tiles.j, input.array_dim, tile_j_rows) ||
        !geometry_detail::checked_multiply(input.tiles.k, input.array_dim, tile_k_rows)) {
        return {{}, GemminiGeometryError::overflow};
    }

    const GemminiOuterCounts outer{
        geometry_detail::ceil_div(input.shape.i, tile_i_rows),
        geometry_detail::ceil_div(input.shape.j, tile_j_rows),
        geometry_detail::ceil_div(input.shape.k, tile_k_rows),
    };
    size_t ij_calls = 0;
    size_t ws_inner_calls = 0;
    if (!geometry_detail::checked_multiply(outer.i, outer.j, ij_calls) ||
        !geometry_detail::checked_multiply(ij_calls, outer.k, ws_inner_calls)) {
        return {{}, GemminiGeometryError::overflow};
    }

    const size_t remainder = input.shape.i % tile_i_rows;
    return {{input.shape, input.tiles, outer, tile_i_rows, outer.i,
             remainder == 0 ? tile_i_rows : remainder, ws_inner_calls},
            GemminiGeometryError::none};
}

} // namespace ggml::gemmini
