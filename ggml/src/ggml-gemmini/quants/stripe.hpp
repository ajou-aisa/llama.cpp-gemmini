// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Orca Contributors
//
// Stripe struct for GEMM tiling geometry - Pure algorithm layer (ggml-free)

#pragma once

#include <cstddef>

namespace ggml::gemmini::quants
{
// Tile stripe geometry for 2D GEMM tiling.
// Geometry only - no data pointers, no strides.
struct Stripe
{
    size_t I = 0;          // number of rows in this stripe
    size_t J = 0;          // number of columns in this stripe
    size_t row_offset = 0; // starting row offset in larger tensor
    size_t col_offset = 0; // starting column offset in larger tensor

    // Default constructor creates empty stripe
    Stripe() = default;

    // Direct initializer
    Stripe(size_t i, size_t j, size_t row_off, size_t col_off)
        : I(i), J(j), row_offset(row_off), col_offset(col_off)
    {
    }

    // Helper: stripe extent in dimension
    size_t extent_i() const { return I; }
    size_t extent_j() const { return J; }

    // Helper: stripe end position (one-past-last element)
    size_t row_end() const { return row_offset + I; }
    size_t col_end() const { return col_offset + J; }

    // Helper: is stripe empty?
    bool empty() const { return I == 0 || J == 0; }

    // Helper: is this an edge/partial stripe?
    bool is_edge() const { return false; } // reserved for edge detection

    // Helper: clamp extent for edge stripes at tensor boundary
    static size_t clamp_extent(size_t offset, size_t extent, size_t tile_size)
    {
        size_t remaining = extent - offset;
        return remaining < tile_size ? remaining : tile_size;
    }

    // Helper: ceiling division for stripe count
    static size_t ceil_div(size_t total, size_t tile_size)
    {
        return (total + tile_size - 1) / tile_size;
    }

    // Number of stripes needed to cover extent
    static size_t stripes_count(size_t total, size_t tile_size)
    {
        return ceil_div(total, tile_size);
    }

    // Create stripe from tile index (row-major ordering)
    //
    // tile_idx: zero-based tile index
    // extent_i, extent_j: total tensor dimensions
    // tile_i, tile_j: hardware tile dimensions
    //
    // Returns stripe with clamped extent for edge tiles.
    static Stripe from_index(
        size_t tile_idx,
        size_t extent_i,
        size_t extent_j,
        size_t tile_i,
        size_t tile_j)
    {
        const size_t num_tiles_j = stripes_count(extent_j, tile_j);
        const size_t tile_row = tile_idx / num_tiles_j;
        const size_t tile_col = tile_idx % num_tiles_j;

        const size_t row_off = tile_row * tile_i;
        const size_t col_off = tile_col * tile_j;

        const size_t i = clamp_extent(row_off, extent_i, tile_i);
        const size_t j = clamp_extent(col_off, extent_j, tile_j);

        return Stripe(i, j, row_off, col_off);
    }
};
} // namespace ggml::gemmini::quants
