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

    Stripe(size_t i, size_t j, size_t row_off, size_t col_off)
        : I(i), J(j), row_offset(row_off), col_offset(col_off) {
    }

    bool empty() const { return I == 0 || J == 0; }
};
} // namespace ggml::gemmini::quants
