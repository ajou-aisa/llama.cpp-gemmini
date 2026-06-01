#pragma once

#include <cstddef>
#include <cstdint>

namespace ggml::gemmini::log::scale
{
struct DumpMeta
{
    const char *layer = "";
    const char *tensor = "";
    const char *phase = "unknown";
    const char *row_axis = "row";
    const char *block_axis = "K";
    uint64_t step_id = 0;

    int64_t I = 0;
    int64_t J = 0;
    int64_t K = 0;
};

struct ScaleTableView
{
    const float *scales = nullptr; // [rows][cols] row-major
    size_t rows = 0;
    size_t cols = 0;
    size_t block_size = 32;
};

namespace config
{
    struct Block
    {
        bool emit_values = true;
        bool emit_stats = false;
    };

    struct Tile
    {
        size_t tile_rows = 0;
        size_t tile_cols = 0;
        bool emit_values = false;
        bool emit_stats = true;
    };

    struct Tensor
    {
        bool emit_values = true;
        bool emit_stats = true;
    };

    struct Auto
    {
        bool enable_block = true;
        bool enable_tile = false;
        bool enable_tensor = false;

        Block block{};
        Tile tile{};
        Tensor tensor{};
    };
} // namespace config

struct DumpResult
{
    bool success = false;
    size_t group_count = 0;
    size_t value_count = 0;
};
} // namespace ggml::gemmini::log::scale
