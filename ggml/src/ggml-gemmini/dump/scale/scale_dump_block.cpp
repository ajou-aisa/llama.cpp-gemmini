#include "scale_dump_internal.hpp"

#include "../dump_common.hpp"

#include <cstdio>

namespace ggml::gemmini::log::scale::detail
{
    namespace
    {
    void write_float(FILE *out, float v)
    {
        char buf[32];
        const int len = std::snprintf(buf, sizeof(buf), "%.9g", static_cast<double>(v));
        if (len > 0)
        {
            std::fwrite(buf, 1, static_cast<size_t>(len), out);
        }
    }

    void write_k_block_hints(FILE *out, const DumpMeta &meta, const ScaleTableView &view)
    {
        std::fputs(",\"k_block_starts\":[", out);
        for (size_t blk = 0; blk < view.cols; ++blk)
        {
            if (blk > 0)
            {
                std::fputc(',', out);
            }
            const size_t start = blk * view.block_size;
            char buf[32];
            const int len = std::snprintf(buf, sizeof(buf), "%zu", start);
            if (len > 0)
            {
                std::fwrite(buf, 1, static_cast<size_t>(len), out);
            }
        }
        std::fputc(']', out);

        std::fputs(",\"k_block_ends\":[", out);
        for (size_t blk = 0; blk < view.cols; ++blk)
        {
            if (blk > 0)
            {
                std::fputc(',', out);
            }
            const size_t end_raw = (blk + 1) * view.block_size;
            size_t end = end_raw;
            if (meta.K > 0)
            {
                const size_t k_max = static_cast<size_t>(meta.K);
                end = end_raw < k_max ? end_raw : k_max;
            }
            char buf[32];
            const int len = std::snprintf(buf, sizeof(buf), "%zu", end);
            if (len > 0)
            {
                std::fwrite(buf, 1, static_cast<size_t>(len), out);
            }
        }
        std::fputc(']', out);
    }

    void write_common_header(FILE *out, const DumpMeta &meta, const ScaleTableView &view)
    {
        std::fputs("{\"layer\":\"", out);
        ggml::gemmini::log::dump_detail::write_json_escaped(out, meta.layer ? meta.layer : "");
        std::fputc('"', out);

        std::fputs(",\"tensor\":\"", out);
        ggml::gemmini::log::dump_detail::write_json_escaped(out, meta.tensor ? meta.tensor : "");
        std::fputc('"', out);

        std::fputs(",\"phase\":\"", out);
        ggml::gemmini::log::dump_detail::write_json_escaped(out, meta.phase ? meta.phase : "unknown");
        std::fputc('"', out);

        char buf[160];
        const int len = std::snprintf(
            buf,
            sizeof(buf),
            ",\"step_id\":%llu,\"I\":%lld,\"J\":%lld,\"K\":%lld",
            static_cast<unsigned long long>(meta.step_id),
            static_cast<long long>(meta.I),
            static_cast<long long>(meta.J),
            static_cast<long long>(meta.K));
        if (len > 0)
        {
            std::fwrite(buf, 1, static_cast<size_t>(len), out);
        }

        std::fputs(",\"row_axis\":\"", out);
        ggml::gemmini::log::dump_detail::write_json_escaped(out, meta.row_axis ? meta.row_axis : "row");
        std::fputc('"', out);

        std::fputs(",\"block_axis\":\"", out);
        ggml::gemmini::log::dump_detail::write_json_escaped(out, meta.block_axis ? meta.block_axis : "K");
        std::fputc('"', out);

        {
            char buf2[192];
            const int len2 = std::snprintf(
                buf2,
                sizeof(buf2),
                ",\"row_count\":%zu,\"block_count\":%zu,\"block_size\":%zu,\"channel_count\":%lld,\"block_index_hint\":\"data[row_idx][k_block_idx]\"",
                view.rows,
                view.cols,
                view.block_size,
                static_cast<long long>(meta.K));
            if (len2 > 0)
            {
                std::fwrite(buf2, 1, static_cast<size_t>(len2), out);
            }
        }

        write_k_block_hints(out, meta, view);
    }

    void write_scale_data_2d(FILE *out, const ScaleTableView &view)
    {
        std::fputs(",\"data\":[", out);
        for (size_t row = 0; row < view.rows; ++row)
        {
            if (row > 0)
            {
                std::fputc(',', out);
            }
            std::fputc('[', out);
            for (size_t col = 0; col < view.cols; ++col)
            {
                if (col > 0)
                {
                    std::fputc(',', out);
                }
                const float scale = view.scales[row * view.cols + col];
                write_float(out, scale);
            }
            std::fputc(']', out);
        }
        std::fputc(']', out);
    }
} // namespace

DumpResult dump_block(
    FILE *out,
    const DumpMeta &meta,
    const ScaleTableView &view,
    const config::Block &cfg)
{
    (void)cfg;
    DumpResult result{};

    if (!out || !view.scales || view.rows == 0 || view.cols == 0 || view.block_size == 0)
    {
        return result;
    }

    write_common_header(out, meta, view);
    write_scale_data_2d(out, view);
    std::fputs("}\n", out);
    std::fflush(out);

    result.success = true;
    result.group_count = view.rows * view.cols;
    result.value_count = result.group_count;
    return result;
    }

} // namespace ggml::gemmini::log::scale::detail
