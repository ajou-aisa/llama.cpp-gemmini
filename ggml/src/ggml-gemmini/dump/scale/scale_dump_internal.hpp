#pragma once

#include "scale_dump_types.hpp"

#include <cstdio>

namespace ggml::gemmini::log::scale::detail
{
    DumpResult dump_block(
        FILE *out,
        const DumpMeta &meta,
        const ScaleTableView &view,
        const config::Block &cfg);

    DumpResult dump_tile(
        FILE *out,
        const DumpMeta &meta,
        const ScaleTableView &view,
        const config::Tile &cfg);

    DumpResult dump_tensor(
        FILE *out,
        const DumpMeta &meta,
        const ScaleTableView &view,
        const config::Tensor &cfg);

} // namespace ggml::gemmini::log::scale::detail
