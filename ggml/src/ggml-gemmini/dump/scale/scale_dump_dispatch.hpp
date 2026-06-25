#pragma once

#include "scale_dump_group_select.hpp"
#include "scale_dump_types.hpp"
#include <gemmini/log.hpp>

#ifndef LOG_DUMP_SCALE
#ifdef LOG_DUMP
#define LOG_DUMP_SCALE LOG_DUMP
#else
#define LOG_DUMP_SCALE 0
#endif
#endif

namespace ggml::gemmini::log::scale
{
#if GGML_GEMMINI_SCALE_DUMP_GROUP_MODE == GGML_GEMMINI_SCALE_DUMP_GROUP_MODE_BLOCK
    using GroupDumpConfig = config::Block;
#elif GGML_GEMMINI_SCALE_DUMP_GROUP_MODE == GGML_GEMMINI_SCALE_DUMP_GROUP_MODE_TILE
    using GroupDumpConfig = config::Tile;
#elif GGML_GEMMINI_SCALE_DUMP_GROUP_MODE == GGML_GEMMINI_SCALE_DUMP_GROUP_MODE_TENSOR
    using GroupDumpConfig = config::Tensor;
#else
    using GroupDumpConfig = config::Auto;
#endif

    DumpResult dump_scale_groups(
        LogTarget target,
        const DumpMeta &meta,
        const ScaleTableView &view,
        const GroupDumpConfig &cfg);
} // namespace ggml::gemmini::log::scale
