#include "scale_dump_internal.hpp"

#include <gemmini/log.hpp>

namespace ggml::gemmini::log::scale::detail
{
    DumpResult dump_tile(
        FILE *out,
        const DumpMeta &meta,
        const ScaleTableView &view,
        const config::Tile &cfg)
    {
        (void)out;
        (void)meta;
        (void)view;
        (void)cfg;
        DumpResult result{};
        return result;
    }

    DumpResult dump_tensor(
        FILE *out,
        const DumpMeta &meta,
        const ScaleTableView &view,
        const config::Tensor &cfg)
    {
        (void)out;
        (void)meta;
        (void)view;
        (void)cfg;
        DumpResult result{};
        return result;
    }

} // namespace ggml::gemmini::log::scale::detail
