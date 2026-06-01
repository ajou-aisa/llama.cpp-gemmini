#include "scale_dump_dispatch.hpp"

#include "scale_dump_internal.hpp"

#ifndef LOG_DUMP_SCALE
#ifdef LOG_DUMP
#define LOG_DUMP_SCALE LOG_DUMP
#else
#define LOG_DUMP_SCALE 0
#endif
#endif

namespace ggml::gemmini::log::scale {
#if LOG_DUMP_SCALE
    namespace
    {
        class ScaleDumpLog : public Log
        {
        public:
            using Log::Log;

            DumpResult write_to(
                LogTarget target,
                const DumpMeta &meta,
                const ScaleTableView &view,
                const GroupDumpConfig &cfg)
            {
                bool owns = false;
                FILE *out = select_output(target.path, &owns);
                DumpResult result = write(out, meta, view, cfg);
                if (owns && out)
                {
                    std::fclose(out);
                }
                return result;
            }

            DumpResult write(
                FILE *out,
                const DumpMeta &meta,
                const ScaleTableView &view,
                const GroupDumpConfig &cfg)
            {
#if GGML_GEMMINI_SCALE_DUMP_GROUP_MODE == GGML_GEMMINI_SCALE_DUMP_GROUP_MODE_BLOCK
                return detail::dump_block(out, meta, view, cfg);
#elif GGML_GEMMINI_SCALE_DUMP_GROUP_MODE == GGML_GEMMINI_SCALE_DUMP_GROUP_MODE_TILE
                return detail::dump_tile(out, meta, view, cfg);
#elif GGML_GEMMINI_SCALE_DUMP_GROUP_MODE == GGML_GEMMINI_SCALE_DUMP_GROUP_MODE_TENSOR
                return detail::dump_tensor(out, meta, view, cfg);
#else
                if (cfg.enable_block)
                {
                    return detail::dump_block(out, meta, view, cfg.block);
                }
                if (cfg.enable_tile)
                {
                    return detail::dump_tile(out, meta, view, cfg.tile);
                }
                if (cfg.enable_tensor)
                {
                    return detail::dump_tensor(out, meta, view, cfg.tensor);
                }
                return DumpResult{};
#endif
            }
        };

        ScaleDumpLog g_dump_scale;

    } // namespace
#endif

    DumpResult dump_scale_groups(
        LogTarget target,
        const DumpMeta &meta,
        const ScaleTableView &view,
        const GroupDumpConfig &cfg)
    {
#if LOG_DUMP_SCALE
        return g_dump_scale.write_to(target, meta, view, cfg);
#else
        (void)target;
        (void)meta;
        (void)view;
        (void)cfg;
        return DumpResult{};
#endif
    }

} // namespace ggml::gemmini::log::scale
