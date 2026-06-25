#pragma once

#ifndef LOG_DUMP
#define LOG_DUMP 0
#endif

#include <cstdint>
#include <gemmini/log.hpp>

struct ggml_tensor;

namespace ggml
{
namespace gemmini
{
namespace log
{
    enum class DumpPhase : uint8_t
    {
        unknown = 0,
        prefill = 1,
        decode = 2,
    };

#if LOG_DUMP
    struct DumpContext
    {
        uint64_t eval_id;
        uint64_t step_id;
        uint64_t seq_in_eval;
        int32_t node_idx;
        DumpPhase phase;
        uint32_t graph_max_I;
    };

    class DumpTensorLog : public Log
    {
    public:
        using Log::Log;

        void operator()(LogTarget target, const char *layer, const ggml_tensor *tensor);
        void operator()(const char *layer, const ggml_tensor *tensor);
    };

    using DumpLog = DumpTensorLog;

    void dump_begin_graph(DumpPhase phase, uint64_t step_id, uint32_t graph_max_I);
    void dump_set_node_idx(int32_t node_idx);
    DumpContext dump_get_context();

    extern DumpTensorLog dump_tensor;
    inline DumpTensorLog &dump = dump_tensor;
#else
    struct DumpContext
    {
        uint64_t eval_id = 0;
        uint64_t step_id = 0;
        uint64_t seq_in_eval = 0;
        int32_t node_idx = 0;
        DumpPhase phase = DumpPhase::unknown;
        uint32_t graph_max_I = 0;
    };

    class DumpTensorLog : public Log
    {
    public:
        using Log::Log;

        void operator()(LogTarget, const char *, const ggml_tensor *) {}
        void operator()(const char *, const ggml_tensor *) {}
    };

    using DumpLog = DumpTensorLog;

    inline void dump_begin_graph(DumpPhase, uint64_t, uint32_t) {}
    inline void dump_set_node_idx(int32_t) {}
    inline DumpContext dump_get_context() { return DumpContext(); }
    inline void dump_reset() {}

    extern DumpTensorLog dump_tensor;
    inline DumpTensorLog &dump = dump_tensor;
#endif
} // namespace log
} // namespace gemmini
} // namespace ggml
