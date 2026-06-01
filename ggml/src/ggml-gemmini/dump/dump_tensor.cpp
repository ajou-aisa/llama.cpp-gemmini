#include "dump_common.hpp"
#include "dump_tensor.hpp"
#include <ggml.h>

#if LOG_DUMP
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdio>

namespace ggml { namespace gemmini { namespace log
{
    namespace
    {
        std::atomic<uint64_t> g_eval_id{0};
        std::atomic<uint64_t> g_step_id{1};
        std::atomic<uint64_t> g_seq_in_eval{0};
        std::atomic<int32_t> g_node_idx{-1};
        std::atomic<uint8_t> g_phase{static_cast<uint8_t>(DumpPhase::unknown)};
        std::atomic<uint32_t> g_graph_max_I{0};

        using LoadFn = float (*)(const char *ptr);

        float load_f32(const char *ptr)
        {
            return *reinterpret_cast<const float *>(ptr);
        }

        float load_f16(const char *ptr)
        {
            return ggml_fp16_to_fp32(*reinterpret_cast<const ggml_fp16_t *>(ptr));
        }

        float load_bf16(const char *ptr)
        {
            return ggml_bf16_to_fp32(*reinterpret_cast<const ggml_bf16_t *>(ptr));
        }

        float load_f64(const char *ptr)
        {
            return static_cast<float>(*reinterpret_cast<const double *>(ptr));
        }

        float load_i8(const char *ptr)
        {
            return static_cast<float>(*reinterpret_cast<const int8_t *>(ptr));
        }

        float load_i16(const char *ptr)
        {
            return static_cast<float>(*reinterpret_cast<const int16_t *>(ptr));
        }

        float load_i32(const char *ptr)
        {
            return static_cast<float>(*reinterpret_cast<const int32_t *>(ptr));
        }

        float load_i64(const char *ptr)
        {
            return static_cast<float>(*reinterpret_cast<const int64_t *>(ptr));
        }

        LoadFn select_loader(enum ggml_type type)
        {
            switch (type)
            {
            case GGML_TYPE_F32:
                return load_f32;
            case GGML_TYPE_F16:
                return load_f16;
            case GGML_TYPE_BF16:
                return load_bf16;
            case GGML_TYPE_F64:
                return load_f64;
            case GGML_TYPE_I8:
                return load_i8;
            case GGML_TYPE_I16:
                return load_i16;
            case GGML_TYPE_I32:
                return load_i32;
            case GGML_TYPE_I64:
                return load_i64;
            default:
                return nullptr;
            }
        }

        void write_tensor_dump(FILE *out, const char *layer, const ggml_tensor *tensor)
        {
            if (!out || !tensor)
            {
                return;
            }

            const char *tensor_name = tensor->name;
            if (!tensor_name || tensor_name[0] == '\0')
            {
                tensor_name = "";
            }

            const int64_t n0 = tensor->ne[0];
            const int64_t n1 = tensor->ne[1] > 0 ? tensor->ne[1] : 1;
            const int64_t n2 = tensor->ne[2] > 0 ? tensor->ne[2] : 1;
            const int64_t n3 = tensor->ne[3] > 0 ? tensor->ne[3] : 1;

            const int64_t rows = n1 * n2 * n3;
            const int64_t cols = n0;

            DumpContext ctx = dump_get_context();
            const int64_t out_I = rows;
            const int64_t out_J = static_cast<int64_t>(ctx.node_idx);
            const int64_t out_K = cols;
            const uint64_t step_id_out = ctx.step_id;

            const char *base = reinterpret_cast<const char *>(tensor->view_src ? tensor->view_src->data : tensor->data);
            const size_t offs = tensor->view_src ? tensor->view_offs : 0;
            if (!base)
            {
                return;
            }

            const size_t nb0 = tensor->nb[0];
            const size_t nb1 = tensor->nb[1];
            const size_t nb2 = tensor->nb[2];
            const size_t nb3 = tensor->nb[3];

            LoadFn load = select_loader(tensor->type);
            if (!load)
            {
                return;
            }

            std::fwrite("{\"layer\":\"", 1, 10, out);
            dump_detail::write_json_escaped(out, layer ? layer : "");
            std::fwrite("\",\"tensor\":\"", 1, 12, out);
            dump_detail::write_json_escaped(out, tensor_name);
            std::fwrite("\",\"phase\":\"", 1, 11, out);
            const char *phase_str = dump_detail::dump_phase_to_string(ctx.phase);
            std::fwrite(phase_str, 1, std::strlen(phase_str), out);
            std::fwrite("\"", 1, 1, out);

            {
                char buf[96];
                const int len = std::snprintf(buf,
                                              sizeof(buf),
                                              ",\"step_id\":%llu",
                                              static_cast<unsigned long long>(step_id_out));
                if (len > 0)
                {
                    std::fwrite(buf, 1, static_cast<size_t>(len), out);
                }
            }

            {
                char buf[128];
                const int len = std::snprintf(buf,
                                              sizeof(buf),
                                              ",\"I\":%lld,\"J\":%lld,\"K\":%lld",
                                              static_cast<long long>(out_I),
                                              static_cast<long long>(out_J),
                                              static_cast<long long>(out_K));
                if (len > 0)
                {
                    std::fwrite(buf, 1, static_cast<size_t>(len), out);
                }
            }

            std::fwrite(",\"data\":[", 1, 9, out);

            bool first_row = true;
            for (int64_t i3 = 0; i3 < n3; ++i3)
            {
                for (int64_t i2 = 0; i2 < n2; ++i2)
                {
                    for (int64_t i1 = 0; i1 < n1; ++i1)
                    {
                        if (!first_row)
                        {
                            std::fputc(',', out);
                        }
                        first_row = false;
                        std::fputc('[', out);

                        const char *row_ptr = base + offs + i1 * nb1 + i2 * nb2 + i3 * nb3;
                        for (int64_t j = 0; j < n0; ++j)
                        {
                            if (j > 0)
                            {
                                std::fputc(',', out);
                            }
                            const char *elem_ptr = row_ptr + j * nb0;
                            const float v = load(elem_ptr);
                            char buf[32];
                            const int len = std::snprintf(buf, sizeof(buf), "%.9g", static_cast<double>(v));
                            if (len > 0)
                            {
                                std::fwrite(buf, 1, static_cast<size_t>(len), out);
                            }
                        }

                        std::fputc(']', out);
                    }
                }
            }

            std::fwrite("]}\n", 1, 3, out);
            std::fflush(out);
        }
    } // namespace

    DumpTensorLog dump_tensor;

    void dump_begin_graph(DumpPhase phase, uint64_t step_id, uint32_t graph_max_I)
    {
        g_phase.store(static_cast<uint8_t>(phase), std::memory_order_relaxed);
        g_step_id.store(step_id, std::memory_order_relaxed);
        g_graph_max_I.store(graph_max_I, std::memory_order_relaxed);
        g_eval_id.fetch_add(1, std::memory_order_relaxed);
        g_seq_in_eval.store(0, std::memory_order_relaxed);
        g_node_idx.store(-1, std::memory_order_relaxed);
    }

    void dump_set_node_idx(int32_t node_idx)
    {
        g_node_idx.store(node_idx, std::memory_order_relaxed);
    }

    DumpContext dump_get_context()
    {
        DumpContext ctx{};
        ctx.eval_id = g_eval_id.load(std::memory_order_relaxed);
        ctx.step_id = g_step_id.load(std::memory_order_relaxed);
        ctx.seq_in_eval = g_seq_in_eval.load(std::memory_order_relaxed);
        ctx.node_idx = g_node_idx.load(std::memory_order_relaxed);
        ctx.phase = static_cast<DumpPhase>(g_phase.load(std::memory_order_relaxed));
        ctx.graph_max_I = g_graph_max_I.load(std::memory_order_relaxed);
        return ctx;
    }

    void DumpTensorLog::operator()(LogTarget target, const char *layer, const ggml_tensor *tensor)
    {
        bool owns = false;
        FILE *out = select_output(target.path, &owns);
        write_tensor_dump(out, layer, tensor);
        if (owns && out)
        {
            std::fclose(out);
        }
    }

    void DumpTensorLog::operator()(const char *layer, const ggml_tensor *tensor)
    {
        write_tensor_dump(out_, layer, tensor);
    }

} // namespace log
} // namespace gemmini
} // namespace ggml

#else

namespace ggml { namespace gemmini { namespace log
{
    DumpTensorLog dump_tensor;
} // namespace log
} // namespace gemmini
} // namespace ggml

#endif
