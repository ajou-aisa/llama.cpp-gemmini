// ggml-gemmini.cpp

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cctype>
#include <vector>
#include <string>
#include <utility>
#include <future>
#include <vector>
#include <limits>
#include <map>
#include <memory>
#include <type_traits>

#include "ggml-impl.h"
#include "ggml-gemmini.h"
#include "ggml-backend-impl.h"
#include "ggml-gemmini-ethos-config.hpp"

#include <orca/log.hpp>
#include <orca/ggml/log_dump.hpp>
#include <orca/ggml/ggml_orca.hpp>
#include <orca/ggml/dec_ggml.hpp>
#include <orca/types/layer.hpp>

#include "include/gemmini.h"
// Legacy aisa::ActivationDEC is replaced by orca::ggml::ggml_gemmini_activation_dec
// #include "error_compensation/activation_DEC.h"
#include "ggml-gemmini-args.h"
//#include "quantization/ggml-gemmini-quantize.h"

#if LOG_DUMP
#include <atomic>
#include <orca/ggml/log_dump.hpp>
#endif

#ifndef TRANSPOSE_B
#define TRANSPOSE_B 1
#endif
#ifndef FULL_C
#define FULL_C 1
#endif
#ifndef LOW_D
#define LOW_D 0
#endif
#ifndef ERROR_COMPENSATION
#define ERROR_COMPENSATION 0
#endif 
#ifndef OPTION
#define OPTION CPU
#endif

struct ggml_backend_gemmini_context
{
    int n_threads = GGML_DEFAULT_N_THREADS;
    std::unique_ptr<char[]> work_data;
    size_t work_size = 0;
#ifndef GGML_USE_OPENMP
    std::vector<std::future<void>> tasks;
#endif
    std::map<const block_q8_0 *, ggml_gemmini_args_t::unpacked_weight> weight_cache; // packed Q8_0 per weight base pointer
    std::string model_arch;
    ggml_gemmini_ethos_config_registry ethos_config;
    
};

using namespace aisa;

// Cycle 측정 용
extern "C" volatile uint64_t gemmini_tiled_matmul_cycles = 0; // gemmini.h

uint64_t start, end; // 일반 사이클 측정

static void ggml_backend_gemmini_mul_mat(ggml_backend_gemmini_context *ctx,
                                         struct ggml_tensor *dst) // FP32 output (I×J)
{
    orca::log::cycle.set_output_path("log/cycle-log.jsonl");
    orca::log::debug.set_output_path("log/debug-log.jsonl");
    const auto layer_type = orca::types::parse_layer(dst->src[1]->name);
    const char *layer = orca::types::to_string(layer_type);
    orca::log::debug(layer, "ggml_backend_gemmini_mul_mat called");

    /* ____________________________________ 0. 원본 FP32 입력 텐서 ____________________________________________ */
    const auto *src0 = dst->src[0]; // src0: weight (J x K), row-major, 전치 상태
    const auto *src1 = dst->src[1]; // src1: activation (I x K) -> 전치 없음 (A)

    /* _______________________ 2. Gemmini용 dimension _____________________ */
    const size_t I = dst->ne[1]; // I = A.ne[1], K = A.ne[0]
    const size_t J = dst->ne[0]; // J = B.ne[1], K = B.ne[0]
    const size_t K = src1->ne[0]; // K = A.ne[0]

#if LOG_DUMP
    orca::log::dump(orca::log::file("log/tensor_data/act.jsonl"), layer, src1);
#endif
    
    orca::log::debug(layer, "I=%zu, J=%zu, K=%zu", I, J, K);

    if (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16) {
            std::vector<float> src0_f32;
            const float *src0_f = (const float *)src0->data;
            if (src0->type == GGML_TYPE_F16) {
                src0_f32.resize(J * K);
                const ggml_fp16_t *src0_f16 = (const ggml_fp16_t *)src0->data;
                for (size_t j = 0; j < J; j++) {
                    for (size_t k = 0; k < K; k++) {
                        src0_f32[j * K + k] = ggml_fp16_to_fp32(src0_f16[j * K + k]);
                    }
                }
                src0_f = src0_f32.data();
            }
            orca::matmul_cpu_fp(false, true, I, J, K,
                                (const float *)src1->data, src0_f, NULL, (float *)dst->data,
                                K, K, 0, J);
            return;
        }

    ggml_gemmini_args_t args; // DEC과 gemmini 호출을 위한 args 

    // set args
    start = orca::cycle::read();
    args.transpose_B = (TRANSPOSE_B != 0);
    args.layer_type = layer_type;
    args.model_arch = ctx->model_arch.c_str();
    orca::log::debug(layer, "model_arch=%s\n", args.model_arch ? args.model_arch : "");
    args.full_C = FULL_C;
    args.low_D = LOW_D;

    const auto ethos_override = ggml_gemmini_resolve_ethos_override(ctx->ethos_config, ctx->model_arch, layer_type);
    if (ethos_override.has_value())
    {
        args.ethos_override_enabled = true;
        args.ethos_q = ethos_override->q;
        args.ethos_delta = ethos_override->delta;
        args.ethos_l2_enabled = ethos_override->l2_enabled;
        args.ethos_l2_c = ethos_override->l2_c;
        args.ethos_l2_d = ethos_override->l2_d;
    }
    
    args.I = I;
    args.J = J;
    args.K = K;

    /* _____ 3. Gemmini용 stride _____ */
    args.sA = K;
    args.sC = J;

    end = orca::cycle::read();
    orca::log::cycle(layer, "cpu.Set Args for calling gemmini", start, end);

    // set tile size
    start = orca::cycle::read();
    orca::gemmini_set_tile(&args);
    end = orca::cycle::read();
    orca::log::cycle(layer, "cpu.Set tile size", start, end);

    // quantize activation
    start = orca::cycle::read();

    static thread_local std::vector<int8_t> activation_q;
    activation_q.resize(I * K);
    int8_t *qx = activation_q.data();

    args.activation_src = src1;
    orca::ggml::quants::ggml_gemmini_quantize_activation(src1, args, qx);
    // ggml_gemmini_quantize_activation(src1, args, qx);

    args.A = reinterpret_cast<elem_t *>(qx);
    static_assert(sizeof(elem_t) == 1, "Q8_0 path assumes elem_t is int8 (1 byte).");

    end = orca::cycle::read();
    orca::log::cycle(layer, "cpu.Quantize activation", start, end);

    // breackdown weight to int8_t & scale (cache per weight tensor)
    start = orca::cycle::read();
    const int64_t dim_k = src0->ne[0];
    const int64_t dim_j = src0->ne[1] ? src0->ne[1] : 1;
    const int64_t dim_z = src0->ne[2] ? src0->ne[2] : 1;
    const int64_t dim_w = src0->ne[3] ? src0->ne[3] : 1;

    GGML_ASSERT(dim_k % QK8_0 == 0);
    const size_t blocks_K = static_cast<size_t>(dim_k) / QK8_0;
    // Overflow guard for logical_rows and subsequent allocations.
    const __int128 logical_rows_128 =
        static_cast<__int128>(dim_j) * static_cast<__int128>(dim_z) * static_cast<__int128>(dim_w);
    if (logical_rows_128 <= 0 ||
        logical_rows_128 > static_cast<__int128>(std::numeric_limits<size_t>::max())) {
        GGML_ASSERT(false);
        return;
    }
    const size_t logical_rows = static_cast<size_t>(logical_rows_128);

    // Weight matrix layout follows TRANSPOSE_B:
    // - true:  JxK row-major (stride = K)
    // - false: KxJ row-major (stride = J_flat)
    args.sB = args.transpose_B ? static_cast<size_t>(dim_k) : logical_rows;

    const size_t logical_panel_J = args.panel_J > 1 ? args.tile_J_elems() : 1;
    const block_q8_0 *block_base = ggml_gemmini_args_block_base(src0);
    ggml_gemmini_args_t::unpacked_weight *cached = nullptr;

    auto it = ctx->weight_cache.find(block_base);
    if (it != ctx->weight_cache.end() &&
        it->second.matches(block_base,
                           dim_k,
                           dim_j,
                           dim_z,
                           dim_w,
                           args.sB,
                           blocks_K,
                           logical_rows,
                           args.transpose_B,
                           logical_panel_J,
                           args.panel_J)) {
        cached = &it->second;
        orca::log::debug(layer,
                         "[Q8_0 cache] hit base=%p K=%lld cols=%zu sB=%zu blocks_K=%zu transpose_B=%d logical_panel_J=%zu panel_J=%zu",
                         (const void *)block_base,
                         static_cast<long long>(dim_k),
                         logical_rows,
                         args.sB,
                         blocks_K,
                         args.transpose_B ? 1 : 0,
                         logical_panel_J,
                         args.panel_J);
    } else {
        if (it == ctx->weight_cache.end()) {
            orca::log::debug(layer,
                             "[Q8_0_R cache] miss base=%p K=%lld cols=%zu blocks_K=%zu logical_panel_J=%zu panel_J=%zu",
                             (const void *)block_base,
                             static_cast<long long>(dim_k),
                             logical_rows,
                             blocks_K,
                             logical_panel_J,
                             args.panel_J);
        } else {
            orca::log::debug(layer,
                             "[Q8_0_R cache] refresh base=%p K=%lld cols=%zu blocks_K=%zu logical_panel_J=%zu panel_J=%zu",
                             (const void *)block_base,
                             static_cast<long long>(dim_k),
                             logical_rows,
                             blocks_K,
                             logical_panel_J,
                             args.panel_J);
        }
        ggml_gemmini_args_t::unpacked_weight entry;
        entry.blocks = block_base;
        entry.dim_k = dim_k;
        entry.dim_j = dim_j;
        entry.dim_z = dim_z;
        entry.dim_w = dim_w;
        entry.logical_cols = logical_rows;
        entry.blocks_K = blocks_K;
        entry.blocks_J = logical_rows;
        entry.blocks_I = logical_rows;
        entry.block_size_k = QK8_0;
        entry.transpose_b = args.transpose_B;
        entry.logical_panel_J = logical_panel_J;

        // Use Q8_0_R wrapper: vectors are resized by the unpack function
        const bool ok = orca::ggml::quants::ggml_gemmini_unpack_q80_r_weight(
            src0,
            args,
            entry.q_qs,
            entry.c_b,
            entry.s_rf,
            entry.R,
            args.panel_J > 1 ? &entry.s_rf_panel : nullptr,
            args.panel_J > 1 ? &entry.R_panel : nullptr
        );
        GGML_ASSERT(ok);
        if (!ok) {
            return;
        }

        entry.blocks = args.B_blocks;
        entry.blocks_K = args.blocks_K;
        entry.blocks_J = args.blocks_J;
        entry.blocks_I = args.blocks_I;
        entry.block_size_k = args.block_size_k;
        entry.stride = args.sB;

        if (args.panel_J > 1) {
            entry.panel_J = args.panel_J;
        }

        if (it == ctx->weight_cache.end()) {
            it = ctx->weight_cache.emplace(block_base, std::move(entry)).first;
        } else {
            it->second = std::move(entry);
        }
        cached = &it->second;
    }

    args.B = reinterpret_cast<elem_t *>(cached->q_qs.data());
    args.B_blocks = cached->blocks;
    args.B_scales = nullptr;
    args.c_b = cached->c_b.data();
    args.s_rf = cached->s_rf.data();
    args.R = cached->R.data();
    args.blocks_per_row = cached->blocks_K;
    args.blocks_K = cached->blocks_K;
    args.blocks_J = cached->blocks_J;
    args.blocks_I = cached->blocks_I;
    args.block_size_k = cached->block_size_k;
    args.sB = static_cast<size_t>(cached->dim_k);
    args.panel_J = cached->panel_J;
    args.s_rf_panel = cached->panel_J > 1 ? cached->s_rf_panel.data() : nullptr;
    args.R_panel = cached->panel_J > 1 ? cached->R_panel.data() : nullptr;
    orca::ggml::ggml_gemmini_prepare_group_meta(args);

    orca::log::debug(layer,
        "[Q8_0_R cache] restore B=%p sB=%zu c_b=%p s_rf=%p R=%p blocks_per_row=%zu logical_panel_J=%zu panel_J=%zu",
        (void *)args.B, args.sB, (void *)args.c_b, (void *)args.s_rf, (void *)args.R,
        args.blocks_per_row, cached->logical_panel_J, args.panel_J);
    // orca::log::debug("[Gemmini addr] layer=%s A=%p B=%p B_blocks=%p B_scales=%p",
    //                  layer, (void *)args.A, (void *)args.B, (const void *)args.B_blocks, (const void *)args.B_scales);

    end = orca::cycle::read();
    orca::log::cycle(layer, "cpu.Breakdown Q8_0_R", start, end);

    start = orca::cycle::read();
    /* ______________________________ 4. bias 텐서 처리 _________________________________ */
    std::vector<int32_t> zero_bias(J, 0);

    const int32_t *bias_data = zero_bias.data();
    const size_t sD = 0;

    args.sD = sD;
    args.D = bias_data;
    args.repeating_bias = true;

    
    // output 버퍼
    static thread_local std::vector<int8_t> c_i8;
    c_i8.resize(I * J);
    
    args.C = c_i8.data();
    args.f_out = static_cast<float*>(dst->data);
    args.col_stride_f_out = dst->nb[0] / sizeof(float);
    args.stride_f_out = dst->nb[1] / sizeof(float);
    args.tiled_matmul_type = OPTION;

    end = orca::cycle::read();
    orca::log::cycle(layer, "cpu.Set Args for calling gemmini", start, end);

    // orca::log::debug("[Gemmini debug] layer=%s A=%p B=%p C=%p D=%p I=%zu J=%zu K=%zu sA=%zu sB=%zu sC=%zu stride_f_out(row)=%zu stride_f_out(col)=%zu nb1=%zu nb0=%zu",
    //                  layer, args.A, args.B, args.C, args.D,
    //                  args.I, args.J, args.K, args.sA, args.sB, args.sC,
    //                  args.stride_f_out, args.col_stride_f_out, dst->nb[1], dst->nb[0]);

    // orca::log::debug("[Gemmini addr] layer=%s f_out=%p stride_f_out=%zu col_stride_f_out=%zu",
    //                  layer, (void *)args.f_out, args.stride_f_out, args.col_stride_f_out);

    /* __ 5. Gemmini 호출 __ */
    args.tiled_matmul_type = CPU;
    orca::tiled_block_matmul_auto(&args);
    // dst에는 gemmini 커널에서 dequantize한 결과가 들어옴 

#if ERROR_COMPENSATION
    orca::ggml::ggml_gemmini_activation_dec(src1, args);
#endif
}

static void ggml_backend_gemmini_add(
                                        ggml_backend_gemmini_context *ctx,
                                        struct ggml_tensor *dst, // FP32 output (I×J)
                                        struct ggml_tensor *bias){

}

static void ggml_backend_gemmini_out_prod(ggml_backend_gemmini_context * ctx, struct ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
}

// backend interface

static const char * ggml_backend_gemmini_get_name(ggml_backend_t backend) {
    return "GEMMINI";

    GGML_UNUSED(backend);
}

static void ggml_backend_gemmini_free(ggml_backend_t backend) {
    ggml_backend_gemmini_context * ctx = (ggml_backend_gemmini_context *)backend->context;
    delete ctx;
    delete backend;
}

static enum ggml_status ggml_backend_gemmini_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    ggml_backend_gemmini_context * ctx = (ggml_backend_gemmini_context *)backend->context;

#if LOG_DUMP
    uint32_t mxI = 0;
    bool decode_start_marker = false;
    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];
        if (node->op == GGML_OP_MUL_MAT) {
            const uint32_t I = node->ne[1] > 0 ? static_cast<uint32_t>(node->ne[1]) : 1u;
            if (I > mxI) {
                mxI = I;
            }
            const struct ggml_tensor * src1 = node->src[1];
            if (!decode_start_marker && src1 && src1->name &&
                std::strcmp(src1->name, "attn_norm-0") == 0 &&
                I == 1 && src1->ne[1] == 1) {
                decode_start_marker = true;
            }
        }
    }

    static std::atomic<bool> g_seen_any_graph{false};
    static std::atomic<bool> g_decode_started{false};
    static std::atomic<uint64_t> g_decode_count{0};
    orca::log::DumpPhase phase = orca::log::DumpPhase::unknown;
    uint64_t step_id = 1;

    const bool first_graph = !g_seen_any_graph.exchange(true, std::memory_order_relaxed);
    if (first_graph) {
        phase = orca::log::DumpPhase::prefill;
        step_id = 1;
        g_decode_started.store(false, std::memory_order_relaxed);
        g_decode_count.store(0, std::memory_order_relaxed);
    } else if (!g_decode_started.load(std::memory_order_relaxed)) {
        if (decode_start_marker) {
            g_decode_started.store(true, std::memory_order_relaxed);
            g_decode_count.store(1, std::memory_order_relaxed);
            phase = orca::log::DumpPhase::decode;
            step_id = 2;
        } else {
            phase = orca::log::DumpPhase::prefill;
            step_id = 1;
        }
    } else {
        if (decode_start_marker) {
            g_decode_count.fetch_add(1, std::memory_order_relaxed);
        }
        phase = orca::log::DumpPhase::decode;
        step_id = 1 + g_decode_count.load(std::memory_order_relaxed);
    }

    orca::log::dump_begin_graph(phase, step_id, mxI);
#endif

    for (int i = 0; i < cgraph->n_nodes; i++)
    {
        struct ggml_tensor *node = cgraph->nodes[i];

        switch (node->op)
        {
        case GGML_OP_MUL_MAT: {
#if LOG_DUMP
            const int32_t node_idx = node->ne[0] > 0 ? static_cast<int32_t>(node->ne[0]) : 1;
            orca::log::dump_set_node_idx(node_idx);
#endif
            ggml_backend_gemmini_mul_mat(ctx, node);
            break;
        }
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_ADD:
#if LOG_DUMP
            orca::log::dump_set_node_idx(-1);
#endif
            break;

        default:
            GGML_ABORT("%s: unsupported op assigned to GEMMINI: %s\n", __func__, ggml_op_desc(node));
        }
    }

    GGML_UNUSED(backend);
    return GGML_STATUS_SUCCESS;
}

static struct ggml_backend_i gemmini_backend_i = {
    /* .get_name                = */ ggml_backend_gemmini_get_name,
    /* .free                    = */ ggml_backend_gemmini_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ NULL,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_gemmini_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
};

static ggml_guid_t ggml_backend_gemmini_guid(void) {
    static ggml_guid guid = { 0x10, 0xa8, 0xae, 0xf4, 0xc0, 0x1e, 0x61, 0x97, 0x8f, 0xeb, 0x33, 0x04, 0xa1, 0x33, 0x51, 0x2d };
    return &guid;
}

ggml_backend_t ggml_backend_gemmini_init(void) {
    ggml_backend_gemmini_context * ctx = new ggml_backend_gemmini_context;

    ggml_backend_t backend = new ggml_backend {
        /* .guid      = */ ggml_backend_gemmini_guid(),
        /* .interface = */ gemmini_backend_i,
        /* .device    = */ ggml_backend_reg_dev_get(ggml_backend_gemmini_reg(), 0),
        /* .context   = */ ctx,
    };

    return backend;
}

// bool ggml_backend_is_gemmini(ggml_backend_t backend) {
//     return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_gemmini_guid());
// }

// device interface

static const char * ggml_backend_gemmini_device_get_name(ggml_backend_dev_t dev) {
    return "GEMMINI";

    GGML_UNUSED(dev);
}

static const char * ggml_backend_gemmini_device_get_description(ggml_backend_dev_t dev) {
    return "GEMMINI";

    GGML_UNUSED(dev);
}

static void ggml_backend_gemmini_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    // TODO
    *free = 0;
    *total = 0;

    GGML_UNUSED(dev);
}

static enum ggml_backend_dev_type ggml_backend_gemmini_device_get_type(ggml_backend_dev_t dev) {
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;

    GGML_UNUSED(dev);
}

static void ggml_backend_gemmini_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_gemmini_device_get_name(dev);
    props->description = ggml_backend_gemmini_device_get_description(dev);
    props->type        = ggml_backend_gemmini_device_get_type(dev);
    ggml_backend_gemmini_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ true,
        /* .buffer_from_host_ptr  = */ true,
        /* .events                = */ false,
    };
}

static ggml_backend_t ggml_backend_gemmini_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    ggml_backend_t backend = ggml_backend_gemmini_init();
    auto * ctx = (ggml_backend_gemmini_context *) backend->context;
    ctx->model_arch = params ? params : "";
    ctx->ethos_config = ggml_gemmini_load_ethos_config_registry();
    return backend;

    GGML_UNUSED(dev);
}

static ggml_backend_buffer_type_t ggml_backend_gemmini_device_get_buffer_type(ggml_backend_dev_t dev) {
    return ggml_backend_cpu_buffer_type();

    GGML_UNUSED(dev);
}

static ggml_backend_buffer_t ggml_backend_gemmini_device_buffer_from_host_ptr(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    return ggml_backend_cpu_buffer_from_ptr(ptr, size);

    GGML_UNUSED(dev);
    GGML_UNUSED(max_tensor_size);
}

static bool ggml_backend_gemmini_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:

        case GGML_OP_TRANSPOSE:
            return true;

        case GGML_OP_MUL_MAT:
        {
            const struct ggml_tensor *a = op->src[0]; // W
            const struct ggml_tensor *b = op->src[1]; // x
            if (!a || !b)
                return false;
            if (op->type != GGML_TYPE_F32)
                return false;
            if (b->type != GGML_TYPE_F32)
                return false;
            if (!ggml_is_contiguous(a) || !ggml_is_contiguous(b))
                return false;
            // Q8_0 weights → Gemmini HW, FP weights → CPU fallback via matmul_cpu_fp
            if (a->type == GGML_TYPE_Q8_0 || a->type == GGML_TYPE_F32 || a->type == GGML_TYPE_F16)
                return true;
            return false;
        }

        case GGML_OP_ADD:
        case GGML_OP_OUT_PROD:
            // return op->src[0]->type == GGML_TYPE_F32 &&
            //        op->src[1]->type == GGML_TYPE_F32 &&
            //        ggml_is_matrix(src0) &&
            //        ggml_is_matrix(src1) &&
            //        ggml_is_contiguous(src0) &&
            //        (ggml_is_contiguous(src1) || ggml_is_transposed(src1)) &&
            //        (src0->type == GGML_TYPE_F32 || ggml_get_type_traits(src0->type)->to_float != NULL);

        default:
            return false;

    }

    GGML_UNUSED(dev);
}

static bool ggml_backend_gemmini_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return ggml_backend_buft_is_host(buft);

    GGML_UNUSED(dev);
}

static const struct ggml_backend_device_i ggml_backend_gemmini_device_i = {
    /* .get_name             = */ ggml_backend_gemmini_device_get_name,
    /* .get_description      = */ ggml_backend_gemmini_device_get_description,
    /* .get_memory           = */ ggml_backend_gemmini_device_get_memory,
    /* .get_type             = */ ggml_backend_gemmini_device_get_type,
    /* .get_props            = */ ggml_backend_gemmini_device_get_props,
    /* .init_backend         = */ ggml_backend_gemmini_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_gemmini_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ ggml_backend_gemmini_device_buffer_from_host_ptr,
    /* .supports_op          = */ ggml_backend_gemmini_device_supports_op,
    /* .supports_buft        = */ ggml_backend_gemmini_device_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

// backend reg interface

static const char * ggml_backend_gemmini_reg_get_name(ggml_backend_reg_t reg) {
    return "GEMMINI";

    GGML_UNUSED(reg);
}

static size_t ggml_backend_gemmini_reg_get_device_count(ggml_backend_reg_t reg) {
    return 1;

    GGML_UNUSED(reg);
}

static ggml_backend_dev_t ggml_backend_gemmini_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);

    static ggml_backend_device ggml_backend_gemmini_device = {
        /* .iface   = */ ggml_backend_gemmini_device_i,
        /* .reg     = */ reg,
        /* .context = */ nullptr,
    };

    return &ggml_backend_gemmini_device;

    GGML_UNUSED(reg);
    GGML_UNUSED(index);
}

static const struct ggml_backend_reg_i ggml_backend_gemmini_reg_i = {
    /* .get_name         = */ ggml_backend_gemmini_reg_get_name,
    /* .get_device_count = */ ggml_backend_gemmini_reg_get_device_count,
    /* .get_device       = */ ggml_backend_gemmini_reg_get_device,
    /* .get_proc_address = */ NULL,
};

ggml_backend_reg_t ggml_backend_gemmini_reg(void) {
    static struct ggml_backend_reg ggml_backend_gemmini_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_gemmini_reg_i,
        /* .context     = */ NULL,
    };

    return &ggml_backend_gemmini_reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_gemmini_reg)
