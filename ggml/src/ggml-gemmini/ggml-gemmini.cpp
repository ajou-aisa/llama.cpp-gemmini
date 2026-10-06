#include <gemmini/trace-context.hpp>
// ggml-gemmini.cpp

#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <new>
#include <cstdlib>

#include <limits>
#include <stdexcept>

#include <atomic>
#include <memory>
#include <mutex>
#include <set>
#include <tuple>
#include "ggml-impl.h"
#include "ggml-gemmini.h"
#include "ggml-gemmini-config.hpp"
#include "ggml-gemmini-buffer.hpp"
#include "ggml-backend-impl.h"
#include "ggml-quants.h"

#include <gemmini/log.hpp>
#include <gemmini/optrace.hpp>
#include <gemmini/performance.hpp>
#if LOG_CYCLE || CYCLE_SIM
#include <gemmini/semantic.hpp>
#endif
#if LOG_CYCLE
#include "../../../common/json.hpp"
#endif
#include "dump/dump_tensor.hpp"

#include <gemmini_params.h>
#include <gemmini.h>
#include "ggml-gemmini-args.h"
#include "ggml-gemmini-matmul.hpp"
#if defined(GGML_GEMMINI_EXECUTION_BACKEND_IM2P_SIM)
#include "ggml-gemmini-im2p.hpp"
#endif
#include "quants/act/quantize.hpp"
// #include "quantization/ggml-gemmini-quantize.h"

#if LOG_DUMP
#include <atomic>
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
#ifndef OPTION
#define OPTION CPU
#endif

#include "ops.hpp"

// Cycle 측정 용
extern "C" volatile uint64_t gemmini_tiled_matmul_cycles = 0; // gemmini.h

void setup_gemmini_log_outputs_if_needed(void) {
    const auto result = ggml::gemmini::log::setup_default_outputs();
    if (!result.cycle) {
        GGML_LOG_WARN("%s: failed to set default cycle log path '%s'\n",
                      __func__,
                      GEMMINI_LOG_DEFAULT_CYCLE_PATH);
    }
    if (!result.debug) {
        GGML_LOG_WARN("%s: failed to set default debug log path '%s'\n",
                      __func__,
                      GEMMINI_LOG_DEFAULT_DEBUG_PATH);
    }
#if defined(GGML_GEMMINI_EXECUTION_BACKEND_IM2P_SIM)
    ggml::gemmini::im2p_adapter::install_rtl_debug_sink();
#endif
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

static enum ggml_status ggml_backend_gemmini_graph_compute(ggml_backend_t       backend,
                                                           struct ggml_cgraph * cgraph) {
    ggml_backend_gemmini_context * ctx = (ggml_backend_gemmini_context *)backend->context;
    setup_gemmini_log_outputs_if_needed();

    size_t graph_mul_mat_count = 0;
    size_t graph_max_i         = 0;
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const struct ggml_tensor * node = cgraph->nodes[i];
        if (node->op == GGML_OP_MUL_MAT) {
            ++graph_mul_mat_count;
            graph_max_i =
                std::max(graph_max_i, static_cast<size_t>(node->ne[1] > 0 ? node->ne[1] : 1));
        }
    }
    ggml::gemmini::log::debug("graph",
                              "nodes=%d mul_mat=%zu max_i=%zu",
                              cgraph->n_nodes,
                              graph_mul_mat_count,
                              graph_max_i);

#if LOG_DUMP
    uint32_t mxI = 0;
    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];
        if (node->op == GGML_OP_MUL_MAT) {
            const uint32_t I = node->ne[1] > 0 ? static_cast<uint32_t>(node->ne[1]) : 1u;
            if (I > mxI)
                mxI = I;
        }
    }

    ggml::gemmini::log::DumpPhase phase   = ggml::gemmini::log::DumpPhase::unknown;
    uint64_t                      step_id = 0;
    const auto                    context = ggml::gemmini::performance::capture_context();
    if (context.request_id && context.operation_id) {
        const bool decode = context.phase == ggml::gemmini::performance::Phase::decode;
        phase =
            decode ? ggml::gemmini::log::DumpPhase::decode : ggml::gemmini::log::DumpPhase::prefill;
        step_id = decode ? context.decode_ordinal + 1 : 1;
    }

    ggml::gemmini::log::dump_begin_graph(phase, step_id, mxI);
#endif

    const auto     graph_trace = gemmini_trace_capture();
    const uint64_t graph_id = gemmini_trace_reserve_ids(static_cast<uint64_t>(cgraph->n_nodes) + 1);
    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];
#if LOG_CYCLE || CYCLE_SIM
        ggml::gemmini::semantic::ScopedNode semantic_scope(node, 1);
#endif
        ggml::gemmini::trace::ScopedContext operator_context(
            gemmini_trace_operator(graph_trace,
                                   graph_id,
                                   graph_id + 1 + static_cast<uint64_t>(i),
                                   static_cast<uint64_t>(i),
                                   ggml_op_name(node->op),
                                   0,
                                   1,
                                   1));
        ggml::gemmini::trace::CpuStage operator_dispatch(
            node->name, "operator.host_dispatch", ggml::gemmini::trace::CpuStage::Scope::envelope);

        switch (node->op) {
        case GGML_OP_GET_ROWS:
            ggml_backend_gemmini_get_rows_q8_channel(node->src[0], node->src[1], node);
            break;
        case GGML_OP_MUL_MAT: {
#if LOG_DUMP
            const int32_t node_idx = node->ne[0] > 0 ? static_cast<int32_t>(node->ne[0]) : 1;
            ggml::gemmini::log::dump_set_node_idx(node_idx);
#endif
#if CYCLE_SIM
            const auto cycle_sim_context = ggml::gemmini::cycle_sim::context_for(node);
            ggml::gemmini::cycle_sim::ScopedContext cycle_sim_scope(cycle_sim_context);
#endif
            ggml_backend_gemmini_mul_mat(ctx, node);
#if CYCLE_SIM
            if (cycle_sim_context)
                cycle_sim_context.session->ensure_healthy();
#endif
#if defined(GGML_GEMMINI_EXECUTION_BACKEND_IM2P_SIM) && defined(GGML_GEMMINI_TESTING)
            if (ggml::gemmini::im2p_adapter::test_production_failed()) {
                return GGML_STATUS_FAILED;
            }
#endif
            break;
        }
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_ADD:
#if LOG_DUMP
            ggml::gemmini::log::dump_set_node_idx(-1);
#endif
            break;

        default:
            GGML_ABORT(
                "%s: unsupported op assigned to GEMMINI: %s\n", __func__, ggml_op_desc(node));
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
    static ggml_guid guid = {0x10,
                             0xa8,
                             0xae,
                             0xf4,
                             0xc0,
                             0x1e,
                             0x61,
                             0x97,
                             0x8f,
                             0xeb,
                             0x33,
                             0x04,
                             0xa1,
                             0x33,
                             0x51,
                             0x2d};
    return &guid;
}

ggml_backend_t ggml_backend_gemmini_init(void) {
    ggml_backend_gemmini_context * ctx = new ggml_backend_gemmini_context;
#if defined(GGML_GEMMINI_EXECUTION_BACKEND_IM2P_SIM)
    ggml::gemmini::log::debug("backend", "compiled_backend=IM2P_SIM transport=Verilator");
#else
    ggml::gemmini::log::debug("backend", "compiled_backend=HARDWARE");
#endif
    ggml_backend_t backend = new ggml_backend{
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

static void
ggml_backend_gemmini_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    // TODO
    *free  = 0;
    *total = 0;

    GGML_UNUSED(dev);
}

static enum ggml_backend_dev_type ggml_backend_gemmini_device_get_type(ggml_backend_dev_t dev) {
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;

    GGML_UNUSED(dev);
}

static void ggml_backend_gemmini_device_get_props(ggml_backend_dev_t              dev,
                                                  struct ggml_backend_dev_props * props) {
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

static ggml_backend_t ggml_backend_gemmini_device_init_backend(ggml_backend_dev_t dev,
                                                               const char *       params) {
    ggml_backend_t backend = ggml_backend_gemmini_init();
    auto *         ctx     = (ggml_backend_gemmini_context *)backend->context;
    ctx->model_arch        = params ? params : "";
    return backend;

    GGML_UNUSED(dev);
}

static ggml_backend_buffer_type_t
ggml_backend_gemmini_device_get_buffer_type(ggml_backend_dev_t dev) {
    return ggml::gemmini::gemmini_buffer_type(dev);

    GGML_UNUSED(dev);
}

static ggml_backend_buffer_t ggml_backend_gemmini_device_buffer_from_host_ptr(
    ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    return ggml::gemmini::gemmini_buffer_from_host_ptr(dev, ptr, size);

    GGML_UNUSED(dev);
    GGML_UNUSED(max_tensor_size);
}

static bool ggml_backend_gemmini_device_supports_buft(ggml_backend_dev_t         dev,
                                                      ggml_backend_buffer_type_t buft) {
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

static ggml_backend_dev_t ggml_backend_gemmini_reg_get_device(ggml_backend_reg_t reg,
                                                              size_t             index) {
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
    /* .get_proc_address = */ nullptr,
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
