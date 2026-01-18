// ggml-gemmini.cpp

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cctype>
#include <vector>
#include <string>
#include <utility>

#include "ggml-gemmini-util.h"
#include "include/gemmini.h"
#include "error_compensation/activation_DEC.h"
#include "ggml-gemmini-args.h"
#include "quantization/ggml-gemmini-quantize.h"

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
#ifndef CYCLE_LOG
#define CYCLE_LOG 1
#endif
#ifndef GEMMINI_GOLDEN_CHECK
#define GEMMINI_GOLDEN_CHECK 0
#endif
#ifndef GEMMINI_GOLDEN_MM
#define GEMMINI_GOLDEN_MM 0
#endif
#ifndef OPTION
#define OPTION CPU
#endif

using namespace aisa;

// Cycle 측정 용
extern "C" volatile uint64_t gemmini_tiled_matmul_cycles = 0; // gemmini.h

uint64_t start, end; // 일반 사이클 측정

static void ggml_backend_gemmini_mul_mat(ggml_backend_gemmini_context *ctx,
                                         struct ggml_tensor *dst) // FP32 output (I×J)
{
    DBG_DETAIL("[Gemmini] mul_mat call\n");

    /* ____________________________________ 0. 원본 FP32 입력 텐서 ____________________________________________ */
    const auto *src0 = dst->src[0]; // src0: weight (K x J) -> 전치
    const auto *src1 = dst->src[1]; // src1: activation (I x K) -> 전치 없음 (A)

    ggml_gemmini_args_t args; // DEC과 gemmini 호출을 위한 args 
    // set args
    start = read_cycles();

    const char *w_name = (src0 && src0->name[0]) ? src0->name : ""; // weight name
    const char *layer = labelFromWeight(w_name); // layer 이름 추출
    (void)TRANSPOSE_B; // 항상 (K x J) row-major 정책 사용

    args.transpose_B = false;
    args.layer_name = layer;
    args.full_C = FULL_C;
    args.low_D = LOW_D;
    
    /* _______________________ 2. Gemmini용 dimension _____________________ */
    const size_t I = dst->ne[1]; // I = A.ne[1], K = A.ne[0]
    const size_t J = dst->ne[0]; // K = B.ne[0], J = B.ne[1] (transpose)
    const size_t K = src1->ne[0]; // K = A.ne[0]
    DBG_DETAIL("I=%zu, J=%zu, K=%zu\n", I, J, K);

    args.I = I;
    args.J = J;
    args.K = K;
    
    /* _____ 3. Gemmini용 stride _____ */
    args.sA = K;
    args.sC = J;

    end = read_cycles();
    PRINT_CYCLE(layer, "Set Args for calling gemmini", start, end, end - start);


    // quantize activation
    start = read_cycles();

    static thread_local std::vector<int8_t> activation_q;
    activation_q.resize(I * K);
    int8_t *qx = activation_q.data();

    ggml_gemmini_quantize_activation(src1, args, qx);

    args.A = reinterpret_cast<elem_t *>(qx);

    end = read_cycles();
    PRINT_CYCLE(layer, "Quantize activation", start, end, end - start);
    
    // breackdown weight to int8_t & scale (cache per weight tensor)
    start = read_cycles();
    const int64_t dim_k = src0->ne[0];
    const int64_t dim_j = src0->ne[1] ? src0->ne[1] : 1;
    const int64_t dim_z = src0->ne[2] ? src0->ne[2] : 1;
    const int64_t dim_w = src0->ne[3] ? src0->ne[3] : 1;

    GGML_ASSERT(dim_k % QK8_0 == 0);
    const size_t blocks_K = static_cast<size_t>(dim_k) / QK8_0;
    const size_t logical_cols = static_cast<size_t>(dim_j * dim_z * dim_w);

    args.sB = logical_cols;

    const block_q8_0 *block_base = ggml_gemmini_args_block_base(src0);
    ggml_gemmini_args_t::unpacked_weight *cached = nullptr;

    auto it = ctx->weight_cache.find(src0);
    if (it != ctx->weight_cache.end() &&
        it->second.matches(block_base,
                           dim_k,
                           dim_j,
                           dim_z,
                           dim_w,
                           args.sB,
                           blocks_K,
                           logical_cols)) {
        cached = &it->second;
        DBG_SIMPLE("[Breakdowned weight cache] hit layer=%s ptr=%p K=%lld cols=%zu sB=%zu blocks_K=%zu",
                   layer, (const void *)src0, static_cast<long long>(dim_k),
                   logical_cols, args.sB, blocks_K);
    } else {
        if (it == ctx->weight_cache.end()) {
            DBG_SIMPLE("[Breakdowned weight cache] miss layer=%s ptr=%p K=%lld cols=%zu sB=%zu blocks_K=%zu",
                       layer, (const void *)src0, static_cast<long long>(dim_k),
                       logical_cols, args.sB, blocks_K);
        } else {
            DBG_SIMPLE("[Breakdowned weight cache] refresh layer=%s ptr=%p K=%lld cols=%zu sB=%zu blocks_K=%zu",
                       layer, (const void *)src0, static_cast<long long>(dim_k),
                       logical_cols, args.sB, blocks_K);
        }
        ggml_gemmini_args_t::unpacked_weight entry;
        entry.blocks = block_base;
        entry.dim_k = dim_k;
        entry.dim_j = dim_j;
        entry.dim_z = dim_z;
        entry.dim_w = dim_w;
        entry.logical_cols = logical_cols;
        entry.blocks_K = blocks_K;
        entry.blocks_J = logical_cols;
        entry.blocks_I = static_cast<size_t>(dim_k);
        entry.block_size_k = QK8_0;
        entry.stride = args.sB;

        const size_t q_size = static_cast<size_t>(dim_k) * logical_cols;
        const size_t scale_size = blocks_K * logical_cols;
        entry.q.resize(q_size);
        entry.scales.resize(scale_size);

        ggml_gemmini_pack_q80(src0,
                              /*transpose=*/true,
                              reinterpret_cast<elem_t *>(entry.q.data()),
                              entry.stride,
                              entry.scales.data(),
                              args);

        entry.blocks = args.B_blocks;
        entry.blocks_K = args.blocks_K;
        entry.blocks_J = args.blocks_J;
        entry.blocks_I = args.blocks_I;
        entry.block_size_k = args.block_size_k;

        if (it == ctx->weight_cache.end()) {
            it = ctx->weight_cache.emplace(src0, std::move(entry)).first;
        } else {
            it->second = std::move(entry);
        }
        cached = &it->second;
    }

    args.B = reinterpret_cast<elem_t *>(cached->q.data());
    args.B_blocks = cached->blocks;
    args.B_scales = cached->scales.data();
    args.blocks_K = cached->blocks_K;
    args.blocks_J = cached->blocks_J;
    args.blocks_I = cached->blocks_I;
    args.block_size_k = cached->block_size_k;
    DBG_SIMPLE("[Gemmini addr] layer=%s A=%p B=%p B_blocks=%p B_scales=%p",
               layer, (void *)args.A, (void *)args.B, (const void *)args.B_blocks, (const void *)args.B_scales);

    end = read_cycles();
    PRINT_CYCLE(layer, "Breakdown Q8_0", start, end, end - start);

    GGML_ASSERT(args.transpose_B == false);
    GGML_ASSERT(args.sB == logical_cols);

    start = read_cycles();
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

    end = read_cycles();
    PRINT_CYCLE(layer, "Set Args for calling gemmini", start, end, end - start);

    DBG_SIMPLE("[Gemmini debug] layer=%s A=%p B=%p C=%p D=%p I=%zu J=%zu K=%zu sA=%zu sB=%zu sC=%zu stride_f_out(row)=%zu stride_f_out(col)=%zu nb1=%zu nb0=%zu",
           layer, args.A, args.B, args.C, args.D,
           args.I, args.J, args.K, args.sA, args.sB, args.sC,
           args.stride_f_out, args.col_stride_f_out, dst->nb[1], dst->nb[0]);
    DBG_SIMPLE("[Gemmini addr] layer=%s f_out=%p stride_f_out=%zu col_stride_f_out=%zu",
               layer, (void *)args.f_out, args.stride_f_out, args.col_stride_f_out);



    /* __ 5. Gemmini 호출 __ */
    aisa::tiled_matmul_auto_fp32(&args); // gemmini 커널에서 tile과 block 매칭 -> tiled_matmul 호출 후 dequantize까지 수행
    // dst에는 gemmini 커널에서 dequantize한 결과가 들어옴 

#if ERROR_COMPENSATION
    ActivationDEC::compensate(src1, &args);
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
    
    for (int i = 0; i < cgraph->n_nodes; i++)
    {
        struct ggml_tensor *node = cgraph->nodes[i];

        switch (node->op)
        {
        case GGML_OP_MUL_MAT: {
            ggml_backend_gemmini_mul_mat(ctx, node);
            break;
        }
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_ADD:
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
    return ggml_backend_gemmini_init();

    GGML_UNUSED(dev);
    GGML_UNUSED(params);
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
            // BLAS usually is only faster for large matrices
            const struct ggml_tensor *a = op->src[0]; // W
            const struct ggml_tensor *b = op->src[1]; // x
            if (!a || !b)
                return false;

            // Q8_0 × F32 -> F32 만 처리 (우리가 구현한 경로)
            if (a->type != GGML_TYPE_Q8_0)
                return false;
            if (b->type != GGML_TYPE_F32)
                return false;
            if (op->type != GGML_TYPE_F32)
                return false;

            // 필요시 연속성 제약은 완화/강화
            if (!ggml_is_contiguous(a) || !ggml_is_contiguous(b))
                return false;
            return true;
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
