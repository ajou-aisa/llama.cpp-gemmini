// ggml-gemmini.cpp

#include <cstring>
#include <vector>

#include "ggml-gemmini-util.h"
#include "gemmini_tensor/gemmini_tensor_interface.h"
#include "include/gemmini.h"
#include "labeling/label.h"
#include "test/ggml-gemmini-test.h"
#include "gemmini_tensor/error_compensation/activation_DEC.h"
#include "gemmini_tensor/quant_tensor_view.h"
#include "ggml-gemmini-args.h"

#ifndef SCALE_A
#define SCALE_A 1.0f
#endif
#ifndef FULL_C
#define FULL_C 0
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

using namespace aisa;

// Cycle 측정 용
extern "C" volatile uint64_t gemmini_tiled_matmul_cycles = 0; // gemmini.h

uint64_t start, end; // 일반 사이클 측정

static void ggml_backend_gemmini_mul_mat(ggml_backend_gemmini_context *ctx,
                                         struct ggml_tensor *dst) // FP32 output (I×J)
{

/* ________________ Test: 테스트 호출용 ___________________ */
#if TEST
    static_assert(TRANSPOSE_B == 1, "This test assumes physical-transposed B (KxJ).");
    ggml_gemmini_test(ctx, dst);
    return;
#endif

    DBG("[Gemmini] mul_mat call\n");

    /* ____________________________________ 0. 원본 FP32 입력 텐서 ____________________________________________ */
    const auto *src0 = dst->src[0]; // src0: weight (K x J) -> 전치
    const auto *src1 = dst->src[1]; // src1: activation (I x K) -> 전치 없음 (A)

    ggml_gemmini_args_t args; // DEC과 gemmini 호출을 위한 args 
    // set args
    start = read_cycles();

    const char *w_name = (src0 && src0->name) ? src0->name : ""; // weight name
    const char *layer = labelFromWeight(w_name); // layer 이름 추출
    const bool pack_transpose_B = TRANSPOSE_B != 0;
    
    args.transpose_B = !pack_transpose_B; // Gemmini 쪽에는 실제 메모리 레이아웃에 맞게 전달
    args.layer_name = layer;
    args.full_C = FULL_C;
    args.low_D = LOW_D;
    
    /* _______________________ 2. Gemmini용 dimension _____________________ */
    const size_t I = dst->ne[1]; // I = A.ne[1], K = A.ne[0]
    const size_t J = dst->ne[0]; // K = B.ne[0], J = B.ne[1] (transpose)
    const size_t K = src1->ne[0]; // K = A.ne[0]
    DBG("I=%zu, J=%zu, K=%zu\n", I, J, K);

    args.I = I;
    args.J = J;
    args.K = K;
    
    /* _____ 3. Gemmini용 stride _____ */
    args.sA = K;
    args.sC = J;

    // quantize activation
    static thread_local std::vector<int8_t> activation_q;
    activation_q.resize(I * K);
    int8_t *qx = activation_q.data();

    ggml_gemmini_quantize_activation(src1, args, qx);
    
    args.A = reinterpret_cast<elem_t *>(qx);
    
    QuantTensorView qA_view{qx, I, K, args.sA};
    ConstQuantTensorView qA_const = make_const_view(qA_view);

    // breackdown weight to int8_t & scale
    const int64_t dim_z = src0->ne[2] ? src0->ne[2] : 1;
    const int64_t dim_w = src0->ne[3] ? src0->ne[3] : 1;

    GGML_ASSERT(K % QK8_0 == 0);
    const size_t blocks_K = static_cast<size_t>(K) / QK8_0;
    const size_t logical_cols = static_cast<size_t>(J * dim_z * dim_w);

    static thread_local std::vector<int8_t> weight_q;
    static thread_local std::vector<float> weight_scales;

    const size_t q_size = static_cast<size_t>(K) * logical_cols;
    if (weight_q.size() != q_size) 
        weight_q.resize(q_size);
    
    const size_t scale_size = blocks_K * logical_cols;
    if (weight_scales.size() != scale_size) 
        weight_scales.resize(scale_size);

    int8_t *qw = weight_q.data();
    float *block_scale_w = weight_scales.data();

    args.sB = pack_transpose_B ? logical_cols : static_cast<size_t>(K);

    ggml_gemmini_pack_q80(src0, pack_transpose_B, reinterpret_cast<elem_t *>(qw), args.sB, block_scale_w, args);

    args.B = reinterpret_cast<elem_t *>(qw);
    
    QuantTensorView qW_view_local;
    if (pack_transpose_B) 
        qW_view_local = QuantTensorView{qw, static_cast<size_t>(K), logical_cols, args.sB};
    else 
        qW_view_local = QuantTensorView{qw, logical_cols, static_cast<size_t>(K), args.sB};
    
    ConstQuantTensorView qW_view = make_const_view(qW_view_local);
    
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
    args.stride_f_out = dst->nb[1] / sizeof(float);
    args.tiled_matmul_type = OPTION;

    end = read_cycles();
#if CYCLE_LOG
    fprintf(stderr, "[layer=%s][Set Args for calling gemmini] start = %lu, end = %lu, elapsed = %lu\n", layer, start, end, end - start);
    fprintf(stderr, "[layer=%s]", layer); // tiled_matmul_auto 내부 사이클 출력에 layer 추가
#endif

    DBG("[Gemmini debug] layer=%s A=%p B=%p C=%p D=%p I=%zu J=%zu K=%zu sA=%zu sB=%zu sC=%zu stride_f_out=%zu nb1=%zu\n",
           layer, args.A, args.B, args.C, args.D,
           args.I, args.J, args.K, args.sA, args.sB, args.sC,
           args.stride_f_out, dst->nb[1]);



    /* __ 5. Gemmini 호출 __ */
    aisa::tiled_matmul_auto_fp32(&args); // gemmini 커널에서 tile과 block 매칭 -> tiled_matmul 호출 후 dequantize까지 수행
    // dst에는 gemmini 커널에서 dequantize한 결과가 들어옴 

#if ERROR_COMPENSATION
    ActivationDEC::compensate(src1, &args);
#else
    GGML_UNUSED(qA_view);
    GGML_UNUSED(qW_view);
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
        case GGML_OP_OUT_PROD:
            // ggml_backend_gemmini_out_prod(ctx, node);
            break;

        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_ADD:
            break;

        default:
            GGML_ABORT("%s: unsupported op %s\n", __func__, ggml_op_desc(node));
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
        /* .host_buffer           = */ false,
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
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];

    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_ADD:

        case GGML_OP_TRANSPOSE:
            return true;

        case GGML_OP_MUL_MAT:
        {
            // BLAS usually is only faster for large matrices
            const struct ggml_tensor * src0 = op->src[0];
            const struct ggml_tensor * src1 = op->src[1];

            const int64_t ne10 = src1->ne[0];

            const int64_t ne0 = op->ne[0];
            const int64_t ne1 = op->ne[1];

            // TODO: find the optimal value
            const int64_t min_batch = 32;

            return ggml_is_contiguous(src0) &&
                   ggml_is_contiguous(src1) &&
                   // src1->type == GGML_TYPE_F32 &&
                   // (ne0 >= min_batch && ne1 >= min_batch && ne10 >= min_batch) &&
                   // (src0->type == GGML_TYPE_F32 || ggml_get_type_traits(src0->type)->to_float != NULL);
                   true;
        }

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
