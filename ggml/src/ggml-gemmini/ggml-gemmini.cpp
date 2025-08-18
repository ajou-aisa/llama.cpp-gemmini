#define DEBUG 1
#define TEST 0
#define OPTION CPU // OS, WS, CPU

#include "ggml-gemmini-tensor.h"
#include "include/gemmini.h"
#include <optional>
#include "test/ggml-gemmini-test.h"

using namespace zerogod;

static void ggml_backend_gemmini_mul_mat(
                                         ggml_backend_gemmini_context *ctx,
                                         struct ggml_tensor *dst, // FP32 output (I×J)
                                         struct ggml_tensor *bias) // optional FP32 bias (->int32)
{
/* ______________ Debug: 헤더 사용량 측정 _______________ */
#if DEBUG
    size_t mu0 = ggml_used_mem(ctx->tmp_ctx);
#endif
/* _____________________________________________________ */

/* ________________ Test: 테스트 호출용 ___________________ */
#if TEST
    {
        // OPTION에 따라 동작
        ggml_backend_gemmini_mul_mat_test(/* I */ 2,
                                          /* J */ 3,
                                          /* K */ 2);
        return;
    }
#endif
    /* ______________________________________________________ */

    DBG("[Gemmini] mul_mat call\n");

    /* ____________________________________ 0. 원본 FP32 입력 텐서 ____________________________________________ */
    const auto *src0 = dst->src[0]; // src0: weight (J × K) -> 전치하여 K x J로 사용 (B)
    const auto *src1 = dst->src[1]; // src1: activation (K x J) -> 전치 없음 (A)

    DBG("\ndst shape:\n ne = [%llu, %llu, %llu, %llu]\n", dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
    DBG("\nsrc0 shape:\n ne = [%llu, %llu, %llu, %llu]\n", src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3]);
    DBG("\nsrc1 shape:\n ne = [%llu, %llu, %llu, %llu]\n", src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3]);
    /* _______________________________________________________________________________________________________ */

    /* _____________________________ 1. Gemmini용 텐서 생성 _____________________________ */
    ggml_gemmini_tensor<int8_t> tA(ctx->tmp_ctx, src1, ".i8");              // IxK (1xK)
    ggml_gemmini_tensor<int8_t> tB(ctx->tmp_ctx, src0, ".i8", false, true); // KxJ, 전치
    ggml_gemmini_tensor<int8_t> tC(ctx->tmp_ctx, dst, ".i8", true);         // IxJ (1xJ)
    std::optional<ggml_gemmini_tensor<int32_t>> tD;                         // optional로 임시 생성
    if (bias)
        tD.emplace(ctx->tmp_ctx, bias, ".i32");

/* ____________________ Debug: 헤더 사용량 측정 ___________________ */
#if DEBUG

    size_t mu1 = ggml_used_mem(ctx->tmp_ctx);
    size_t mul_hdr = (mu1 >= mu0) ? (mu1 - mu0) : mu1;
    DBG("[Gemmini] MUL_MAT header used = %zu bytes\n", mul_hdr);
    
#endif
/* ______________________________________________________________ */

    /* ___________________ Guard: runtime dimension _____________________ */
    GGML_ASSERT(tA.get_rows() == dst->ne[1]);    // I == dst rows (보통 1)
    GGML_ASSERT(tA.get_cols() == tB.get_rows()); // K == K
    GGML_ASSERT(tC.get_rows() == tA.get_rows()); // I == I
    GGML_ASSERT(tC.get_cols() == tB.get_cols()); // J(padded) == J(padded)
    GGML_ASSERT(dst->ne[0] <= tC.get_cols());    // 논리 J <= 패딩 J
    /* __________________________________________________________________ */

    /* _______________________ 2. Gemmini용 dimension _____________________ */
    const size_t I = tC.get_rows(); // I = A.ne[1], K = A.ne[0]
    const size_t J = tC.get_cols(); // K = B.ne[0], J = B.ne[1] (transpose)
    const size_t K = tA.get_cols();
    DBG("I=%zu, J=%zu, K=%zu\n", I, J, K);
    /* ____________________________________________________________________ */

    /* _____ 3. Gemmini용 stride _____ */
    const size_t sA = tA.get_stride();
    const size_t sB = tB.get_stride();
    const size_t sC = tC.get_stride();
    GGML_ASSERT(sA % 16 == 0);
    GGML_ASSERT(sB % 16 == 0);
    GGML_ASSERT(sC % 16 == 0);
    /* _______________________________ */

    /* ______________________________ 4. bias 텐서 처리 _________________________________ */
    std::vector<int32_t> zero_bias(tC.get_cols(), 0);

    const int32_t *bias_data = tD ? static_cast<int32_t *>(tD->get()) : zero_bias.data();
    const size_t sD = tD ? tD->get_stride() : 0;
    const bool repeating = tD ? tD->get_rows() == 1 : true;

    DBG("calling tiled_matmul_auto: ptrA=%p ptrB=%p ptrD=%p ptrC=%p\n",
        (void *)tA.get(), (void *)tB.get(), (void *)bias_data, (void *)tC.get());
    /* _________________________________________________________________________________ */

    /* __ 5. Gemmini tiled_matmul_auto 호출 __ */
    tiled_matmul_auto(I, J, K,
                      (elem_t *)tA.get(),
                      (elem_t *)tB.get(),
                      (void *)bias_data,
                      (elem_t *)tC.get(),
                      sA, sB, sD, sC,
                      1.f, 1.f, 1.f,
                      NO_ACTIVATION,
                      1, 1,
                      repeating,
                      false, // transpose_A
                      false, // transpose_B
                      false, false,
                      0, CPU);
    /* _______________________________________ */

    /* _____________ 6. Gemmini 연산 결과를 원본 출력 텐서로 반영 _____________ */
    const size_t nb1_out = dst->nb[1]; // 출력 텐서 행 stride (bytes)
    const size_t J_log = dst->ne[0];   // 실제 논리 열 수

    int8_t *c_i8 = static_cast<int8_t *>(tC.get());
    uint8_t *out_base = static_cast<uint8_t *>(dst->data);

    for (size_t r = 0; r < I; ++r)
    {
        const int8_t *row_c = c_i8 + r * sC;
        float *row_out = reinterpret_cast<float *>(out_base + r * nb1_out);

        // 스케일/클리핑 없이 그대로 float 캐스팅
        for (size_t c = 0; c < J_log; ++c)
            row_out[c] = static_cast<float>(row_c[c]);
        // 패딩 영역(row_c[J..sC-1])은 무시
    }
    /* ____________________________________________________________________ */
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

    // (1) bias_map 갱신
    ctx->bias_map.clear();
    for (int i = 0; i < cgraph->n_nodes; i++) {
        auto *node = cgraph->nodes[i];
        if (node->op == GGML_OP_ADD && node->src[0]->op == GGML_OP_MUL_MAT)
            ctx->bias_map[node->src[0]] = node->src[1];
    }

    struct ggml_init_params ip = {
        /* .mem_size   = */ 8ull * 1024 * 1024, // 8MiB
        /* .mem_buffer = */ NULL,
        /* .no_alloc   = */ true, // 헤더만
    };

    ctx->tmp_ctx = ggml_init(ip);
    GGML_ASSERT(ctx->tmp_ctx);

/* __________________________ Debug: 헤더 사용량 측정 ____________________________ */
#if DEBUG

    DBG("[Gemmini] sizeof(ggml_tensor) = %zu\n", sizeof(struct ggml_tensor));
    size_t used0 = ggml_used_mem(ctx->tmp_ctx);
    DBG("[Gemmini] tmp_ctx used(start) = %zu bytes\n", used0);

#endif
/* _____________________________________________________________________________ */

    for (int i = 0; i < cgraph->n_nodes; i++)
    {
        struct ggml_tensor *node = cgraph->nodes[i];

        switch (node->op)
        {
        case GGML_OP_MUL_MAT: {
            ggml_tensor *bias = nullptr;
            auto it = ctx->bias_map.find(node);
            if (it != ctx->bias_map.end())
                bias = it->second;

            ggml_backend_gemmini_mul_mat(ctx, node, bias);
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
            break;

        default:
            GGML_ABORT("%s: unsupported op %s\n", __func__, ggml_op_desc(node));
        }
    }

/* _______________________________________ Debug: 헤더 사용량 측정 ________________________________________________ */
#if DEBUG

    size_t used1 = ggml_used_mem(ctx->tmp_ctx);
    size_t hdr_bytes = (used1 >= used0) ? (used1 - used0) : used1; // 방어적
    DBG("[Gemmini] tmp_ctx header used(total) = %zu bytes (%.2f MiB)\n", hdr_bytes, hdr_bytes / (1024.0 * 1024.0));

#endif
/* _______________________________________________________________________________________________________________ */

    ctx->bias_map.clear();
    // tmp_ctx 해제
    ggml_free(ctx->tmp_ctx);
    ctx->tmp_ctx = nullptr;

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
