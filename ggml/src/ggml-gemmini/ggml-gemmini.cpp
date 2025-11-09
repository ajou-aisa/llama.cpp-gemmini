// ggml-gemmini.cpp

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cctype>
#include <vector>
#include <string>

#include "ggml-gemmini-util.h"
#include "include/gemmini.h"
#include "labeling/label.h"
#include "error_compensation/activation_DEC.h"
#include "ggml-gemmini-args.h"

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

using namespace aisa;

// Cycle 측정 용
extern "C" volatile uint64_t gemmini_tiled_matmul_cycles = 0; // gemmini.h

uint64_t start, end; // 일반 사이클 측정

#if defined(GGML_GEMMINI_FORCE_GGML_OUTPUT) && GGML_GEMMINI_FORCE_GGML_OUTPUT
static inline const char *ggml_tensor_data_start(const struct ggml_tensor *tensor)
{
    const char *base = reinterpret_cast<const char *>(tensor->view_src ? tensor->view_src->data : tensor->data);
    const size_t offs = tensor->view_src ? tensor->view_offs : 0;
    return base + offs;
}

static inline char *ggml_tensor_data_start(struct ggml_tensor *tensor)
{
    char *base = reinterpret_cast<char *>(tensor->view_src ? tensor->view_src->data : tensor->data);
    const size_t offs = tensor->view_src ? tensor->view_offs : 0;
    return base + offs;
}

static inline std::string ggml_gemmini_trim(const std::string &s)
{
    const auto first = s.find_first_not_of(" \t\r\n");
    if (first == std::string::npos)
        return {};
    const auto last = s.find_last_not_of(" \t\r\n");
    return s.substr(first, last - first + 1);
}

static inline const std::vector<std::string> &ggml_gemmini_force_layer_tokens()
{
    static bool initialized = false;
    static std::vector<std::string> tokens;
    if (!initialized)
    {
        const char *env = std::getenv("GGML_GEMMINI_FORCE_GGML_LAYERS");
        if (env && *env)
        {
            std::string raw(env);
            size_t pos = 0;
            while (pos < raw.size())
            {
                const size_t comma = raw.find(',', pos);
                const std::string token = ggml_gemmini_trim(raw.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos));
                if (!token.empty())
                    tokens.push_back(token);
                if (comma == std::string::npos)
                    break;
                pos = comma + 1;
            }
        }
        if (tokens.empty())
        {
            tokens.push_back("output");   // lm_head 등
            tokens.push_back("ffn_down"); // FFN projection
        }
        initialized = true;
    }
    return tokens;
}

static inline bool ggml_gemmini_force_all_layers()
{
    static bool initialized = false;
    static bool force_all = true;
    if (!initialized)
    {
        const char *env = std::getenv("GGML_GEMMINI_FORCE_GGML_ALL");
        if (env && *env)
        {
            std::string value = ggml_gemmini_trim(env);
            std::transform(value.begin(), value.end(), value.begin(),
                           [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
            if (value == "0" || value == "false" || value == "no")
                force_all = false;
            else
                force_all = true;
        }
        else
        {
            force_all = true;
        }
        initialized = true;
    }
    return force_all;
}

static inline bool ggml_gemmini_should_force_ggml(const char *layer_name, const struct ggml_tensor *dst)
{
    if (layer_name)
    {
        for (const auto &token : ggml_gemmini_force_layer_tokens())
        {
            if (!token.empty() && std::strstr(layer_name, token.c_str()) != nullptr)
                return true;
        }
    }
    const struct ggml_tensor *src0 = dst->src[0];
    // vocab size tends to be large (e.g. 50k). Use heuristic to detect output-like heads automatically.
    if (src0 && src0->ne[1] >= 32000)
        return true;
    return false;
}

static bool ggml_gemmini_mul_mat_cpu_output(struct ggml_tensor *dst)
{
    const struct ggml_tensor *src0 = dst->src[0];
    const struct ggml_tensor *src1 = dst->src[1];

    if (src0 == nullptr || src1 == nullptr)
        return false;
    if (src0->type != GGML_TYPE_Q8_0 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32)
        return false;

    const size_t I = dst->ne[1] ? dst->ne[1] : 1;
    const size_t J = dst->ne[0];
    const size_t K = src1->ne[0];

    if (K % QK8_0 != 0)
        return false;

    const int64_t dim_j = src0->ne[1] ? src0->ne[1] : 1;
    const int64_t dim_z = src0->ne[2] ? src0->ne[2] : 1;
    const int64_t dim_w = src0->ne[3] ? src0->ne[3] : 1;
    const size_t logical_cols = static_cast<size_t>(dim_j * dim_z * dim_w);
    GGML_ASSERT(J <= logical_cols);

    const char *a_base = ggml_tensor_data_start(src1);
    char *dst_base = ggml_tensor_data_start(dst);

    for (size_t i = 0; i < I; ++i)
    {
        for (size_t j = 0; j < J; ++j)
        {
            size_t rem = j;
            const size_t cols_per_w = static_cast<size_t>(dim_j * dim_z);
            const int64_t iw_idx = cols_per_w ? static_cast<int64_t>(rem / cols_per_w) : 0;
            rem = cols_per_w ? rem % cols_per_w : 0;
            const int64_t iz_idx = dim_j ? static_cast<int64_t>(rem / dim_j) : 0;
            const int64_t iy_idx = dim_j ? static_cast<int64_t>(rem % dim_j) : 0;

            const block_q8_0 *row_blocks = ggml_gemmini_get_q80_row_ptr(src0, iy_idx, iz_idx, iw_idx);
            GGML_ASSERT(row_blocks != nullptr);

            double acc = 0.0;
            for (size_t k = 0; k < K; ++k)
            {
                const size_t blk = k / QK8_0;
                const size_t off = k % QK8_0;
                const block_q8_0 &b = row_blocks[blk];
                const float w = ggml_fp16_to_fp32(b.d) * static_cast<float>(b.qs[off]);
                const float *a_ptr = reinterpret_cast<const float *>(a_base + i * src1->nb[1] + k * src1->nb[0]);
                acc += static_cast<double>(*a_ptr) * static_cast<double>(w);
            }

            float *dst_ptr = reinterpret_cast<float *>(dst_base + i * dst->nb[1] + j * dst->nb[0]);
            *dst_ptr = static_cast<float>(acc);
        }
    }

    return true;
}
#endif

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

#if defined(GGML_GEMMINI_FORCE_GGML_OUTPUT) && GGML_GEMMINI_FORCE_GGML_OUTPUT
    const bool force_all_layers = ggml_gemmini_force_all_layers();
    if (force_all_layers || ggml_gemmini_should_force_ggml(layer, dst))
    {
        if (ggml_gemmini_mul_mat_cpu_output(dst))
        {
            DBG_SIMPLE("[Gemmini] layer=%s handled by ggml CPU (%s)\n",
                       layer ? layer : "(unnamed)", force_all_layers ? "force all" : "force list");
            return;
        }
        else
        {
            DBG_SIMPLE("[Gemmini] layer=%s requested CPU force but shape/type unsupported, falling back to Gemmini\n",
                       layer ? layer : "(unnamed)");
        }
    }
#endif

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

    // quantize activation
    static thread_local std::vector<int8_t> activation_q;
    activation_q.resize(I * K);
    int8_t *qx = activation_q.data();

    ggml_gemmini_quantize_activation(src1, args, qx);
    
    args.A = reinterpret_cast<elem_t *>(qx);

    // breackdown weight to int8_t & scale
    const int64_t dim_j = src0->ne[1] ? src0->ne[1] : 1;
    const int64_t dim_z = src0->ne[2] ? src0->ne[2] : 1;
    const int64_t dim_w = src0->ne[3] ? src0->ne[3] : 1;

    GGML_ASSERT(K % QK8_0 == 0);
    const size_t blocks_K = static_cast<size_t>(K) / QK8_0;
    const size_t logical_cols = static_cast<size_t>(dim_j * dim_z * dim_w);

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

    args.sB = logical_cols;

    ggml_gemmini_pack_q80(src0, /*transpose=*/true, reinterpret_cast<elem_t *>(qw), args.sB, block_scale_w, args);

    args.B = reinterpret_cast<elem_t *>(qw);

    // Ensure the logical transpose/stride flags match the physical packing.
#if (GEMMINI_GOLDEN_CHECK || GEMMINI_GOLDEN_MM)
    const auto get_q80_row_ptr = [&](size_t logical_col) -> const block_q8_0 *
    {
        size_t rem = logical_col;
        const size_t dim_j_sz = static_cast<size_t>(dim_j ? dim_j : 1);
        const size_t dim_z_sz = static_cast<size_t>(dim_z ? dim_z : 1);
        const size_t cols_per_w = dim_j_sz * dim_z_sz;
        const int64_t iw_idx = cols_per_w ? static_cast<int64_t>(rem / cols_per_w) : 0;
        rem = cols_per_w ? rem % cols_per_w : 0;
        const int64_t iz_idx = dim_j_sz ? static_cast<int64_t>(rem / dim_j_sz) : 0;
        const int64_t iy_idx = dim_j_sz ? static_cast<int64_t>(rem % dim_j_sz) : 0;
        return ggml_gemmini_get_q80_row_ptr(src0, iy_idx, iz_idx, iw_idx);
    };
#endif

    GGML_ASSERT(args.transpose_B == false);
    GGML_ASSERT(args.sB == logical_cols);

#if defined(GEMMINI_GOLDEN_CHECK) && GEMMINI_GOLDEN_CHECK
    // Compare the packed (qs, d) representation against the original Q8_0 blocks.
    GGML_ASSERT(args.B_scales != nullptr);
    GGML_ASSERT(args.blocks_J >= static_cast<size_t>(J));

    const auto get_w_ref = [&](size_t k, size_t j) -> float
    {
        const size_t blk = k / QK8_0;
        const size_t off = k % QK8_0;
        const block_q8_0 *row_blocks = get_q80_row_ptr(j);
        return ggml_fp16_to_fp32(row_blocks[blk].d) * static_cast<float>(row_blocks[blk].qs[off]);
    };

    const auto get_w_sep = [&](size_t k, size_t j) -> float
    {
        const size_t blk = k / args.block_size_k;
        const float d = args.B_scales[blk * args.blocks_J + j];
        const int8_t q = qw[k * args.sB + j];
        return d * static_cast<float>(q);
    };

    size_t bad = 0, tot = 0;
    double mae = 0.0, rmse = 0.0, maxd = 0.0;
    const size_t J_chk = std::min(static_cast<size_t>(J), static_cast<size_t>(8));
    const size_t K_chk = std::min(static_cast<size_t>(K), static_cast<size_t>(64));
    for (size_t jx = 0; jx < J_chk; ++jx)
    {
        for (size_t kx = 0; kx < K_chk; ++kx)
        {
            const double diff = static_cast<double>(get_w_ref(kx, jx)) - static_cast<double>(get_w_sep(kx, jx));
            mae += std::fabs(diff);
            rmse += diff * diff;
            maxd = std::max(maxd, std::fabs(diff));
            ++tot;
            if (std::fabs(diff) > 1e-5)
            {
                ++bad;
            }
        }
    }
    if (tot)
    {
        mae /= static_cast<double>(tot);
        rmse = std::sqrt(rmse / static_cast<double>(tot));
    }
    DBG_SIMPLE("[golden-W] bad=%zu/%zu mae=%.3e rmse=%.3e max|d|=%.3e tB=%d sB=%zu\n",
               bad, tot, mae, rmse, maxd, static_cast<int>(args.transpose_B), args.sB);
    GGML_ASSERT(bad == 0 && "B(qs,d) != dequant(Q8_0): transpose/stride/order mismatch");
#endif
    
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
#if CYCLE_LOG
    PRINT_CYCLE(layer, "Set Args for calling gemmini", start, end, end - start);
#endif

    DBG_SIMPLE("[Gemmini debug] layer=%s A=%p B=%p C=%p D=%p I=%zu J=%zu K=%zu sA=%zu sB=%zu sC=%zu stride_f_out(row)=%zu stride_f_out(col)=%zu nb1=%zu nb0=%zu\n",
           layer, args.A, args.B, args.C, args.D,
           args.I, args.J, args.K, args.sA, args.sB, args.sC,
           args.stride_f_out, args.col_stride_f_out, dst->nb[1], dst->nb[0]);



    /* __ 5. Gemmini 호출 __ */
    aisa::tiled_matmul_auto_fp32(&args); // gemmini 커널에서 tile과 block 매칭 -> tiled_matmul 호출 후 dequantize까지 수행
    // dst에는 gemmini 커널에서 dequantize한 결과가 들어옴 

#if defined(GEMMINI_GOLDEN_MM) && GEMMINI_GOLDEN_MM
    {
        const int8_t *A_q = reinterpret_cast<const int8_t *>(args.A);
        float *dst_f32 = static_cast<float *>(dst->data);
        if (A_q != nullptr && dst_f32 != nullptr)
        {
            const size_t stride_a_q = args.sA ? args.sA : K;
            const size_t stride_dst = args.stride_f_out ? args.stride_f_out : J;
            const size_t stride_dst_col = args.col_stride_f_out ? args.col_stride_f_out : 1;

            const auto weight_ref_at = [&](size_t k, size_t j) -> float
            {
                const size_t blk = k / QK8_0;
                const size_t off = k % QK8_0;
                const block_q8_0 *row_blocks = get_q80_row_ptr(j);
                return ggml_fp16_to_fp32(row_blocks[blk].d) * static_cast<float>(row_blocks[blk].qs[off]);
            };

            double mae = 0.0, rmse = 0.0, maxd = 0.0;
            size_t bad = 0, tot = 0;
            const size_t I_chk = std::min(static_cast<size_t>(I), static_cast<size_t>(4));
            const size_t J_chk = std::min(static_cast<size_t>(J), static_cast<size_t>(8));

            for (size_t i0 = 0; i0 < I_chk; ++i0)
            {
                const int8_t *row_a = A_q + i0 * stride_a_q;
                const float *row_dst = dst_f32 + i0 * stride_dst;
                for (size_t j0 = 0; j0 < J_chk; ++j0)
                {
                    double ref = 0.0;
                    for (size_t k0 = 0; k0 < K; ++k0)
                    {
                        const double a_deq = static_cast<double>(row_a[k0]) * static_cast<double>(args.scale_A);
                        ref += a_deq * static_cast<double>(weight_ref_at(k0, j0));
                    }
                    const double got = static_cast<double>(row_dst[j0 * stride_dst_col]);
                    const double diff = ref - got;
                    mae += std::fabs(diff);
                    rmse += diff * diff;
                    maxd = std::max(maxd, std::fabs(diff));
                    ++tot;
                    if (std::fabs(diff) > 5e-3)
                    {
                        ++bad;
                    }
                }
            }
            if (tot)
            {
                mae /= static_cast<double>(tot);
                rmse = std::sqrt(rmse / static_cast<double>(tot));
            }
            // golden-MM 로그 해석:
            //  - bad: 허용 오차(현재 5e-3)보다 큰 비교 지점 수 / 전체 샘플 수
            //  - mae/rmse: 양자화 경로 vs 참조(복원된 qA × fp32 W)의 평균/제곱근 평균 오차
            //  - max|d|: 단일 요소 최대 오차 (이 값이 작으면 전체 벡터도 잘 맞음)
            DBG_SIMPLE("[golden-MM] bad=%zu/%zu mae=%.3e rmse=%.3e max|d|=%.3e\n", bad, tot, mae, rmse, maxd);
        }
        else
        {
            DBG_SIMPLE("[golden-MM] skipped (missing src/dst data)\n");
        }
    }
#endif

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
