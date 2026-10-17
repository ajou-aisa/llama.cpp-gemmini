#include "dense.hpp"

#include "ggml-backend.h"
#include "potal_hw_params.h"

#include "../quants/act/quantize.hpp"
#include "../quants/act/tensor/types.hpp"
#include "../quants/common/hp1_scu.hpp"
#include "../quants/common/tensor_util.hpp"

#include <cstdlib>
#include <variant>
#include <cstring>

namespace ggml::potal {

namespace {

constexpr size_t DIM   = POTAL_HW_DIM;
constexpr size_t ALIGN = POTAL_HW_MAX_BYTES;

size_t round_up(size_t v, size_t to) {
    return (v + to - 1) / to * to;
}

bool is_2d(const ggml_tensor * t) {
    return t->ne[2] == 1 && t->ne[3] == 1;
}

bool in_host_memory(const ggml_tensor * t) {
    const ggml_tensor * base = t->view_src ? t->view_src : t;
    return base->buffer == nullptr || ggml_backend_buffer_is_host(base->buffer);
}

void * zeroed(size_t bytes) {
    void * p = std::aligned_alloc(ALIGN, round_up(bytes, ALIGN));
    if (p != nullptr) {
        std::memset(p, 0, round_up(bytes, ALIGN));
    }
    return p;
}

} // namespace

ggml_potal_reject check_mul_mat_hp1(const ggml_tensor * op, bool require_data, hp1_shape * shape) {
    const ggml_tensor * w = op->src[0];
    const ggml_tensor * x = op->src[1];
    if (op->op != GGML_OP_MUL_MAT || w == nullptr || x == nullptr) {
        return GGML_POTAL_REJECT_OP;
    }
    for (size_t i = 1; i < GGML_MAX_OP_PARAMS / sizeof(int32_t); ++i) {
        if (op->op_params[i] != 0) {
            return GGML_POTAL_REJECT_OP_PARAMS;
        }
    }
    if (w->type != GGML_TYPE_Q8_HP1 || x->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
        return GGML_POTAL_REJECT_TYPE;
    }
    const int64_t k = w->ne[0];
    const int64_t n = w->ne[1];
    const int64_t m = x->ne[1];
    if (k <= 0 || n <= 0 || m <= 0 || k % (int64_t) quants::weight::hp1::block_k != 0 || x->ne[0] != k ||
        op->ne[0] != n || op->ne[1] != m || !is_2d(w) || !is_2d(x) || !is_2d(op)) {
        return GGML_POTAL_REJECT_SHAPE;
    }
    // Whole-block weight rows; the copied TENSOR quantizer reads src1 as a dense I x K tensor.
    if (!ggml_is_contiguous(w) || !ggml_is_contiguous(x) || !ggml_is_contiguous(op)) {
        return GGML_POTAL_REJECT_LAYOUT;
    }
    if (!in_host_memory(w) || !in_host_memory(x) || !in_host_memory(op) ||
        (require_data && (w->data == nullptr || x->data == nullptr || op->data == nullptr))) {
        return GGML_POTAL_REJECT_MEMORY;
    }
    if (shape != nullptr) {
        *shape = { m, n, k };
    }
    return GGML_POTAL_REJECT_NONE;
}

hp1_operands::~hp1_operands() {
    for (void * p : blocks) {
        std::free(p);
    }
}

void hp1_operands::abandon() {
    for (void *& p : blocks) {
        p = nullptr;
    }
}

bool hp1_operands::allocate(const hp1_shape & s) {
    m_pad   = round_up((size_t) s.m, DIM);
    n_pad   = round_up((size_t) s.n, DIM);
    lda     = round_up((size_t) s.k, ALIGN);
    ldb     = round_up(n_pad, ALIGN);
    cstride = n_pad;
    ldc     = n_pad;
    const size_t k_blocks = (size_t) s.k / quants::weight::hp1::block_k;

    a        = static_cast<int8_t *>(blocks[0] = zeroed(m_pad * lda));
    b        = static_cast<int8_t *>(blocks[1] = zeroed((size_t) s.k * ldb));
    carriers = static_cast<uint32_t *>(blocks[2] = zeroed(k_blocks * cstride * sizeof(uint32_t)));
    scale    = static_cast<float *>(blocks[3] = zeroed(n_pad * sizeof(float)));
    c        = static_cast<int32_t *>(blocks[4] = zeroed(m_pad * ldc * sizeof(int32_t)));
    return a && b && carriers && scale && c;
}

bool init_args(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const hp1_shape & s,
               const char * model_arch, const char * layer, ggml_potal_args_t & args) {
    (void) src1;
    args.I                     = (size_t) s.m;
    args.J                     = (size_t) s.n;
    args.K                     = (size_t) s.k;
    args.sA                    = args.K;
    args.sB                    = args.K;
    args.sC                    = args.J;
    args.weight_format         = ggml_potal_args_t::weight_format_t::q8_hp1;
    args.q8_hp1_blocks         = weight_hp1_base(src0);
    args.q8_hp1_blocks_per_row = args.K / QK8_HP;
    args.q8_hp1_block_count    = args.J * args.q8_hp1_blocks_per_row;
    args.native_weight_bytes   = ggml_nbytes(src0);
    args.block_size_k          = QK8_HP;
    args.f_out                 = static_cast<float *>(dst->data);
    args.col_stride_f_out      = 1;
    args.stride_f_out          = dst->nb[1] / sizeof(float);
    args.model_arch            = model_arch;
    args.matmul_layer          = layer != nullptr ? layer : "";
    return args.A.allocate(args.I, args.K, GGML_GEMMINI_ACTIVATION_BITS);
}

bool quantize_activations(const ggml_tensor * src1, ggml_potal_args_t & args, hp1_operands & ops,
                          float * activation_scale) {
    if (!quants::quantize_activation(src1, args)) {
        return false;
    }
    const auto * meta = std::get_if<quants::act::tensor::Meta>(&args.act_quant.storage());
    const int8_t * codes = args.A;
    if (meta == nullptr || codes == nullptr) {
        return false;
    }
    for (size_t m = 0; m < args.I; ++m) {
        std::memcpy(ops.a + m * ops.lda, codes + m * args.K, args.K);
    }
    *activation_scale = meta->scale;
    return true;
}

quants::weight::hp1::status unpack_weights(const ggml_potal_args_t & args, hp1_operands & ops,
                                           quants::weight::hp1::stats * stats,
                                           quants::wreader::WeightReaderStatus * reader) {
    return quants::weight::hp1::unpack(args, ops.b, ops.ldb, ops.carriers, ops.cstride, ops.scale, stats, reader);
}

void reconstruct(const hp1_operands & ops, const ggml_potal_args_t & args, float activation_scale) {
    for (size_t m = 0; m < args.I; ++m) {
        float *         out = args.f_out + m * args.stride_f_out;
        const int32_t * c   = ops.c + m * ops.ldc;
        for (size_t n = 0; n < args.J; ++n) {
            out[n * args.col_stride_f_out] = static_cast<float>(
                static_cast<double>(c[n]) * static_cast<double>(activation_scale) * static_cast<double>(ops.scale[n]));
        }
    }
}

void reference_gemm_i32(const int8_t * a, size_t lda, const int8_t * b, size_t ldb, const uint32_t * carriers,
                        size_t cstride, size_t block_k, size_t M, size_t N, size_t K, int32_t * c, size_t ldc) {
    namespace hp1 = quants::hp1; // PoTalSCU + Sat32 ACC (copied hp1_scu.hpp)
    const size_t passes = (K + DIM - 1) / DIM;
    for (size_t m = 0; m < M; ++m) {
        for (size_t n = 0; n < N; ++n) {
            int32_t acc = 0;
            for (size_t p = 0; p < passes; ++p) {
                int32_t partial = 0; // |partial| <= DIM * 128 * 128 fits easily
                for (size_t k = p * DIM; k < K && k < (p + 1) * DIM; ++k) {
                    partial += static_cast<int32_t>(a[m * lda + k]) * b[k * ldb + n];
                }
                const int32_t v = hp1::apply_validated(partial, carriers[(p * DIM / block_k) * cstride + n]);
                acc             = p == 0 ? v : hp1::accumulate(acc, v);
            }
            c[m * ldc + n] = acc;
        }
    }
}

} // namespace ggml::potal
