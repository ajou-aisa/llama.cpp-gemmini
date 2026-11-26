#include "dense.hpp"

#include "ggml-backend.h"
#include "potal_hw_params.h"
#include "potal-int4.h"

#include "../../ggml-npu-common/types/geometry.hpp"
#include "../../ggml-npu-common/quantization/activation/dispatch.hpp"
#include "../../ggml-npu-common/quantization/activation/quantize.hpp"
#include "../../ggml-npu-common/quantization/common/hp1_scu.hpp"
#include "../../ggml-npu-common/quantization/common/tensor_util.hpp"
#include "../../ggml-npu-common/residual/direct/direct-executor.hpp"
#include "../../ggml-npu-common/residual/rmd/rmd-compose.hpp"

#include <cstdlib>
#include <cstring>
#include <variant>

static_assert(ggml::npu::quants::act::block::kGroupSize == ggml::potal::quants::weight::hp1::block_k,
              "Block+block needs the activation group to be the HP1 scale block");

namespace ggml::potal {

namespace {

namespace act = ggml::npu::quants::act;

constexpr size_t HW_DIM = POTAL_HW_DIM; // not NPU_D: <gemmini_params.h> defines that macro
constexpr size_t ALIGN  = POTAL_HW_MAX_BYTES > HW_DIM * sizeof(int32_t)
                             ? POTAL_HW_MAX_BYTES : HW_DIM * sizeof(int32_t);

bool is_2d(const ggml_tensor * t) {
    return t->ne[2] == 1 && t->ne[3] == 1;
}

bool in_host_memory(const ggml_tensor * t) {
    const ggml_tensor * base = t->view_src ? t->view_src : t;
    return base->buffer == nullptr || ggml_backend_buffer_is_host(base->buffer);
}

void add_work(potal_ws_report & sum, const potal_ws_report & w) {
    sum.invocations += w.invocations;
    sum.validated_gemms += w.validated_gemms;
    sum.fences += w.fences;
    sum.scale_writes += w.scale_writes;
    sum.scale_releases += w.scale_releases;
    sum.tlb_flushes += w.tlb_flushes;
    sum.last_generation = w.last_generation != 0 ? w.last_generation : sum.last_generation;
    sum.max_scale_slots = sum.max_scale_slots > w.max_scale_slots ? sum.max_scale_slots : w.max_scale_slots;
    sum.scale_s += w.scale_s;
    sum.issue_s += w.issue_s;
    sum.fence_s += w.fence_s;
}

// One PoTal job over the K range [k0, k0 + k) of the staged operands.
tiled_job job_of(const npu_matmul_args & args, const hp1_operands & ops, size_t k0, size_t k) {
    const size_t block = k0 / quants::weight::hp1::block_k;
    return {
        args.I, args.J, k,
        ops.a + k0 / 2, ops.lda, ops.b + k0 * (ops.ldb / 2), ops.ldb,
        ops.carriers + block * ops.cstride, ops.cstride, quants::weight::hp1::block_k,
        ops.c, ops.ldc,
    };
}

} // namespace

size_t align_up(size_t v, size_t to) {
    return (v + to - 1) / to * to;
}

void * dma_alloc(size_t bytes) {
    if (bytes == 0 || bytes > SIZE_MAX - (ALIGN - 1)) return nullptr;
    void * p = std::aligned_alloc(ALIGN, align_up(bytes, ALIGN));
    if (p != nullptr) {
        std::memset(p, 0, align_up(bytes, ALIGN));
    }
    return p;
}

void reconstruct_row(const int32_t * c, size_t n, double activation_scale, const float * channel_scale, float * out,
                     size_t col_stride) {
    for (size_t j = 0; j < n; ++j) {
        out[j * col_stride] =
            static_cast<float>(static_cast<double>(c[j]) * activation_scale * static_cast<double>(channel_scale[j]));
    }
}

bool copy_activation_rows(const npu_matmul_args & args, size_t row_begin, size_t rows, int8_t * dst, size_t lda) {
    const auto &    A     = args.A;
    const uint8_t * codes = A.raw_data();
    if (codes == nullptr || args.K == 0 || A.bits != 4 || A.cols != args.K || A.row_stride_bytes != args.K ||
        lda < args.K || (lda & 1) || !rows || rows > SIZE_MAX / (lda / 2) ||
        row_begin > A.rows || rows > A.rows - row_begin ||
        A.row_stride_bytes == 0 || row_begin + rows > A.raw_size() / A.row_stride_bytes) {
        return false;
    }
    return potal_int4_pack_rows(reinterpret_cast<uint8_t *>(dst), rows * (lda / 2), lda / 2,
        reinterpret_cast<const int8_t *>(codes + row_begin * A.row_stride_bytes),
        A.row_stride_bytes, rows, args.K) == POTAL_OK;
}

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
    if (w->type != GGML_TYPE_MXINT4 || x->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
        return GGML_POTAL_REJECT_TYPE;
    }
    const int64_t k = w->ne[0];
    const int64_t n = w->ne[1];
    const int64_t m = x->ne[1];
    if (k <= 0 || n <= 0 || m <= 0 || k % (int64_t) quants::weight::hp1::block_k != 0 || x->ne[0] != k ||
        op->ne[0] != n || op->ne[1] != m || !is_2d(w) || !is_2d(x) || !is_2d(op)) {
        return GGML_POTAL_REJECT_SHAPE;
    }
    // Whole-block weight rows; the Gemmini quantizers read src1 as a dense I x K tensor.
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

hp1_weight::~hp1_weight() {
    for (void * p : blocks) {
        std::free(p);
    }
}

void hp1_weight::abandon() {
    for (void *& p : blocks) {
        p = nullptr;
    }
}

bool hp1_weight::required_bytes(int64_t n, int64_t k, size_t & bytes) {
    if (n <= 0 || k <= 0 || k % quants::weight::hp1::block_k != 0 ||
        uint64_t(n) > SIZE_MAX - (HW_DIM - 1) || uint64_t(k) > SIZE_MAX) {
        return false;
    }
    const size_t padded = align_up(size_t(n), HW_DIM);
    if (padded > SIZE_MAX - 1 || padded > SIZE_MAX / sizeof(uint32_t) ||
        padded > SIZE_MAX / sizeof(float)) return false;
    const size_t packed = potal_int4_row_bytes(padded);
    if (packed > SIZE_MAX - (ALIGN - 1)) return false;
    const size_t row = align_up(packed, ALIGN);
    const size_t groups = size_t(k) / quants::weight::hp1::block_k;
    if (row > SIZE_MAX / 2 || size_t(k) > SIZE_MAX / row ||
        groups > SIZE_MAX / (padded * sizeof(uint32_t))) return false;
    bytes = 0;
    for (size_t part : { size_t(k) * row, groups * padded * sizeof(uint32_t), padded * sizeof(float) }) {
        if (part > SIZE_MAX - (ALIGN - 1)) return false;
        const size_t rounded = align_up(part, ALIGN);
        if (rounded > SIZE_MAX - bytes) return false;
        bytes += rounded;
    }
    return true;
}

bool hp1_weight::allocate(int64_t n_, int64_t k_) {
    size_t required;
    if (!required_bytes(n_, k_, required)) return false;
    if (n_ <= 0 || k_ <= 0 || k_ % quants::weight::hp1::block_k != 0 ||
        (uint64_t)n_ > SIZE_MAX - 2 * ALIGN) return false;
    n       = (size_t) n_;
    k       = (size_t) k_;
    n_pad   = align_up(n, HW_DIM);
    ldb     = 2 * align_up(potal_int4_row_bytes(n_pad), ALIGN);
    cstride = n_pad;
    const size_t k_blocks = k / quants::weight::hp1::block_k;
    if (k > SIZE_MAX / (ldb / 2) || n_pad > SIZE_MAX / sizeof(float) ||
        cstride > SIZE_MAX / sizeof(uint32_t) ||
        k_blocks > SIZE_MAX / (cstride * sizeof(uint32_t))) return false;

    b        = static_cast<int8_t *>(blocks[0] = dma_alloc(k * (ldb / 2)));
    carriers = static_cast<uint32_t *>(blocks[1] = dma_alloc(k_blocks * cstride * sizeof(uint32_t)));
    scale    = static_cast<float *>(blocks[2] = dma_alloc(n_pad * sizeof(float)));
    return b && carriers && scale;
}

size_t hp1_weight::bytes() const {
    return align_up(k * (ldb / 2), ALIGN) + align_up(k / quants::weight::hp1::block_k * cstride * sizeof(uint32_t), ALIGN) +
           align_up(n_pad * sizeof(float), ALIGN);
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
    own.abandon();
}

bool hp1_operands::allocate_dynamic(const hp1_shape & s) {
    if (s.m <= 0 || s.n <= 0 || s.k <= 0 ||
        (uint64_t)s.m > SIZE_MAX - HW_DIM ||
        (uint64_t)s.n > SIZE_MAX - HW_DIM ||
        (uint64_t)s.k > SIZE_MAX - 2 * ALIGN) return false;
    m_pad = align_up((size_t) s.m, HW_DIM);
    n_pad = align_up((size_t) s.n, HW_DIM);
    lda   = 2 * align_up(potal_int4_row_bytes((size_t) s.k), ALIGN);
    ldc   = n_pad;
    if (m_pad > SIZE_MAX / (lda / 2) || ldc > SIZE_MAX / sizeof(int32_t) ||
        m_pad > SIZE_MAX / (ldc * sizeof(int32_t))) return false;
    a     = static_cast<int8_t *>(blocks[0] = dma_alloc(m_pad * (lda / 2)));
    c     = static_cast<int32_t *>(blocks[1] = dma_alloc(m_pad * ldc * sizeof(int32_t)));
    return a && c;
}

bool hp1_operands::allocate(const hp1_shape & s) {
    if (!own.allocate(s.n, s.k)) {
        return false;
    }
    return allocate(s, own);
}

bool hp1_operands::allocate(const hp1_shape & s, const hp1_weight & w) {
    if (w.n != (size_t) s.n || w.k != (size_t) s.k || w.b == nullptr) {
        return false;
    }
    b        = w.b;
    carriers = w.carriers;
    scale    = w.scale;
    ldb      = w.ldb;
    cstride  = w.cstride;
    return allocate_dynamic(s);
}

bool hp1_operands::allocate(const hp1_shape & s, const hp1_weight & w, const hp1_operands & activation) {
    if (s.m <= 0 || s.n <= 0 || s.k <= 0 ||
        (uint64_t)s.m > SIZE_MAX - HW_DIM || (uint64_t)s.n > SIZE_MAX - HW_DIM ||
        w.n != size_t(s.n) || w.k != size_t(s.k) || !w.b || !activation.a ||
        activation.m_pad < size_t(s.m) || activation.lda < size_t(s.k) || (activation.lda & 1)) {
        return false;
    }
    m_pad = align_up(size_t(s.m), HW_DIM);
    n_pad = align_up(size_t(s.n), HW_DIM);
    ldc = n_pad;
    if (ldc > SIZE_MAX / sizeof(int32_t) || m_pad > SIZE_MAX / (ldc * sizeof(int32_t))) return false;
    a = activation.a;
    lda = activation.lda;
    b = w.b;
    ldb = w.ldb;
    carriers = w.carriers;
    cstride = w.cstride;
    scale = w.scale;
    c = static_cast<int32_t *>(blocks[1] = dma_alloc(m_pad * ldc * sizeof(int32_t)));
    return c != nullptr;
}

bool init_weight_args(const ggml_tensor * src0, npu_matmul_args & args) {
    if (src0 == nullptr || src0->type != GGML_TYPE_MXINT4 || !is_2d(src0) ||
        src0->ne[0] <= 0 || src0->ne[1] <= 0 || src0->ne[0] % QK4_HP != 0 ||
        uint64_t(src0->ne[0]) > SIZE_MAX / sizeof(block_q4_hp1) ||
        uint64_t(src0->ne[1]) > SIZE_MAX / (size_t(src0->ne[0]) / QK4_HP) ||
        src0->data == nullptr || (src0->view_src && src0->view_src->data == nullptr)) return false;
    const size_t row = size_t(src0->ne[0]) / QK4_HP * sizeof(block_q4_hp1);
    if (size_t(src0->ne[1]) > SIZE_MAX / row || src0->nb[0] != sizeof(block_q4_hp1) ||
        (src0->ne[1] != 1 && src0->nb[1] != row)) return false;
    args.J                     = size_t(src0->ne[1]);
    args.K                     = size_t(src0->ne[0]);
    args.weight_format         = npu_matmul_args::weight_format_t::q4_hp1;
    args.q4_hp1_blocks          = static_cast<const block_q4_hp1 *>(ggml::npu::tensor_data(src0));
    args.native_blocks_per_row = args.K / QK4_HP;
    args.native_block_count    = args.J * args.native_blocks_per_row;
    args.native_weight_bytes   = ggml_nbytes(src0);
    args.block_size_k          = QK4_HP;
    args.folding_anchor        = 3;
    return true;
}

bool init_args(const ggml_tensor * src0, ggml_tensor * dst, const hp1_shape & s, const char * model_arch,
               const char * layer, ggml::npu::residual::ResidualRoute residual_route,
               const potal_tile_plan & plan, npu_matmul_args & args) {
    if (s.m <= 0 || !init_weight_args(src0, args) || args.J != size_t(s.n) || args.K != size_t(s.k)) return false;
    args.I                     = (size_t) s.m;
    args.sA                    = args.K;
    args.sC                    = args.J;
    args.residual_route        = residual_route;
    args.f_out                 = static_cast<float *>(dst->data);
    args.col_stride_f_out      = 1;
    args.stride_f_out          = dst->nb[1] / sizeof(float);
    args.model_arch            = model_arch;
    args.matmul_layer          = layer != nullptr ? layer : "";
    // Tile counts of the PoTal plan, as gemmini_set_tile_ws() sets them for Gemmini.
    args.tile_I                = plan.tile_I;
    args.tile_J                = plan.tile_J;
    args.tile_K                = plan.tile_K;
    const auto geometry        = ggml::npu::make_npu_geometry(
        { { args.I, args.J, args.K }, { args.tile_I, args.tile_J, args.tile_K }, HW_DIM });
    if (!geometry.ok()) {
        return false;
    }
    args.activation_rows_per_stripe = geometry.geometry.stripe_rows;
    return args.A.allocate(args.I, args.K, GGML_GEMMINI_ACTIVATION_BITS);
}

bool quantize_activations(const ggml_tensor * src1, npu_matmul_args & args, hp1_operands & ops) {
    if (!ggml::npu::quants::quantize_activation(src1, args)) {
        return false;
    }
    return copy_activation_rows(args, 0, args.I, ops.a, ops.lda);
}

quants::weight::hp1::status unpack_weights(const npu_matmul_args & args, hp1_operands & ops,
                                           quants::weight::hp1::stats *                   stats,
                                           ggml::npu::quants::wreader::WeightReaderStatus * reader) {
    hp1_weight & w = ops.own;
    return quants::weight::hp1::unpack(args, w.b, w.ldb, w.carriers, w.cstride, w.scale, stats, reader);
}

quants::weight::hp1::status prepare_weight(const npu_matmul_args & args, hp1_weight & w,
                                           ggml::npu::quants::wreader::WeightReaderStatus * reader) {
    if (w.n != args.J || w.k != args.K) {
        return quants::weight::hp1::status::invalid;
    }
    return quants::weight::hp1::unpack(args, w.b, w.ldb, w.carriers, w.cstride, w.scale, &w.stats, reader);
}

const char * act_route_name(act_route r) {
    switch (r) {
        case act_route::tensor: return "Tensor";
        case act_route::token:  return "Token";
        case act_route::stripe: return "Stripe";
        case act_route::exsia:  return "ExSIA";
        case act_route::block:  return "Block";
    }
    return "unknown";
}

bool select_route(const npu_matmul_args & args, route_plan & route) {
    switch (args.act_quant.kind()) {
        case act::MetaKind::tensor: route = { act_route::tensor, "Tensor+block", false }; return true;
        case act::MetaKind::token:  route = { act_route::token, "Token+block", false };   return true;
        case act::MetaKind::stripe: route = { act_route::stripe, "Stripe+block", false }; return true;
        case act::MetaKind::exsia:  route = { act_route::exsia, "ExSIA+block", false };   return true;
        case act::MetaKind::block:  route = { act_route::block, "Block+block", true };    return true;
        case act::MetaKind::none:   break;
    }
    return false;
}

route_result execute_route(const route_plan & route, const npu_matmul_args & args, hp1_operands & ops,
                           const tiled_runner & run) {
    route_result r;
    const auto   fail = [&](const char * stage) {
        r.stage = stage;
        return r;
    };
    const auto   run_job = [&](size_t k0, size_t k) {
        r.last = run(job_of(args, ops, k0, k));
        r.jobs++;
        add_work(r.work, r.last.work);
        r.fatal = r.last.fatal;
        return r.last.ok;
    };

    if (!route.k_groups) {
        // Scale constant along K per row: one GEMM over the whole K.
        const std::vector<float> row_scale = act::activation_scales(args, args.I);
        if (row_scale.size() != args.I) {
            return fail("activation_scales");
        }
        if (!run_job(0, args.K)) {
            return fail("potal_gemm");
        }
        for (size_t m = 0; m < args.I; ++m) {
            reconstruct_row(ops.c + m * ops.ldc, args.J, static_cast<double>(row_scale[m]), ops.scale,
                            args.f_out + m * args.stride_f_out, args.col_stride_f_out);
        }
        r.ok = true;
        return r;
    }

    // Block+block: the activation scale changes every K group, so each group is
    // its own GEMM and the groups are summed after scaling (never in the ACC).
    const act::ActivationMetadataView view(args, args.activation_row_offset, args.activation_row_offset + args.I);
    if (!view.valid()) {
        return fail("activation_metadata");
    }
    std::vector<double> acc;
    std::vector<float>  group_scale(args.I);
    try {
        acc.assign(args.I * args.J, 0.0);
    } catch (const std::bad_alloc &) {
        return fail("alloc");
    }
    const size_t group = quants::weight::hp1::block_k;
    for (size_t k0 = 0; k0 < args.K; k0 += group) {
        for (size_t m = 0; m < args.I; ++m) {
            if (!view.scale(m, k0, group_scale[m])) {
                return fail("activation_metadata");
            }
        }
        if (!run_job(k0, group)) {
            return fail("potal_gemm");
        }
        r.k_groups++;
        for (size_t m = 0; m < args.I; ++m) {
            const int32_t * c   = ops.c + m * ops.ldc;
            double *        row = acc.data() + m * args.J;
            const double    s_m = static_cast<double>(group_scale[m]);
            for (size_t n = 0; n < args.J; ++n) {
                row[n] += static_cast<double>(c[n]) * s_m * static_cast<double>(ops.scale[n]);
            }
        }
    }
    for (size_t m = 0; m < args.I; ++m) {
        float * out = args.f_out + m * args.stride_f_out;
        for (size_t n = 0; n < args.J; ++n) {
            out[n * args.col_stride_f_out] = static_cast<float>(acc[m * args.J + n]);
        }
    }
    r.ok = true;
    return r;
}

compensation_result apply_compensation(const npu_matmul_args & args) {
    compensation_result r;
#if GGML_GEMMINI_ENABLE_RMD
    namespace rmd      = ggml::npu::rmd;
    namespace residual = ggml::npu::residual;
    const auto & packets = ggml::npu::quants::activation_rmd_packets(args);
    for (const auto & packet : packets) {
        if (packet != nullptr) {
            r.payloads++;
            r.ok     = false;
            r.detail = "ws_packet residual payload: the Gemmini WS residual executor is not part of PoTal";
            return r;
        }
    }
    for (const auto & payload : ggml::npu::quants::activation_direct_residuals(args)) {
        if (payload == nullptr) {
            continue;
        }
        r.payloads++;
        size_t row_end = 0;
        if (__builtin_add_overflow(payload->row_begin, payload->row_count, &row_end)) {
            r.ok     = false;
            r.detail = "residual payload rows overflow";
            return r;
        }
        // Same calls, in the same order, as the Gemmini dense cpu_direct route.
        rmd::Correction                  correction = rmd::BlockScaledInt64Correction{};
        residual::DirectExecutionMetrics metrics;
        if (const auto * meta = std::get_if<ggml::npu::quants::act::exsia::Meta>(&args.act_quant.storage())) {
            metrics.run_id = meta->run_id;
        }
        rmd::RmdStatus status = residual::execute_direct_stripe(args, *payload, correction, &metrics);
        if (status == rmd::RmdStatus::success) {
            size_t nonzero = 0;
            status         = rmd::merge_rmd_correction(args, payload->row_begin, row_end, correction, &nonzero);
            r.nonzero += nonzero;
        }
        if (status != rmd::RmdStatus::success) {
            r.ok     = false;
            r.detail = rmd::rmd_status_message(status);
            return r;
        }
        r.executed++;
        r.residuals += metrics.event_count;
    }
#else
    (void) args;
#endif
    return r;
}

void reference_gemm_i32(const int8_t * a, size_t lda, const int8_t * b, size_t ldb, const uint32_t * carriers,
                        size_t cstride, size_t block_k, size_t M, size_t N, size_t K, int32_t * c, size_t ldc) {
    namespace hp1 = ggml::npu::quants::hp1; // PoTalSCU + Sat32 ACC
    const size_t passes = (K + HW_DIM - 1) / HW_DIM;
    for (size_t m = 0; m < M; ++m) {
        for (size_t n = 0; n < N; ++n) {
            int32_t acc = 0;
            for (size_t p = 0; p < passes; ++p) {
                int32_t partial = 0; // |partial| <= NPU_D * 128 * 128 fits easily
                for (size_t k = p * HW_DIM; k < K && k < (p + 1) * HW_DIM; ++k) {
                    partial += static_cast<int32_t>(potal_int4_get(
                        reinterpret_cast<const uint8_t *>(a) + m * (lda / 2), k)) *
                        potal_int4_get(reinterpret_cast<const uint8_t *>(b) + k * (ldb / 2), n);
                }
                const int32_t v = hp1::apply_validated(partial, carriers[(p * HW_DIM / block_k) * cstride + n]);
                acc             = p == 0 ? v : hp1::accumulate(acc, v);
            }
            c[m * ldc + n] = acc;
        }
    }
}

} // namespace ggml::potal
