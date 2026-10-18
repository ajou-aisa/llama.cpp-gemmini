#include "dense.hpp"

#include "ggml-backend.h"
#include "potal_hw_params.h"

#include "ggml-gemmini-geometry.hpp"
#include "quants/act/dispatch.hpp"
#include "quants/act/quantize.hpp"
#include "quants/common/hp1_scu.hpp"
#include "quants/common/tensor_util.hpp"
#include "residual/direct/direct-executor.hpp"
#include "residual/rmd/rmd-compose.hpp"

#include <cstdlib>
#include <cstring>
#include <variant>

static_assert(ggml::gemmini::quants::act::block::kGroupSize == ggml::potal::quants::weight::hp1::block_k,
              "Block+block needs the activation group to be the HP1 scale block");

namespace ggml::potal {

namespace {

namespace act = ggml::gemmini::quants::act;

constexpr size_t HW_DIM = POTAL_HW_DIM; // not DIM: <gemmini_params.h> defines that macro
constexpr size_t ALIGN  = POTAL_HW_MAX_BYTES;

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
tiled_job job_of(const ggml_gemmini_args_t & args, const hp1_operands & ops, size_t k0, size_t k) {
    const size_t block = k0 / quants::weight::hp1::block_k;
    return {
        args.I, args.J, k,
        ops.a + k0, ops.lda, ops.b + k0 * ops.ldb, ops.ldb,
        ops.carriers + block * ops.cstride, ops.cstride, quants::weight::hp1::block_k,
        ops.c, ops.ldc,
    };
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

bool hp1_weight::allocate(int64_t n_, int64_t k_) {
    n       = (size_t) n_;
    k       = (size_t) k_;
    n_pad   = round_up(n, HW_DIM);
    ldb     = round_up(n_pad, ALIGN);
    cstride = n_pad;
    const size_t k_blocks = k / quants::weight::hp1::block_k;

    b        = static_cast<int8_t *>(blocks[0] = zeroed(k * ldb));
    carriers = static_cast<uint32_t *>(blocks[1] = zeroed(k_blocks * cstride * sizeof(uint32_t)));
    scale    = static_cast<float *>(blocks[2] = zeroed(n_pad * sizeof(float)));
    return b && carriers && scale;
}

size_t hp1_weight::bytes() const {
    return round_up(k * ldb, ALIGN) + round_up(k / quants::weight::hp1::block_k * cstride * sizeof(uint32_t), ALIGN) +
           round_up(n_pad * sizeof(float), ALIGN);
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
    m_pad = round_up((size_t) s.m, HW_DIM);
    n_pad = round_up((size_t) s.n, HW_DIM);
    lda   = round_up((size_t) s.k, ALIGN);
    ldc   = n_pad;
    a     = static_cast<int8_t *>(blocks[0] = zeroed(m_pad * lda));
    c     = static_cast<int32_t *>(blocks[1] = zeroed(m_pad * ldc * sizeof(int32_t)));
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

bool init_args(const ggml_tensor * src0, ggml_tensor * dst, const hp1_shape & s, const char * model_arch,
               const char * layer, ggml::gemmini::residual::ResidualRoute residual_route,
               const potal_tile_plan & plan, ggml_gemmini_args_t & args) {
    args.I                     = (size_t) s.m;
    args.J                     = (size_t) s.n;
    args.K                     = (size_t) s.k;
    args.sA                    = args.K;
    args.sC                    = args.J;
    args.residual_route        = residual_route;
    args.weight_format         = ggml_gemmini_args_t::im2p_weight_format_t::q8_hp1;
    args.q8_hp1_blocks         = static_cast<const block_q8_hp1 *>(ggml::gemmini::tensor_data(src0));
    args.q8_hp1_blocks_per_row = args.K / QK8_HP;
    args.q8_hp1_block_count    = args.J * args.q8_hp1_blocks_per_row;
    args.native_weight_bytes   = ggml_nbytes(src0);
    args.block_size_k          = QK8_HP;
    args.f_out                 = static_cast<float *>(dst->data);
    args.col_stride_f_out      = 1;
    args.stride_f_out          = dst->nb[1] / sizeof(float);
    args.model_arch            = model_arch;
    args.matmul_layer          = layer != nullptr ? layer : "";
    // Tile counts of the PoTal plan, as gemmini_set_tile_ws() sets them for Gemmini.
    args.tile_I                = plan.tile_I;
    args.tile_J                = plan.tile_J;
    args.tile_K                = plan.tile_K;
    const auto geometry        = ggml::gemmini::make_gemmini_geometry(
        { { args.I, args.J, args.K }, { args.tile_I, args.tile_J, args.tile_K }, HW_DIM });
    if (!geometry.ok()) {
        return false;
    }
    args.activation_rows_per_stripe = geometry.geometry.stripe_rows;
    return args.A.allocate(args.I, args.K, GGML_GEMMINI_ACTIVATION_BITS);
}

bool quantize_activations(const ggml_tensor * src1, ggml_gemmini_args_t & args, hp1_operands & ops) {
    if (!ggml::gemmini::quants::quantize_activation(src1, args)) {
        return false;
    }
    const int8_t * codes = args.A;
    if (codes == nullptr) {
        return false;
    }
    for (size_t m = 0; m < args.I; ++m) {
        std::memcpy(ops.a + m * ops.lda, codes + m * args.K, args.K);
    }
    return true;
}

quants::weight::hp1::status unpack_weights(const ggml_gemmini_args_t & args, hp1_operands & ops,
                                           quants::weight::hp1::stats *                   stats,
                                           ggml::gemmini::quants::wreader::WeightReaderStatus * reader) {
    hp1_weight & w = ops.own;
    return quants::weight::hp1::unpack(args, w.b, w.ldb, w.carriers, w.cstride, w.scale, stats, reader);
}

quants::weight::hp1::status prepare_weight(const ggml_gemmini_args_t & args, hp1_weight & w,
                                           ggml::gemmini::quants::wreader::WeightReaderStatus * reader) {
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

bool select_route(const ggml_gemmini_args_t & args, route_plan & route) {
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

route_result execute_route(const route_plan & route, const ggml_gemmini_args_t & args, hp1_operands & ops,
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
            float *         out = args.f_out + m * args.stride_f_out;
            const int32_t * c   = ops.c + m * ops.ldc;
            const double    s_m = static_cast<double>(row_scale[m]);
            for (size_t n = 0; n < args.J; ++n) {
                out[n * args.col_stride_f_out] =
                    static_cast<float>(static_cast<double>(c[n]) * s_m * static_cast<double>(ops.scale[n]));
            }
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

compensation_result apply_compensation(const ggml_gemmini_args_t & args) {
    compensation_result r;
#if GGML_GEMMINI_ENABLE_RMD
    namespace rmd      = ggml::gemmini::rmd;
    namespace residual = ggml::gemmini::residual;
    const auto & packets = ggml::gemmini::quants::activation_rmd_packets(args);
    for (const auto & packet : packets) {
        if (packet != nullptr) {
            r.payloads++;
            r.ok     = false;
            r.detail = "ws_packet residual payload: the Gemmini WS residual executor is not part of PoTal";
            return r;
        }
    }
    for (const auto & payload : ggml::gemmini::quants::activation_direct_residuals(args)) {
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
        if (const auto * meta = std::get_if<ggml::gemmini::quants::act::exsia::Meta>(&args.act_quant.storage())) {
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
    namespace hp1 = ggml::gemmini::quants::hp1; // PoTalSCU + Sat32 ACC
    const size_t passes = (K + HW_DIM - 1) / HW_DIM;
    for (size_t m = 0; m < M; ++m) {
        for (size_t n = 0; n < N; ++n) {
            int32_t acc = 0;
            for (size_t p = 0; p < passes; ++p) {
                int32_t partial = 0; // |partial| <= DIM * 128 * 128 fits easily
                for (size_t k = p * HW_DIM; k < K && k < (p + 1) * HW_DIM; ++k) {
                    partial += static_cast<int32_t>(a[m * lda + k]) * b[k * ldb + n];
                }
                const int32_t v = hp1::apply_validated(partial, carriers[(p * HW_DIM / block_k) * cstride + n]);
                acc             = p == 0 ? v : hp1::accumulate(acc, v);
            }
            c[m * ldc + n] = acc;
        }
    }
}

} // namespace ggml::potal
