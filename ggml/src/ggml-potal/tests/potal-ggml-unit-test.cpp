/*
 * PoTal GGML backend checks without accelerator access (host and RISC-V target).
 *
 *   A registry   POTAL reg/device found by name, one ACCEL device, enumerated
 *                once and before CPU, no GEMMINI device in this build
 *   B backend    name, GUID distinct from Gemmini and CPU, host buffer type, props
 *   C contract   supports_op accepts I8 M=N=16 K=16|32 and rejects other shapes,
 *                types, layouts, ops and tags with the expected reason; counters
 *   D packing    pack_a/pack_b/unpack_c against the GGML storage order, and the
 *                packed product against a CPU-backend F32 MUL_MAT golden
 *   E refusal    graphs outside the contract fail before any hardware request;
 *                off RISC-V a valid graph fails with POTAL_ERR_NO_HARDWARE and a
 *                NaN dst (no fake hardware success)
 *   F scheduler  POTAL + CPU: I8 MUL_MAT placed on POTAL, F32 ops on CPU (and
 *                computed there), an op neither backend supports is not placed
 *   G buffer     POTAL buffer type: own identity (not CPU/GEMMINI), host memory
 *                the CPU backend accepts, owned allocation, zero size, tensor
 *                I/O and copies, host-pointer and read-only mmap aliases that
 *                are never freed by the buffer, per-buffer generations, and the
 *                Q8_HP1 ops placed by llama.cpp (GET_ROWS on CPU, MUL_MAT on POTAL)
 *
 * The accelerator is never used: on RISC-V the one graph POTAL could execute
 * (E3) is skipped; potal-ggml-smoke covers hardware execution.
 * Output: "PASS <name>" / "FAIL <name>: ...", final "POTAL_GGML_UNIT_TEST PASS|FAIL".
 */
#include "ggml-cpp.h"
#include "ggml-potal-buffer.hpp"
#include "ggml-potal.h"
#include "matmul/pack.hpp"
#include "matmul/weight_cache.hpp"
#include "ops.hpp"
#include "potal.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

namespace {

using ggml::potal::mul_mat_shape;

constexpr int64_t TILE = ggml::potal::TILE;

int g_failures = 0;

void check(const std::string & name, bool ok, const std::string & detail = std::string()) {
    if (ok) {
        std::printf("PASS %s\n", name.c_str());
    } else {
        std::printf("FAIL %s%s%s\n", name.c_str(), detail.empty() ? "" : ": ", detail.c_str());
        g_failures++;
    }
}

ggml_context_ptr make_context(bool alloc) {
    const ggml_init_params params = { /* .mem_size = */ 4u << 20, /* .mem_buffer = */ nullptr, /* .no_alloc = */ !alloc };
    return ggml_context_ptr(ggml_init(params));
}

ggml_backend_potal_stats stats_of(ggml_backend_t potal) {
    ggml_backend_potal_stats st = {};
    ggml_backend_potal_get_stats(potal, &st);
    return st;
}

// GGML storage values: signed and different under any transposition of the
// indices, so a swapped row/column changes the result.
int8_t act_at(int64_t m, int64_t k) {
    return static_cast<int8_t>(((m * 37 + k * 11 + 5) & 0xff) - 128);
}

int8_t wt_at(int64_t n, int64_t k) {
    return static_cast<int8_t>(((n * 23 + k * 59 + 17) & 0xff) - 128);
}

// Element (row, col) of a 2-D tensor, addressed through nb[] like GGML does.
template <typename T> T & at(const ggml_tensor * t, int64_t row, int64_t col) {
    return *reinterpret_cast<T *>(static_cast<char *>(t->data) + col * t->nb[0] + row * t->nb[1]);
}

template <typename T> void fill(const ggml_tensor * t, int8_t (*value)(int64_t, int64_t)) {
    for (int64_t row = 0; row < t->ne[1]; ++row) {
        for (int64_t col = 0; col < t->ne[0]; ++col) {
            at<T>(t, row, col) = static_cast<T>(value(row, col));
        }
    }
}

// dst[m][n] = sum_k act[m][k] * wt[n][k], straight from the GGML definition.
std::vector<int64_t> golden(int64_t k) {
    std::vector<int64_t> out(TILE * TILE, 0);
    for (int64_t m = 0; m < TILE; ++m) {
        for (int64_t n = 0; n < TILE; ++n) {
            for (int64_t kk = 0; kk < k; ++kk) {
                out[m * TILE + n] += static_cast<int64_t>(act_at(m, kk)) * wt_at(n, kk);
            }
        }
    }
    return out;
}

bool all_nan(const ggml_tensor * t) {
    const float * p = static_cast<const float *>(t->data);
    for (int64_t i = 0; i < ggml_nelements(t); ++i) {
        if (!std::isnan(p[i])) {
            return false;
        }
    }
    return true;
}

// ---- A. registry -----------------------------------------------------------

void test_registry() {
    ggml_backend_reg_t reg = ggml_backend_reg_by_name("POTAL");
    check("A reg POTAL registered", reg != nullptr);
    ggml_backend_dev_t dev = ggml_backend_dev_by_name("POTAL");
    check("A device POTAL registered", dev != nullptr);
    if (reg == nullptr || dev == nullptr) {
        return;
    }
    check("A reg is ggml_backend_potal_reg()", reg == ggml_backend_potal_reg());
    check("A reg name POTAL", std::strcmp(ggml_backend_reg_name(reg), "POTAL") == 0);
    check("A reg has one device", ggml_backend_reg_dev_count(reg) == 1);
    check("A device belongs to the POTAL reg", ggml_backend_dev_backend_reg(dev) == reg);
    check("A device type ACCEL", ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_ACCEL);

    const std::string desc = ggml_backend_dev_description(dev);
    std::printf("  description: %s\n", desc.c_str());
    check("A description states the hardware format",
          desc.find("A8W8") != std::string::npos && desc.find("Sat32") != std::string::npos &&
              desc.find("DIM16") != std::string::npos && desc.find("I8 x I8 -> F32") != std::string::npos,
          desc);

    int    potal_seen  = 0;
    size_t potal_index = SIZE_MAX;
    size_t cpu_index   = SIZE_MAX;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        std::printf("  device[%zu] name=%s type=%d\n", i, ggml_backend_dev_name(d), (int) ggml_backend_dev_type(d));
        if (d == dev) {
            potal_seen++;
            potal_index = i;
        }
        if (ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_CPU && cpu_index == SIZE_MAX) {
            cpu_index = i;
        }
    }
    check("A POTAL enumerated exactly once", potal_seen == 1);
    check("A CPU device enumerated after POTAL", cpu_index != SIZE_MAX && potal_index < cpu_index);
    check("A no GEMMINI device in a GGML_POTAL build", ggml_backend_dev_by_name("GEMMINI") == nullptr);
}

// ---- B. backend instance ---------------------------------------------------

void test_backend(ggml_backend_t potal, ggml_backend_t cpu) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_name("POTAL");
    if (dev == nullptr) {
        check("B POTAL device", false, "not registered");
        return;
    }
    ggml_backend_ptr via_device(ggml_backend_dev_init(dev, nullptr));
    check("B ggml_backend_dev_init(POTAL)", via_device != nullptr);
    if (!via_device) {
        return;
    }
    check("B backend name POTAL", std::strcmp(ggml_backend_name(potal), "POTAL") == 0);
    check("B ggml_backend_is_potal", ggml_backend_is_potal(potal) && ggml_backend_is_potal(via_device.get()));
    check("B CPU backend is not POTAL", !ggml_backend_is_potal(cpu));
    check("B backend device is the POTAL device", ggml_backend_get_device(potal) == dev);

    // ggml_backend_gemmini_guid() in ggml-gemmini.cpp.
    static ggml_guid gemmini_guid = { 0x10, 0xa8, 0xae, 0xf4, 0xc0, 0x1e, 0x61, 0x97,
                                      0x8f, 0xeb, 0x33, 0x04, 0xa1, 0x33, 0x51, 0x2d };
    check("B GUID differs from Gemmini", !ggml_guid_matches(ggml_backend_guid(potal), &gemmini_guid));
    check("B GUID differs from CPU", !ggml_guid_matches(ggml_backend_guid(potal), ggml_backend_guid(cpu)));
    check("B GUID shared by POTAL instances",
          ggml_guid_matches(ggml_backend_guid(potal), ggml_backend_guid(via_device.get())));

    check("B default buffer type is host memory",
          ggml_backend_buft_is_host(ggml_backend_get_default_buffer_type(potal)));
    ggml_backend_dev_props props;
    ggml_backend_dev_get_props(dev, &props);
    check("B props: synchronous, no events, no pinned host buffer type, host-pointer buffers",
          !props.caps.async && !props.caps.events && !props.caps.host_buffer && props.caps.buffer_from_host_ptr);

    ggml_backend_potal_stats st = {};
    check("B stats of a fresh backend",
          ggml_backend_potal_get_stats(potal, &st) && st.mul_mat_nodes == 0 && st.hw_requests == 0 &&
              st.gemm_issued == 0);
    check("B stats refused for a CPU backend", !ggml_backend_potal_get_stats(cpu, &st));
    check("B no issuer thread before a hardware request", !st.issuer_ready && st.issuer_cpu == -1);
}

// ---- C. supports_op contract -----------------------------------------------

void test_contract(ggml_backend_t potal) {
    ggml_backend_dev_t dev = ggml_backend_get_device(potal);
    ggml_context_ptr   ctx = make_context(false);
    ggml_context *     c   = ctx.get();

    auto matrix = [c](ggml_type type, int64_t cols, int64_t rows) { return ggml_new_tensor_2d(c, type, cols, rows); };
    // ggml_mul_mat(weights ne = {K, N}, activations ne = {K, M}).
    auto mul_mat = [c, &matrix](ggml_type wt, ggml_type xt, int64_t k, int64_t n, int64_t m) {
        return ggml_mul_mat(c, matrix(wt, k, n), matrix(xt, k, m));
    };

    ggml_tensor * w16  = matrix(GGML_TYPE_I8, 16, 16);
    ggml_tensor * x16  = matrix(GGML_TYPE_I8, 16, 16);
    ggml_tensor * wide = matrix(GGML_TYPE_I8, 32, 16); // rows of 32: a 16-wide view is strided
    ggml_tensor * tall = matrix(GGML_TYPE_I8, 16, 32); // rows 16..31 form a contiguous view

    ggml_tensor * tagged = mul_mat(GGML_TYPE_I8, GGML_TYPE_I8, 16, 16, 16);
    tagged->op_params[1] = 0x5054514b; // GGML_GEMMINI_ATTENTION_QK

    ggml_tensor * no_sources = matrix(GGML_TYPE_F32, 16, 16);
    no_sources->op           = GGML_OP_MUL_MAT;

    ggml_tensor * q = ggml_new_tensor_4d(c, GGML_TYPE_F32, 16, 16, 1, 1);
    ggml_tensor * k = ggml_new_tensor_4d(c, GGML_TYPE_F16, 16, 16, 1, 1);
    ggml_tensor * v = ggml_new_tensor_4d(c, GGML_TYPE_F16, 16, 16, 1, 1);

    struct contract_case {
        const char *      name;
        ggml_tensor *     op;
        ggml_potal_reject want;
    };

    const contract_case cases[] = {
        { "I8 M16 N16 K16", mul_mat(GGML_TYPE_I8, GGML_TYPE_I8, 16, 16, 16), GGML_POTAL_REJECT_NONE },
        { "I8 M16 N16 K32", mul_mat(GGML_TYPE_I8, GGML_TYPE_I8, 32, 16, 16), GGML_POTAL_REJECT_NONE },
        { "contiguous src1 view at an offset",
          ggml_mul_mat(c, w16, ggml_view_2d(c, tall, 16, 16, tall->nb[1], 16 * tall->nb[1])),
          GGML_POTAL_REJECT_NONE },
        { "VIEW of host data", ggml_view_2d(c, tall, 16, 8, tall->nb[1], 0), GGML_POTAL_REJECT_NONE },
        { "RESHAPE of host data", ggml_reshape_2d(c, x16, 32, 8), GGML_POTAL_REJECT_NONE },
        { "M8", mul_mat(GGML_TYPE_I8, GGML_TYPE_I8, 16, 16, 8), GGML_POTAL_REJECT_SHAPE },
        { "M32", mul_mat(GGML_TYPE_I8, GGML_TYPE_I8, 16, 16, 32), GGML_POTAL_REJECT_SHAPE },
        { "N32", mul_mat(GGML_TYPE_I8, GGML_TYPE_I8, 16, 32, 16), GGML_POTAL_REJECT_SHAPE },
        { "K48", mul_mat(GGML_TYPE_I8, GGML_TYPE_I8, 48, 16, 16), GGML_POTAL_REJECT_SHAPE },
        { "K8", mul_mat(GGML_TYPE_I8, GGML_TYPE_I8, 8, 16, 16), GGML_POTAL_REJECT_SHAPE },
        { "batched src1 (ne2=2)", ggml_mul_mat(c, w16, ggml_new_tensor_3d(c, GGML_TYPE_I8, 16, 16, 2)),
          GGML_POTAL_REJECT_SHAPE },
        { "F32 x F32", mul_mat(GGML_TYPE_F32, GGML_TYPE_F32, 16, 16, 16), GGML_POTAL_REJECT_TYPE },
        { "F16 weights", mul_mat(GGML_TYPE_F16, GGML_TYPE_F32, 16, 16, 16), GGML_POTAL_REJECT_TYPE },
        { "F16 activations", mul_mat(GGML_TYPE_I8, GGML_TYPE_F16, 16, 16, 16), GGML_POTAL_REJECT_TYPE },
        { "I8 weights, F32 activations", mul_mat(GGML_TYPE_I8, GGML_TYPE_F32, 16, 16, 16), GGML_POTAL_REJECT_TYPE },
        { "Q4_0 weights", mul_mat(GGML_TYPE_Q4_0, GGML_TYPE_F32, 32, 16, 16), GGML_POTAL_REJECT_TYPE },
        { "Q8_0 weights", mul_mat(GGML_TYPE_Q8_0, GGML_TYPE_F32, 32, 16, 16), GGML_POTAL_REJECT_TYPE },
        { "transposed src1", ggml_mul_mat(c, w16, ggml_transpose(c, matrix(GGML_TYPE_I8, 16, 16))),
          GGML_POTAL_REJECT_LAYOUT },
        { "permuted src1", ggml_mul_mat(c, w16, ggml_permute(c, matrix(GGML_TYPE_I8, 16, 16), 1, 0, 2, 3)),
          GGML_POTAL_REJECT_LAYOUT },
        { "strided src1 view (nb1=32)", ggml_mul_mat(c, w16, ggml_view_2d(c, wide, 16, 16, wide->nb[1], 0)),
          GGML_POTAL_REJECT_LAYOUT },
        { "strided src0 view (nb1=32)",
          ggml_mul_mat(c, ggml_view_2d(c, matrix(GGML_TYPE_I8, 32, 16), 16, 16, wide->nb[1], 0), x16),
          GGML_POTAL_REJECT_LAYOUT },
        { "attention role rejects I8 operand types", tagged, GGML_POTAL_REJECT_TYPE },
        { "bias ADD", ggml_add(c, matrix(GGML_TYPE_F32, 16, 16), matrix(GGML_TYPE_F32, 16, 1)),
          GGML_POTAL_REJECT_OP },
        { "SOFT_MAX", ggml_soft_max(c, matrix(GGML_TYPE_F32, 16, 16)), GGML_POTAL_REJECT_OP },
        { "FLASH_ATTN_EXT", ggml_flash_attn_ext(c, q, k, v, nullptr, 1.0f, 0.0f, 0.0f), GGML_POTAL_REJECT_OP },
        { "MUL_MAT without sources", no_sources, GGML_POTAL_REJECT_OP },
    };

    const ggml_backend_potal_stats before = stats_of(potal);
    uint64_t                       want_accepted                          = 0;
    uint64_t                       want_rejected[GGML_POTAL_REJECT_COUNT] = {};
    for (const contract_case & tc : cases) {
        const ggml_potal_reject reason    = ggml::potal::check_op(tc.op);
        const bool              supported = ggml_backend_dev_supports_op(dev, tc.op);
        if (tc.want == GGML_POTAL_REJECT_NONE) {
            want_accepted++;
        } else {
            want_rejected[tc.want]++;
        }
        check(std::string("C ") + tc.name + (tc.want == GGML_POTAL_REJECT_NONE ? " accepted" : " rejected"),
              reason == tc.want && supported == (tc.want == GGML_POTAL_REJECT_NONE),
              std::string("reason=") + ggml_potal_reject_name(reason) + " want=" + ggml_potal_reject_name(tc.want) +
                  " supports_op=" + (supported ? "1" : "0"));
    }

    const ggml_backend_potal_stats after   = stats_of(potal);
    bool                           counted = after.supports_op_accepted - before.supports_op_accepted == want_accepted;
    for (int r = 1; r < GGML_POTAL_REJECT_COUNT; ++r) {
        const uint64_t delta = after.supports_op_rejected[r] - before.supports_op_rejected[r];
        std::printf("  supports_op rejected reason=%s count=%llu\n",
                    ggml_potal_reject_name(static_cast<ggml_potal_reject>(r)), (unsigned long long) delta);
        counted = counted && delta == want_rejected[r];
    }
    check("C supports_op counters by reason", counted);

    // Compute-time contract: everything above was unallocated.
    ggml_context_ptr data_ctx = make_context(true);
    ggml_tensor *    y = ggml_mul_mat(data_ctx.get(), ggml_new_tensor_2d(data_ctx.get(), GGML_TYPE_I8, 16, 16),
                                      ggml_new_tensor_2d(data_ctx.get(), GGML_TYPE_I8, 16, 16));
    mul_mat_shape    s = {};
    check("C compute contract refuses unallocated tensors",
          ggml::potal::check_mul_mat(cases[0].op, true, &s) == GGML_POTAL_REJECT_MEMORY);
    check("C compute contract accepts allocated tensors",
          ggml::potal::check_mul_mat(y, true, &s) == GGML_POTAL_REJECT_NONE && s.m == 16 && s.n == 16 && s.k == 16);
}

// ---- D. packing ------------------------------------------------------------

void test_packing(int64_t k, ggml_backend_t cpu) {
    const std::string tag = "D K" + std::to_string(k);
    ggml_context_ptr  ctx = make_context(true);
    ggml_context *    c   = ctx.get();

    ggml_tensor * w = ggml_new_tensor_2d(c, GGML_TYPE_I8, k, TILE);
    ggml_tensor * x = ggml_new_tensor_2d(c, GGML_TYPE_I8, k, TILE);
    fill<int8_t>(w, wt_at);
    fill<int8_t>(x, act_at);
    ggml_tensor * y = ggml_mul_mat(c, w, x);

    mul_mat_shape s = {};
    check(tag + " contract", ggml::potal::check_mul_mat(y, true, &s) == GGML_POTAL_REJECT_NONE && s.m == TILE &&
                                 s.n == TILE && s.k == k);

    std::vector<int8_t> a(TILE * k);
    std::vector<int8_t> b(k * TILE);
    ggml::potal::pack_a(x, s, a.data());
    ggml::potal::pack_b(w, s, b.data());
    bool a_ok = true;
    bool b_ok = true;
    for (int64_t kk = 0; kk < k; ++kk) {
        for (int64_t i = 0; i < TILE; ++i) {
            a_ok = a_ok && a[i * k + kk] == act_at(i, kk);
            b_ok = b_ok && b[kk * TILE + i] == wt_at(i, kk);
        }
    }
    check(tag + " pack_a: A[m][k] = src1 row m, element k", a_ok);
    check(tag + " pack_b: B[k][n] = src0 row n, element k (transposed)", b_ok);

    // What the accelerator computes from the packed operands with x1 carriers.
    std::vector<int32_t> product(TILE * TILE, 0);
    for (int64_t m = 0; m < TILE; ++m) {
        for (int64_t n = 0; n < TILE; ++n) {
            for (int64_t kk = 0; kk < k; ++kk) {
                product[m * TILE + n] += static_cast<int32_t>(a[m * k + kk]) * b[kk * TILE + n];
            }
        }
    }
    ggml::potal::unpack_c(product.data(), s, y);

    // Independent golden: the CPU backend's F32 MUL_MAT of the same values
    // (integer sums below 2^24 are exact in F32).
    ggml_tensor * wf = ggml_new_tensor_2d(c, GGML_TYPE_F32, k, TILE);
    ggml_tensor * xf = ggml_new_tensor_2d(c, GGML_TYPE_F32, k, TILE);
    fill<float>(wf, wt_at);
    fill<float>(xf, act_at);
    ggml_tensor * yf = ggml_mul_mat(c, wf, xf);
    ggml_cgraph * gf = ggml_new_graph(c);
    ggml_build_forward_expand(gf, yf);
    const bool computed = ggml_backend_graph_compute(cpu, gf) == GGML_STATUS_SUCCESS;

    const std::vector<int64_t> want       = golden(k);
    int                        mismatches = 0;
    for (int64_t m = 0; m < TILE; ++m) {
        for (int64_t n = 0; n < TILE; ++n) {
            const float got = at<float>(y, m, n);
            if (got != at<float>(yf, m, n) || got != static_cast<float>(want[m * TILE + n])) {
                if (mismatches++ < 4) {
                    std::printf("  %s dst[m=%lld][n=%lld] packed=%.1f cpu=%.1f golden=%lld\n", tag.c_str(),
                                (long long) m, (long long) n, got, at<float>(yf, m, n),
                                (long long) want[m * TILE + n]);
                }
            }
        }
    }
    check(tag + " pack -> PoTal product -> unpack == CPU backend MUL_MAT == golden", computed && mismatches == 0,
          "computed=" + std::to_string(computed) + " mismatches=" + std::to_string(mismatches));
}

void test_unpack_extremes() {
    ggml_context_ptr ctx = make_context(true);
    ggml_tensor *    dst = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, TILE, TILE); // ne = {N, M}
    const int32_t    values[] = { 524288, -524288, -1, 0, 1, 16384, -16256, 123457 };
    std::vector<int32_t> c(TILE * TILE);
    for (size_t i = 0; i < c.size(); ++i) {
        c[i] = values[(i * 3) % 8] + static_cast<int32_t>(i % 5);
    }
    const mul_mat_shape s = { TILE, TILE, 32 };
    ggml::potal::unpack_c(c.data(), s, dst);
    bool ok = true;
    for (int64_t m = 0; m < TILE; ++m) {
        for (int64_t n = 0; n < TILE; ++n) {
            ok = ok && at<float>(dst, m, n) == static_cast<float>(c[m * TILE + n]);
        }
    }
    check("D unpack_c: dst row m, element n = (float) C[m][n], exact at +/-2^19", ok);
}

void test_view_packing() {
    ggml_context_ptr ctx  = make_context(true);
    ggml_context *   c    = ctx.get();
    ggml_tensor *    tall = ggml_new_tensor_2d(c, GGML_TYPE_I8, 16, 2 * TILE);
    ggml_tensor *    w    = ggml_new_tensor_2d(c, GGML_TYPE_I8, 16, TILE);
    fill<int8_t>(tall, act_at);
    fill<int8_t>(w, wt_at);
    ggml_tensor * rows = ggml_view_2d(c, tall, 16, TILE, tall->nb[1], TILE * tall->nb[1]); // rows 16..31
    ggml_tensor * y    = ggml_mul_mat(c, w, rows);

    mul_mat_shape s = {};
    check("D contiguous view at an offset accepted", ggml::potal::check_mul_mat(y, true, &s) == GGML_POTAL_REJECT_NONE);
    std::vector<int8_t> a(TILE * 16);
    ggml::potal::pack_a(rows, s, a.data());
    bool ok = true;
    for (int64_t m = 0; m < TILE; ++m) {
        for (int64_t kk = 0; kk < 16; ++kk) {
            ok = ok && a[m * 16 + kk] == act_at(m + TILE, kk);
        }
    }
    check("D pack_a reads the view's own rows", ok);
}

// ---- E. refusal without hardware -------------------------------------------

void test_refusal(ggml_backend_t potal) {
    {
        ggml_context_ptr ctx = make_context(true);
        ggml_context *   c   = ctx.get();
        ggml_tensor *    sum = ggml_add(c, ggml_new_tensor_2d(c, GGML_TYPE_F32, 16, 16),
                                        ggml_new_tensor_2d(c, GGML_TYPE_F32, 16, 16));
        ggml_cgraph *    gf  = ggml_new_graph(c);
        ggml_build_forward_expand(gf, sum);
        const ggml_backend_potal_stats before = stats_of(potal);
        const ggml_status              status = ggml_backend_graph_compute(potal, gf);
        const ggml_backend_potal_stats after  = stats_of(potal);
        check("E1 ADD computed on POTAL fails without a hardware request",
              status == GGML_STATUS_FAILED && after.rejected_nodes == before.rejected_nodes + 1 &&
                  after.hw_requests == before.hw_requests);
        check("E1 refused dst set to NaN", all_nan(sum));
    }
    {
        ggml_context_ptr ctx = make_context(true);
        ggml_context *   c   = ctx.get();
        ggml_tensor *    y   = ggml_mul_mat(c, ggml_new_tensor_2d(c, GGML_TYPE_I8, 16, 16),
                                            ggml_new_tensor_2d(c, GGML_TYPE_I8, 16, 8));
        ggml_cgraph *    gf  = ggml_new_graph(c);
        ggml_build_forward_expand(gf, y);
        const ggml_backend_potal_stats before = stats_of(potal);
        const ggml_status              status = ggml_backend_graph_compute(potal, gf);
        const ggml_backend_potal_stats after  = stats_of(potal);
        check("E2 MUL_MAT M=8 computed on POTAL fails without a hardware request",
              status == GGML_STATUS_FAILED && after.rejected_nodes == before.rejected_nodes + 1 &&
                  after.mul_mat_nodes == before.mul_mat_nodes && after.hw_requests == before.hw_requests);
        check("E2 refused dst set to NaN", all_nan(y));
    }
#if defined(__riscv)
    std::printf("SKIP E3 valid graph without hardware: RISC-V target (potal-ggml-smoke runs it on hardware)\n");
#else
    {
        ggml_context_ptr ctx = make_context(true);
        ggml_context *   c   = ctx.get();
        ggml_tensor *    w   = ggml_new_tensor_2d(c, GGML_TYPE_I8, 16, TILE);
        ggml_tensor *    x   = ggml_new_tensor_2d(c, GGML_TYPE_I8, 16, TILE);
        fill<int8_t>(w, wt_at);
        fill<int8_t>(x, act_at);
        ggml_tensor * y = ggml_mul_mat(c, w, x);
        std::fill(static_cast<float *>(y->data), static_cast<float *>(y->data) + ggml_nelements(y), 7.0f);
        ggml_cgraph * gf = ggml_new_graph(c);
        ggml_build_forward_expand(gf, y);

        const ggml_backend_potal_stats before = stats_of(potal);
        const ggml_status              status = ggml_backend_graph_compute(potal, gf);
        const ggml_backend_potal_stats after  = stats_of(potal);
        check("E3 valid graph off RISC-V returns GGML_STATUS_FAILED", status == GGML_STATUS_FAILED);
        check("E3 request reached the PoTal API and was not issued",
              after.mul_mat_nodes == before.mul_mat_nodes + 1 && after.hw_requests == before.hw_requests + 1 &&
                  after.gemm_issued == before.gemm_issued && after.gemm_fenced == before.gemm_fenced &&
                  after.gemm_completed == before.gemm_completed && after.gemm_failed == before.gemm_failed + 1);
        check("E3 status POTAL_ERR_NO_HARDWARE", after.last_status == POTAL_ERR_NO_HARDWARE,
              "last_status=" + std::to_string(after.last_status));
        check("E3 dst set to NaN, no fake result", all_nan(y));
        check("E3 no issuer thread off RISC-V", !after.issuer_ready && after.issuer_cpu == -1);
    }
#endif
}

// ---- F. scheduler placement --------------------------------------------------

void test_scheduler(ggml_backend_t potal, ggml_backend_t cpu) {
    ggml_backend_t backends[2] = { potal, cpu };

    {
        ggml_context_ptr        inputs = make_context(false);
        ggml_tensor *           w      = ggml_new_tensor_2d(inputs.get(), GGML_TYPE_I8, 16, TILE);
        ggml_tensor *           x      = ggml_new_tensor_2d(inputs.get(), GGML_TYPE_I8, 16, TILE);
        ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(inputs.get(), cpu));
        ggml_context_ptr        graph = make_context(false);
        ggml_tensor *           y     = ggml_mul_mat(graph.get(), w, x);
        ggml_tensor *           z     = ggml_scale(graph.get(), y, 0.5f);
        ggml_cgraph *           gf    = ggml_new_graph(graph.get());
        ggml_build_forward_expand(gf, z);

        ggml_backend_sched_ptr sched(ggml_backend_sched_new(backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, true));
        const bool             allocated = buffer && ggml_backend_sched_alloc_graph(sched.get(), gf);
        check("F1 I8 MUL_MAT placed on POTAL",
              allocated && ggml_backend_sched_get_tensor_backend(sched.get(), y) == potal);
        check("F1 F32 SCALE placed on CPU", allocated && ggml_backend_sched_get_tensor_backend(sched.get(), z) == cpu);
    }

    {
        ggml_context_ptr        inputs = make_context(false);
        ggml_tensor *           w      = ggml_new_tensor_2d(inputs.get(), GGML_TYPE_F32, 16, TILE);
        ggml_tensor *           x      = ggml_new_tensor_2d(inputs.get(), GGML_TYPE_F32, 16, TILE);
        ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(inputs.get(), cpu));
        std::vector<float>      wv(16 * TILE);
        std::vector<float>      xv(16 * TILE);
        for (int64_t row = 0; row < TILE; ++row) {
            for (int64_t kk = 0; kk < 16; ++kk) {
                wv[row * 16 + kk] = wt_at(row, kk);
                xv[row * 16 + kk] = act_at(row, kk);
            }
        }
        ggml_backend_tensor_set(w, wv.data(), 0, ggml_nbytes(w));
        ggml_backend_tensor_set(x, xv.data(), 0, ggml_nbytes(x));
        ggml_context_ptr graph = make_context(false);
        ggml_tensor *    y     = ggml_mul_mat(graph.get(), w, x);
        ggml_cgraph *    gf    = ggml_new_graph(graph.get());
        ggml_build_forward_expand(gf, y);

        ggml_backend_sched_ptr sched(ggml_backend_sched_new(backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, true));
        const ggml_backend_potal_stats before    = stats_of(potal);
        const bool                     allocated = ggml_backend_sched_alloc_graph(sched.get(), gf);
        const bool on_cpu = allocated && ggml_backend_sched_get_tensor_backend(sched.get(), y) == cpu;
        check("F2 F32 MUL_MAT refused by POTAL, placed on CPU", on_cpu);
        std::printf("SCHED_FALLBACK mul_mat_nodes potal=%d cpu=%d\n", on_cpu ? 0 : 1, on_cpu ? 1 : 0);

        const ggml_status              status = ggml_backend_sched_graph_compute(sched.get(), gf);
        const ggml_backend_potal_stats after  = stats_of(potal);
        std::vector<float>             out(TILE * TILE);
        ggml_backend_tensor_get(y, out.data(), 0, ggml_nbytes(y));
        const std::vector<int64_t> want       = golden(16);
        int                        mismatches = 0;
        for (size_t i = 0; i < out.size(); ++i) {
            mismatches += out[i] != static_cast<float>(want[i]);
        }
        check("F2 computed by the CPU backend, exact",
              status == GGML_STATUS_SUCCESS && mismatches == 0,
              "mismatches=" + std::to_string(mismatches));
        check("F2 POTAL executed nothing",
              after.mul_mat_nodes == before.mul_mat_nodes && after.hw_requests == before.hw_requests);
    }

    {
        ggml_context_ptr ctx = make_context(false);
        ggml_tensor *    y   = ggml_mul_mat(ctx.get(), ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I8, 16, 16),
                                            ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I8, 16, 8));
        check("F3 I8 M=8 refused by POTAL", !ggml_backend_supports_op(potal, y));
        check("F3 I8 x I8 refused by CPU too: not scheduled, no forced fallback", !ggml_backend_supports_op(cpu, y));
    }
}

// ---- G. POTAL buffer type ---------------------------------------------------

void test_buffer(ggml_backend_t potal, ggml_backend_t cpu) {
    using ggml::potal::is_potal_buffer;
    using ggml::potal::potal_buffer_generation;
    ggml_backend_dev_t         dev  = ggml_backend_get_device(potal);
    ggml_backend_dev_t         cdev = ggml_backend_get_device(cpu);
    ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);
    ggml_backend_buffer_type_t cpu_buft = ggml_backend_cpu_buffer_type();

    check("G1 device default = backend default buffer type", buft == ggml_backend_get_default_buffer_type(potal));
    check("G1 POTAL buffer type is not the CPU buffer type", buft != cpu_buft);
    check("G1 named POTAL (not CPU, not GEMMINI)", std::strcmp(ggml_backend_buft_name(buft), "POTAL") == 0);
    check("G1 no GEMMINI device whose buffer type it could be", ggml_backend_dev_by_name("GEMMINI") == nullptr);
    check("G1 buffer type device is the POTAL ACCEL device",
          ggml_backend_buft_get_device(buft) == dev && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_ACCEL);
    check("G1 host memory, CPU alignment", ggml_backend_buft_is_host(buft) &&
                                               ggml_backend_buft_get_alignment(buft) ==
                                                   ggml_backend_buft_get_alignment(cpu_buft));
    check("G1 POTAL accepts POTAL and CPU buffers; CPU accepts POTAL buffers",
          ggml_backend_dev_supports_buft(dev, buft) && ggml_backend_dev_supports_buft(dev, cpu_buft) &&
              ggml_backend_dev_supports_buft(cdev, buft) && ggml_backend_supports_buft(cpu, buft));

    // G2: owned allocations.
    {
        ggml_backend_buffer_ptr a(ggml_backend_buft_alloc_buffer(buft, 1000));
        ggml_backend_buffer_ptr b(ggml_backend_buft_alloc_buffer(buft, 1000));
        const bool              ok = a && b;
        check("G2 alloc: POTAL buffers, aligned, sized, host",
              ok && is_potal_buffer(a.get()) && ggml_backend_buffer_get_type(a.get()) == buft &&
                  ggml_backend_buffer_get_size(a.get()) == 1000 && ggml_backend_buffer_is_host(a.get()) &&
                  reinterpret_cast<uintptr_t>(ggml_backend_buffer_get_base(a.get())) %
                          ggml_backend_buft_get_alignment(buft) == 0);
        check("G2 independent buffers: distinct memory and nonzero distinct generations",
              ok && ggml_backend_buffer_get_base(a.get()) != ggml_backend_buffer_get_base(b.get()) &&
                  potal_buffer_generation(a.get()) != 0 && potal_buffer_generation(b.get()) != 0 &&
                  potal_buffer_generation(a.get()) != potal_buffer_generation(b.get()));
        if (ok) {
            ggml_backend_buffer_clear(a.get(), 0x5a);
            ggml_backend_buffer_clear(b.get(), 0xa5);
            const uint8_t * pa = static_cast<const uint8_t *>(ggml_backend_buffer_get_base(a.get()));
            const uint8_t * pb = static_cast<const uint8_t *>(ggml_backend_buffer_get_base(b.get()));
            check("G2 clear fills the whole buffer, the other untouched",
                  std::all_of(pa, pa + 1000, [](uint8_t v) { return v == 0x5a; }) &&
                      std::all_of(pb, pb + 1000, [](uint8_t v) { return v == 0xa5; }));
        }
        check("G2 null buffer: not a POTAL buffer, generation 0",
              !is_potal_buffer(nullptr) && potal_buffer_generation(nullptr) == 0);
        ggml_backend_buffer_ptr c(ggml_backend_buft_alloc_buffer(cpu_buft, 64));
        check("G2 CPU buffer is not a POTAL buffer", c && !is_potal_buffer(c.get()) && potal_buffer_generation(c.get()) == 0);
    }

    // G3: zero size is GGML's dummy buffer (no allocation, no POTAL context).
    {
        ggml_backend_buffer_ptr z(ggml_backend_buft_alloc_buffer(buft, 0));
        check("G3 zero-size buffer: dummy of the POTAL type, size 0",
              z && ggml_backend_buffer_get_type(z.get()) == buft && ggml_backend_buffer_get_size(z.get()) == 0 &&
                  !is_potal_buffer(z.get()) && potal_buffer_generation(z.get()) == 0);
    }

    // G4: tensor I/O, offsets, memset and copies between POTAL and CPU buffers.
    {
        ggml_context_ptr ctx = make_context(false);
        ggml_tensor *    t   = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 64);
        ggml_tensor *    u   = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 64);
        ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft));
        ggml_context_ptr cctx = make_context(false);
        ggml_tensor *    c    = ggml_new_tensor_1d(cctx.get(), GGML_TYPE_F32, 64);
        ggml_backend_buffer_ptr cbuf(ggml_backend_alloc_ctx_tensors_from_buft(cctx.get(), cpu_buft));
        bool ok = buf && cbuf && t->buffer == buf.get() && u->buffer == buf.get() && t->data != u->data;
        std::vector<float> in(64), out(64, -1.0f);
        for (int i = 0; i < 64; ++i) {
            in[i] = 0.25f * (float) i - 3.0f;
        }
        if (ok) {
            ggml_backend_tensor_set(t, in.data(), 0, 64 * sizeof(float));
            ggml_backend_tensor_get(t, out.data(), 0, 64 * sizeof(float));
            ok = std::memcmp(in.data(), out.data(), 64 * sizeof(float)) == 0 &&
                 std::memcmp(t->data, in.data(), 64 * sizeof(float)) == 0;
        }
        check("G4 set/get round trip, CPU-readable data", ok);
        if (ok) {
            const float v = 7.5f;
            ggml_backend_tensor_set(t, &v, 10 * sizeof(float), sizeof(float));
            float back = 0.0f;
            ggml_backend_tensor_get(t, &back, 10 * sizeof(float), sizeof(float));
            ggml_backend_tensor_memset(t, 0, 60 * sizeof(float), 4 * sizeof(float));
            const float * d = static_cast<const float *>(t->data);
            check("G4 offset set/get and tail memset touch only their range",
                  back == v && d[9] == in[9] && d[11] == in[11] && d[59] == in[59] && d[60] == 0.0f && d[63] == 0.0f);
            ggml_backend_tensor_copy(t, c);
            ggml_backend_tensor_copy(c, u);
            check("G4 POTAL -> CPU -> POTAL copies", std::memcmp(c->data, t->data, ggml_nbytes(t)) == 0 &&
                                                         std::memcmp(u->data, t->data, ggml_nbytes(t)) == 0);
        }
    }

    // G5: host-pointer alias (the loader's mmap path).
    {
        const size_t size = 4096;
        void *       mem  = std::aligned_alloc(64, size);
        std::memset(mem, 0x11, size);
        ggml_backend_buffer_t alias = ggml_backend_dev_buffer_from_host_ptr(dev, mem, size, size);
        const bool            ok    = alias != nullptr;
        check("G5 host-pointer buffer: POTAL type, base is the pointer, not owned memory",
              ok && is_potal_buffer(alias) && ggml_backend_buffer_get_type(alias) == buft &&
                  ggml_backend_buffer_get_base(alias) == mem && ggml_backend_buffer_get_size(alias) == size &&
                  potal_buffer_generation(alias) != 0);
        if (ok) {
            ggml_backend_buffer_clear(alias, 0x22);
            ggml_backend_buffer_free(alias); // must not free mem
        }
        check("G5 writes reach the caller memory, which outlives the buffer",
              ok && static_cast<uint8_t *>(mem)[0] == 0x22 && static_cast<uint8_t *>(mem)[size - 1] == 0x22);
        std::free(mem); // a double free here would abort
        alignas(64) static uint8_t raw[128];
        check("G5 misaligned host pointer refused",
              ggml_backend_dev_buffer_from_host_ptr(dev, raw + 1, 64, 64) == nullptr);
    }

    // G6: read-only file mapping, as llama_mmap creates it.
    {
        char       path[] = "/tmp/potal-buffer-test-XXXXXX";
        const int  fd     = mkstemp(path);
        const size_t size = 8192;
        std::vector<uint8_t> bytes(size);
        for (size_t i = 0; i < size; ++i) {
            bytes[i] = (uint8_t) (i * 131 + 7);
        }
        bool ok = fd >= 0 && write(fd, bytes.data(), size) == (ssize_t) size;
        void * map = ok ? mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0) : MAP_FAILED;
        ok         = ok && map != MAP_FAILED;
        if (ok) {
            ggml_backend_buffer_t alias = ggml_backend_dev_buffer_from_host_ptr(dev, map, size, size);
            ggml_context_ptr      ctx   = make_context(false);
            ggml_tensor *         t     = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I8, 256);
            const bool alloc = alias && ggml_backend_tensor_alloc(alias, t, static_cast<char *>(map) + 1024) ==
                                            GGML_STATUS_SUCCESS;
            std::vector<uint8_t> got(256);
            if (alloc) {
                ggml_backend_tensor_get(t, got.data(), 0, 256);
            }
            check("G6 read-only mmap alias: tensor reads the mapped file bytes",
                  alloc && t->data == static_cast<char *>(map) + 1024 &&
                      std::memcmp(got.data(), bytes.data() + 1024, 256) == 0);
            if (alias) {
                ggml_backend_buffer_free(alias);
            }
            check("G6 mapping intact after the buffer is freed",
                  std::memcmp(map, bytes.data(), size) == 0 && munmap(map, size) == 0);
        } else {
            check("G6 read-only mmap alias: temporary file mapped", false);
        }
        if (fd >= 0) {
            close(fd);
            unlink(path);
        }
    }

    // G7: Q8_HP1 weights in a POTAL weights buffer, as llama.cpp places GPT-2's
    // token_embd.weight: GET_ROWS on CPU, MUL_MAT on POTAL, cache-eligible.
    {
        ggml_context_ptr ctx = make_context(false);
        ggml_tensor *    w   = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_Q8_HP1, 64, 32);
        ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft));
        if (buf) {
            ggml_backend_buffer_set_usage(buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        }
        ggml_tensor * ids  = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 4);
        ggml_tensor * rows = ggml_get_rows(ctx.get(), w, ids);
        ggml_tensor * x    = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 64, 4);
        ggml_tensor * y    = ggml_mul_mat(ctx.get(), w, x);
        check("G7 Q8_HP1 weights buffer: usage WEIGHTS, cache-eligible",
              buf && ggml_backend_buffer_get_usage(buf.get()) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
                  ggml::potal::weight_cache::eligible(w));
        check("G7 GET_ROWS of a POTAL-buffer Q8_HP1 weight: CPU yes, POTAL no",
              ggml_backend_dev_supports_op(cdev, rows) && !ggml_backend_dev_supports_op(dev, rows));
        check("G7 F32 x Q8_HP1 MUL_MAT of it: POTAL yes, CPU no",
              ggml_backend_dev_supports_op(dev, y) && !ggml_backend_dev_supports_op(cdev, y));
    }
}

} // namespace

int main() {
    std::printf("POTAL_GGML_UNIT_TEST starting\n");

    test_registry();

    ggml_backend_ptr potal(ggml_backend_potal_init());
    ggml_backend_ptr cpu(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr));
    if (!potal || !cpu) {
        std::printf("FAIL backend init (potal=%p cpu=%p)\nPOTAL_GGML_UNIT_TEST FAIL\n", (void *) potal.get(),
                    (void *) cpu.get());
        return 1;
    }

    test_backend(potal.get(), cpu.get());
    test_contract(potal.get());
    test_packing(16, cpu.get());
    test_packing(32, cpu.get());
    test_unpack_extremes();
    test_view_packing();
    test_refusal(potal.get());
    test_scheduler(potal.get(), cpu.get());
    test_buffer(potal.get(), cpu.get());

    if (g_failures != 0) {
        std::printf("POTAL_GGML_UNIT_TEST FAIL (%d)\n", g_failures);
        return 1;
    }
    std::printf("POTAL_GGML_UNIT_TEST PASS\n");
    return 0;
}
