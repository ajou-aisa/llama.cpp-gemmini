#pragma once

#include "execution.hpp"
#include "ggml-potal.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace ggml::potal::attention {

constexpr size_t block_k = 32;
constexpr size_t max_limbs = 9; // INT32_MAX needs the ninth balanced radix-16 carry digit.

enum class role { none, qk, pv };
enum class pv_algorithm { legacy, balanced };
constexpr pv_algorithm default_pv_algorithm = pv_algorithm::balanced;

// Reuse the existing graph role in op_params[1], not tensor names or shapes.
role get_role(const ggml_tensor * op);
ggml_potal_reject check(const ggml_tensor * op, bool require_data);

struct stationary {
    size_t k = 0, n = 0, kp = 0;
    std::vector<int16_t> base;
    std::vector<uint32_t> carriers; // [kp / 32][n], zero carrier or exponent difference.
    std::array<std::vector<int8_t>, 2> limbs; // [kp][n]; K lower/upper, or V.
};

struct stream {
    pv_algorithm algorithm = default_pv_algorithm;
    size_t rows = 0, k = 0, kp = 0, limb_count = 0;
    int scale = 0;
    std::vector<int32_t> codes; // Folded Q or clipped unsigned P, including zero padding.
    std::array<std::vector<int8_t>, max_limbs> limbs; // [rows][kp].
};

// Preparation reuses Gemmini's tensor reads, MXINT quantizer, ExSIA local stage
// and folding. P has no local adaptation or outlier selection. Outputs publish
// only on success; finite input and bounds are checked before quantization.
potal_status prepare_stationary(const ggml_tensor & tensor, size_t head, size_t batch,
                                 role operation, stationary & out, const std::vector<int> * qk_shifts = nullptr);
potal_status prepare_stream(const ggml_tensor & tensor, size_t head, size_t batch,
                            size_t begin, size_t rows, role operation, stream & out,
                            pv_algorithm algorithm = default_pv_algorithm,
                            const std::vector<int> * qk_shifts = nullptr);

// Per-KV-head channel exponents from the first min(128, query rows) inputs,
// shared by all broadcast query heads/batches. Q *= 2^d; K *= 2^-d.
// Passing this plan to prepare_stream selects rho2 Q with selective residuals.
potal_status prepare_qk_transfer(const ggml_tensor & keys, const ggml_tensor & queries,
                                 size_t head, size_t batch, std::vector<int> & shifts);

// Software reference only. Independent INT32 SCU/ACC epochs for each digit pair;
// checked INT64 merges in stationary-pass order, main before original upper lanes.
// QK uses rho6 K/Q codes, two K passes, and nine possible Q digit positions.
// PV uses unsigned eight-bit P = P0 + 16*upper + 8, signed INT4 V, and adds
// 8*carrier-scaled V column sum once on the CPU after all digit GEMM merges.
potal_status reference(const stream & activation, const stationary & weight,
                       role operation, std::vector<int64_t> & out);
// Exact CPU zero-point compensation, checked after each block and final shift.
potal_status column_bias(const stationary & weight, std::vector<int64_t> & out);

struct result {
    potal_status status = POTAL_OK;
    const char * stage = nullptr;
    bool fatal = false;
    uint64_t jobs = 0, residual_jobs = 0, qk_limb_jobs = 0, correction_jobs = 0;
    uint64_t bias_updates = 0;
    uint64_t pv_bitplane_stripes = 0, pv_bitplane_jobs = 0;
    uint64_t loops = 0, fences = 0, scale_writes = 0, scale_releases = 0, tlb_flushes = 0;
    double scale_s = 0, issue_s = 0, fence_s = 0, reconstruct_s = 0;
    // execute_strict (H3-8I-25V): stripes, and CPU numeric steps seen on the issuer (must stay 0).
    uint64_t strict_tasks = 0, cpu_work_on_issuer_thread = 0, cpu_work_on_issuer_cpu = 0;
};

// Existing scalar-Index paired runner: packed Main and original-block upper
// packets share immutable resident B. No INT32 bias MVIN or CPU fallback.
// The balanced enum value names this fork's signed-digit algorithm; legacy is refused.
result execute(ggml_tensor * op, const paired_runner & run,
               pv_algorithm algorithm = default_pv_algorithm);
result execute_hardware(ggml_tensor * op);

// H3-8I-25V strict five-hart attention: the same stripes, operands, GEMMs and reconstruction as
// execute(), bitwise identical, with every CPU numeric step on the hart-role threads (local 1/2,
// fold 3, final 4, pinned to role_cpus when pin) and only the owner's planned runner on its thread
// (hart 0 in the backend). Signed-digit PV and QK only; legacy PV is refused. execute_strict_hardware
// binds the hart-0 issuer and the /proc/cpuinfo role CPUs.
result execute_strict(ggml_tensor * op, stripe_owner & owner, const std::array<int, 5> & role_cpus, bool pin,
                      pv_algorithm algorithm = default_pv_algorithm);
result execute_strict_hardware(ggml_tensor * op);

} // namespace ggml::potal::attention
