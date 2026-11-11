#pragma once

#include "execution.hpp"
#include "ggml-potal.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace ggml::potal::attention {

constexpr size_t block_k = 32;
constexpr size_t max_limbs = 5; // INT32_MAX needs a fifth balanced radix-256 digit.

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
                                 role operation, stationary & out);
potal_status prepare_stream(const ggml_tensor & tensor, size_t head, size_t batch,
                            size_t begin, size_t rows, role operation, stream & out,
                            pv_algorithm algorithm = default_pv_algorithm);

// Software integer reference, never a backend execution route. For every Q
// limb: independently completed K-low and K-high INT32 ACCs use the same SCU
// Delta. Checked Host INT64 combines low + 256*high, then the Q radix limbs.
// Explicit legacy PV adds the exact block-column bias ONCE to
// the completed low-limb INT32 ACC. Upper P limbs have separate INT32 ACCs
// and reconstruct with the corrected low result in checked Host INT64, as
// explicitly selected for H3-7B. Bias outside INT32 is rejected, not clipped.
// H3-7C extends only a rejected PV bias: unchanged unsigned P codes run as
// radix-2 INT8 planes if every DIM16/DIM32 SCU and ACC prefix is proven nonsaturating.
// Those completed jobs reconstruct the exact quantized dot in checked INT64.
// Inputs that cannot be certified remain unsupported; column_bias stays strict.
// Default balanced PV instead uses direct unsigned-code signed radix-256 limbs,
// ordinary independent GEMMs and checked reconstruction, with no bias or plane
// certificate. Its exact reference retains SCU/ACC saturation in physical order.
potal_status reference(const stream & activation, const stationary & weight,
                       role operation, std::vector<int64_t> & out);
// Exact 128 * sum_block(2^carrier * sum_k(Vcode)) with checked INT32 output.
potal_status column_bias(const stationary & weight, std::vector<int32_t> & out);

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

// The runner is the existing planned WS execution seam. Tests bind the
// RTL-semantics emulator; production binds the hart-0 issuer. No CPU fallback.
// QK K-low/K-high are separate complete GEMMs: each starts a fresh ACC epoch,
// and C is absent until that limb's final K panel. PV's corrected runner keeps
// a known ACC bank, applies one final full-width bias MVIN and reads it back.
using corrected_runner = std::function<planned_result(const potal_stripe_gemm_desc &,
                                                       const int32_t *, size_t)>;
result execute(ggml_tensor * op, const planned_runner & run, const corrected_runner & corrected = {},
               pv_algorithm algorithm = default_pv_algorithm);
// On the calling accelerator-owner thread; sink is only for command/emulator tests.
planned_result run_corrected(const potal_stripe_gemm_desc & desc, const int32_t * bias, size_t columns,
                             const potal_paired_sink * sink = nullptr);
result execute_hardware(ggml_tensor * op);

// H3-8I-25V strict five-hart attention: the same stripes, operands, GEMMs and reconstruction as
// execute(), bitwise identical, with every CPU numeric step on the hart-role threads (local 1/2,
// fold 3, final 4, pinned to role_cpus when pin) and only the owner's planned runner on its thread
// (hart 0 in the backend). Balanced PV and QK only; legacy PV is refused. execute_strict_hardware
// binds the hart-0 issuer and the /proc/cpuinfo role CPUs.
result execute_strict(ggml_tensor * op, stripe_owner & owner, const std::array<int, 5> & role_cpus, bool pin,
                      pv_algorithm algorithm = default_pv_algorithm);
result execute_strict_hardware(ggml_tensor * op);

} // namespace ggml::potal::attention
