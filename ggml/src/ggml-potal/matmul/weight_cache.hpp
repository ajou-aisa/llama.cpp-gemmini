#pragma once

// Prepared Q8_HP1 weights of one POTAL backend instance (backend_state).
//
// A model weight is immutable for the life of the backend that computes with
// it: llama.cpp creates the backends with a llama_context, and the model (its
// GGML_BACKEND_BUFFER_USAGE_WEIGHTS buffers) must outlive every context. So
// the validated, unpacked B / carriers / channel_scale of such a weight are
// prepared on its first MUL_MAT and reused by every later one of this backend.
//
//   eligible  src0 lives in a buffer with usage GGML_BACKEND_BUFFER_USAGE_WEIGHTS.
//             Anything else (compute buffers, tensors without a buffer) may be
//             rewritten between graphs and is prepared per call, as before.
//   key       buffer, its POTAL generation (0 for other buffer types), data
//             address (includes a view's offset), type, ne, nb: another view,
//             shape, stride or alignment is another entry, and a POTAL buffer
//             allocated later at a reused address never matches an old entry.
//   insert    only after prepare_weight() succeeded; a failed weight is never
//             cached and fails again (full validation) on its next use.
//   budget    GGML_POTAL_WEIGHT_CACHE_MB (default 1024, 0 disables). A weight
//             that does not fit is prepared per call (bypass); entries are
//             never evicted while the backend lives, so no buffer the
//             accelerator may still read is ever freed under it.
//   release   all entries are freed with the backend, or by clear(). Both must
//             not overlap a graph_compute of this backend (the GGML contract
//             for one backend instance). Mutating or freeing a weight buffer
//             while the backend lives (e.g. a LoRA adapter freed and another
//             one allocated at the same address) requires clear() first.
//   fatal     after a fatal hardware failure abandon() leaks the entry instead
//             of freeing it (the accelerator may still reach its buffers).
//
// Lookups take a mutex, so concurrent backends or threads never prepare one
// entry twice or race on the map; a hit never touches the weight data.

#include "dense.hpp"

#include "ggml.h"
#include "ggml-backend.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <tuple>

namespace ggml::potal {

enum class weight_cache_outcome : uint8_t {
    hit,      // prepared earlier, reused: no validation, no unpacking
    miss,     // prepared now and cached
    bypass,   // prepared now, not cached: over the budget
    uncached, // prepared now, not cached: not in a weights buffer
    failed,   // allocation or preparation failed; nothing cached
};

const char * weight_cache_outcome_name(weight_cache_outcome o);

struct weight_lookup {
    weight_cache_outcome        outcome = weight_cache_outcome::failed;
    const hp1_weight *          weight  = nullptr;    // valid while the entry (or `transient`) lives
    std::unique_ptr<hp1_weight> transient;            // owner of a bypass/uncached weight
    quants::weight::hp1::status status  = quants::weight::hp1::status::ok;
    ggml::gemmini::quants::wreader::WeightReaderStatus reader =
        ggml::gemmini::quants::wreader::WeightReaderStatus::Success;
    bool   alloc_failed = false;
    double alloc_s      = 0; // zeroed buffer allocation (not hit)
    double prepare_s    = 0; // prepare_weight(): validation + unpacking (not hit)
};

class weight_cache {
  public:
    weight_cache();                             // budget from GGML_POTAL_WEIGHT_CACHE_MB
    explicit weight_cache(size_t budget_bytes); // 0 disables caching

    // The prepared operands of src0 (args: init_args of this MUL_MAT).
    weight_lookup get(const ggml_tensor * src0, const ggml_gemmini_args_t & args);

    void abandon(const hp1_weight * w); // drop the entry of w and leak its buffers
    void clear();                       // free every entry

    struct counters {
        uint64_t hits = 0, misses = 0, bypasses = 0, uncached = 0, failures = 0, abandoned = 0;
        uint64_t entries = 0, bytes = 0, budget = 0;
    };
    counters snapshot() const;

    static bool eligible(const ggml_tensor * src0);

  private:
    using key = std::tuple<const void *, uint64_t, const void *, int, int64_t, int64_t, int64_t, int64_t, size_t,
                           size_t, size_t, size_t>;
    static key key_of(const ggml_tensor * t);

    mutable std::mutex                             mutex_;
    std::map<key, std::unique_ptr<hp1_weight>>     entries_;
    counters                                       c_;
};

} // namespace ggml::potal
