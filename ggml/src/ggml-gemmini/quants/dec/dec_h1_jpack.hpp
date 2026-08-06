#pragma once

#ifndef GGML_COMMON_DECL
#define DEC_H1_JPACK_DEFINE_GGML_COMMON
#define GGML_COMMON_DECL_CPP
#endif
#include "../../../ggml-common.h"
#ifdef DEC_H1_JPACK_DEFINE_GGML_COMMON
#undef GGML_COMMON_DECL_CPP
#undef DEC_H1_JPACK_DEFINE_GGML_COMMON
#endif

#include <cstddef>
#include <cstdint>

namespace ggml::gemmini::quants::dec
{
inline constexpr size_t kDecH1JPackGroupSize = QK8_0;
inline constexpr size_t kDecH1JPackNr = 8;

struct DecH1JPackAccounting
{
    size_t packed_bytes = 0;
    size_t allocated_bytes = 0;
    uint64_t pack_time_ns = 0;
    size_t source_code_bytes_read = 0;
    size_t packed_bytes_written = 0;
    size_t panel_count = 0;
    size_t group_count = 0;
    size_t tail_panel_count = 0;
};

struct DecH1JPackEntry
{
    size_t k = 0;
    int32_t residual = 0;
};

enum class DecH1JPackSafety : uint8_t
{
    INT32_SAFE,
    INT64_REQUIRED,
    INVALID,
};

struct DecH1JPackCounters
{
    uint64_t vector_kernel_hits = 0;
    uint64_t vector_loads = 0;
    uint64_t scalar_fallback_loads = 0;
    uint64_t processed_macs = 0;
};

class DecH1JPack
{
public:
    DecH1JPack(const block_q8_h1 *source, size_t columns, size_t depth);
    ~DecH1JPack();

    DecH1JPack(const DecH1JPack &) = delete;
    DecH1JPack & operator=(const DecH1JPack &) = delete;
    DecH1JPack(DecH1JPack &&other) noexcept;
    DecH1JPack & operator=(DecH1JPack &&other) noexcept;

    bool valid() const noexcept;
    size_t columns() const noexcept;
    size_t depth() const noexcept;
    size_t panel_count() const noexcept;
    size_t group_count() const noexcept;
    const DecH1JPackAccounting & accounting() const noexcept;

    const int8_t *code_pointer(size_t k, size_t panel) const noexcept;
    int8_t packed_code(size_t k_group, size_t j_panel, size_t k_offset, size_t j_lane) const noexcept;

private:
    void reset() noexcept;
    size_t offset(size_t k_group, size_t j_panel, size_t k_offset, size_t j_lane) const noexcept;

    int8_t *data_ = nullptr;
    size_t columns_ = 0;
    size_t depth_ = 0;
    DecH1JPackAccounting accounting_{};
};

DecH1JPackSafety dec_h1_jpack_classify(
    const DecH1JPack &pack, const DecH1JPackEntry *entries, size_t entry_count) noexcept;

bool dec_h1_jpack_execute_source_scalar(
    const block_q8_h1 *source, size_t columns, size_t depth,
    const DecH1JPackEntry *entries, size_t entry_count, int64_t *output) noexcept;

bool dec_h1_jpack_execute_packed(
    const DecH1JPack &pack, const DecH1JPackEntry *entries, size_t entry_count,
    int64_t *output, DecH1JPackCounters &counters) noexcept;

bool dec_h1_jpack_has_neon() noexcept;
}

#if defined(_MSC_VER)
#define DEC_H1_JPACK_NOINLINE __declspec(noinline)
#else
#define DEC_H1_JPACK_NOINLINE __attribute__((noinline))
#endif

extern "C" DEC_H1_JPACK_NOINLINE void dec_h1_jpack_neon_nr8(
    const int8_t *const *codes, const int32_t *residuals, size_t entry_count,
    size_t panel_count, int64_t *output, uint64_t *vector_kernel_hits,
    uint64_t *vector_loads, uint64_t *scalar_fallback_loads, uint64_t *processed_macs);

#undef DEC_H1_JPACK_NOINLINE
