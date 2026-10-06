#pragma once

#include "types.hpp"
#include "exsia-profile.hpp"
#include "../../../residual/residual-capture.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace ggml::gemmini::quants::act::exsia {
struct BitMask {
    size_t                rows = 0;
    size_t                cols = 0;
    std::vector<uint64_t> words;

    static bool checked_mul(size_t lhs, size_t rhs, size_t & out) {
        if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs)
            return false;

        out = lhs * rhs;
        return true;
    }

    static bool checked_add(size_t lhs, size_t rhs, size_t & out) {
        if (lhs > std::numeric_limits<size_t>::max() - rhs)
            return false;

        out = lhs + rhs;
        return true;
    }

    // resize the bitmask matrix to the specified number of rows and columns
    bool resize(size_t row_count, size_t col_count) {
        size_t bit_count         = 0;
        size_t rounded_bit_count = 0;
        if (!checked_mul(row_count, col_count, bit_count) ||
            !checked_add(bit_count, 63, rounded_bit_count)) {
            clear();
            return false;
        }

        rows = row_count;
        cols = col_count;
        words.assign(rounded_bit_count / 64, 0);
        return true;
    }

    size_t active_word_count() const {
        size_t bit_count         = 0;
        size_t rounded_bit_count = 0;
        return checked_mul(rows, cols, bit_count) && checked_add(bit_count, 63, rounded_bit_count)
                   ? rounded_bit_count / 64
                   : 0;
    }

    bool prepare(size_t row_count, size_t col_count) {
        size_t bit_count         = 0;
        size_t rounded_bit_count = 0;
        if (!checked_mul(row_count, col_count, bit_count) ||
            !checked_add(bit_count, 63, rounded_bit_count)) {
            clear();
            return false;
        }

        rows                    = row_count;
        cols                    = col_count;
        const size_t word_count = rounded_bit_count / 64;
        if (words.size() < word_count)
            words.resize(word_count);

        std::fill(words.begin(), words.begin() + word_count, 0);
        return true;
    }

    // clear the bitmask matrix, setting all bits to 0
    void clear() {
        words.clear();
        rows = 0;
        cols = 0;
    }

    void clear_active_bits() {
        std::fill(words.begin(), words.begin() + active_word_count(), 0);
    }

    // set the bit at the specified row and column to 1
    void set(size_t row, size_t col) {
        assert(row < rows && col < cols);
        const size_t   idx      = row * cols + col;
        const size_t   word_idx = idx / 64;
        const uint64_t bit      = uint64_t(1) << (idx % 64);
#if defined(GGML_GEMMINI_HAS_OPENMP)
#pragma omp atomic update
        words[word_idx] |= bit;
#else
        words[word_idx] |= bit;
#endif
    }

    // reset the bit at the specified row and column to 0
    void reset(size_t row, size_t col) {
        assert(row < rows && col < cols);
        const size_t   idx      = row * cols + col;
        const size_t   word_idx = idx / 64;
        const uint64_t bit      = uint64_t(1) << (idx % 64);
#if defined(GGML_GEMMINI_HAS_OPENMP)
#pragma omp atomic update
        words[word_idx] &= ~bit;
#else
        words[word_idx] &= ~bit;
#endif
    }

    // test whether the bit at the specified row and column is set to 1
    bool is_set(size_t row, size_t col) const {
        assert(row < rows && col < cols);
        const size_t idx      = row * cols + col;
        const size_t word_idx = idx / 64;
        uint64_t     word     = 0;
#if defined(GGML_GEMMINI_HAS_OPENMP)
#pragma omp atomic read
        word = words[word_idx];
#else
        word = words[word_idx];
#endif
        return (word >> (idx % 64)) & 1;
    }
};

struct BlockMask {
    size_t     bit_count = 0;
    uint64_t * words     = nullptr;

    BlockMask() = default;

    BlockMask(uint64_t * storage, size_t count) : bit_count(count), words(storage) {
        assert(words != nullptr);
    }

    static size_t word_count(size_t count) {
        return count / 64 + (count % 64 != 0 ? 1 : 0);
    }

    void clear() {
        std::fill_n(words, word_count(bit_count), uint64_t{0});
    }

    void set(size_t index) {
        assert(index < bit_count);
        words[index / 64] |= uint64_t{1} << (index % 64);
    }

    bool is_set(size_t index) const {
        assert(index < bit_count);
        return (words[index / 64] >> (index % 64)) & 1;
    }
};

struct BlockState {
    size_t               blk_size = 0;
    std::vector<int16_t> e;
    int16_t e1      = std::numeric_limits<int16_t>::min(); // top-1 distinct exponent of a block
    int16_t e2      = std::numeric_limits<int16_t>::min(); // top-2 distinct exponent of a block
    int16_t e_b     = std::numeric_limits<int16_t>::min(); // final max exponent of a block
    int16_t theta_b = std::numeric_limits<int16_t>::min(); // final scale exponent of a block

    std::vector<float> x; // temporary original activations for a block

    bool prepare(size_t block_size) {
        blk_size = block_size;
        if (e.size() < block_size)
            e.resize(block_size);
        if (x.size() < block_size)
            x.resize(block_size);
        reset();
        return true;
    }

    void reset() {
        e1      = std::numeric_limits<int16_t>::min();
        e2      = std::numeric_limits<int16_t>::min();
        e_b     = std::numeric_limits<int16_t>::min();
        theta_b = std::numeric_limits<int16_t>::min();
    }
};

struct StripeScratch {
    BlockState block;
    BitMask    folding_inlier_mask;
#if GGML_GEMMINI_ACT_QUANT_METRICS
    bool actual_requantized = false;
#endif
#if EXSIA_VALIDATION
    struct ReferenceScratch {
        std::vector<int32_t>  q_tmp;
        std::vector<int32_t>  q_final;
        int16_t               p0_e1    = std::numeric_limits<int16_t>::min();
        int16_t               p0_e2    = std::numeric_limits<int16_t>::min();
        int16_t               p0_e_pre = std::numeric_limits<int16_t>::min();
        std::vector<uint64_t> p0_top_mask_words;
        __int128_t            p1_S  = 0;
        __int128_t            p1_SS = 0;
        size_t                p1_N  = 0;

        void prepare(size_t block_size) {
            if (q_tmp.size() < block_size)
                q_tmp.resize(block_size);
            if (q_final.size() < block_size)
                q_final.resize(block_size);
            if (p0_top_mask_words.size() < BlockMask::word_count(block_size))
                p0_top_mask_words.resize(BlockMask::word_count(block_size));
        }
    } reference;
#endif

    bool prepare(size_t block_size) {
        if (!block.prepare(block_size) || !folding_inlier_mask.prepare(1, block_size)) {
            return false;
        }

#if EXSIA_VALIDATION
        reference.prepare(block_size);
#endif
        return true;
    }

    void reset() {
        block.reset();
        folding_inlier_mask.clear_active_bits();
    }
};

struct StripeState {
    size_t row_start       = 0; // starting row index of a stripe
    size_t row_end         = 0; // ending row index of a stripe (exclusive)
    size_t blk_num         = 0; // number of blocks in a stripe
    size_t num_real_blocks = 0; // number of real blocks in a stripe (excluding padding blocks)

    int16_t e1 =
        std::numeric_limits<int16_t>::min(); // top-1 distinct exponent among all blocks in a stripe
    int16_t e2 =
        std::numeric_limits<int16_t>::min(); // top-2 distinct exponent among all blocks in a stripe
    int16_t e_s = std::numeric_limits<int16_t>::min(); // final scale exponent of a stripe
    bool    promote_top_block =
        false; // set by StripeFolding when e2 != -inf; promotes top-1 exponent blocks to outliers

    BitMask       outlier_mask; // bitmask indicating outlier positions in a stripe
    StripeScratch scratch;
#if EXSIA_STAGE_PROFILE_ENABLED
    uint64_t selected_positions = 0; // final logical mask positions, including zero residuals
    uint64_t residual_nnz       = 0;
#endif

    size_t row_count() const {
        return row_end - row_start;
    }

    size_t local_row(size_t global_row) const {
        assert(global_row >= row_start && global_row < row_end);
        return global_row - row_start;
    }
};

enum class StripePipelineSlotState : uint8_t {
    Released,
    Acquired,
    LocalFilled,
    FoldingCommitted,
};

// Owns every mutable value for one in-flight stripe. Separate slots keep Local(s+1)
// q_wide and block masks disjoint from Folding(s) reads and output row writes.
struct StripePipelineSlot {
    size_t                  stripe_idx = 0;
    size_t                  row_start  = 0;
    size_t                  row_end    = 0;
    StripePipelineSlotState lifecycle  = StripePipelineSlotState::Released;

    StripeState stripe; // canonical stripe state, mask, and folding scratch

    std::vector<int32_t>  q_wide;    // slot-owned local output, max_stripe_rows * K_padded
    std::vector<int16_t>  block_exp; // slot-owned local exponents, max_stripe_rows * blocks_per_row
    std::vector<uint64_t> block_mask_words; // slot-owned storage for per-block BlockMask views
    size_t                block_mask_words_per_block = 0;
    size_t                active_block_count         = 0;
    ggml::gemmini::residual::TimedResidualCapture      rmd_builder; // selected once per run
    ggml::gemmini::rmd::StripePacketHandle             rmd_packet;
    ggml::gemmini::residual::DirectStripePayloadHandle direct_residual; // sealed at folding commit
    uint64_t                                           rmd_pack_ns = 0; // packet seal duration
    uint64_t                                           folding_commit_ns     = 0;
    uint64_t                                           quantization_start    = 0;
    uint64_t                                           quantization_end      = 0;
    uint64_t                                           quantization_start_ns = 0;
    uint64_t                                           quantization_end_ns   = 0;

#if EXSIA_BRANCH_COUNTS_ENABLED
    StripeCycleStats cycle_stats;
#endif

    // One-time capacity allocation sized for the largest stripe. Counts are
    // overflow-checked by the caller before this is invoked.
    bool prepare(size_t elem_capacity,
                 size_t max_block_count,
                 size_t max_rows,
                 size_t K_padded,
                 size_t block_size) {
        const size_t words_per_block          = BlockMask::word_count(block_size);
        size_t       block_mask_word_capacity = 0;
        if (!BitMask::checked_mul(max_block_count, words_per_block, block_mask_word_capacity))
            return false;

        if (q_wide.size() < elem_capacity)
            q_wide.resize(elem_capacity);
        if (block_exp.size() < max_block_count)
            block_exp.resize(max_block_count);
        if (block_mask_words.size() < block_mask_word_capacity)
            block_mask_words.resize(block_mask_word_capacity);
        block_mask_words_per_block = words_per_block;
        if (!stripe.outlier_mask.prepare(max_rows, K_padded) || !stripe.scratch.prepare(block_size))
            return false;

        return true;
    }

    void reset_for_run() {
        const int16_t neg_inf    = std::numeric_limits<int16_t>::min();
        lifecycle                = StripePipelineSlotState::Released;
        stripe_idx               = 0;
        row_start                = 0;
        row_end                  = 0;
        active_block_count       = 0;
        stripe.row_start         = 0;
        stripe.row_end           = 0;
        stripe.e1                = neg_inf;
        stripe.e2                = neg_inf;
        stripe.e_s               = neg_inf;
        stripe.promote_top_block = false;
        stripe.outlier_mask.rows = 0;
        stripe.outlier_mask.cols = 0;
        stripe.scratch.reset();
        rmd_packet.reset();
        direct_residual.reset();
        rmd_pack_ns           = 0;
        folding_commit_ns     = 0;
        quantization_start    = 0;
        quantization_end      = 0;
        quantization_start_ns = 0;
        quantization_end_ns   = 0;
#if EXSIA_BRANCH_COUNTS_ENABLED
        cycle_stats.reset();
#endif
    }

    void acquire(size_t idx) {
        assert(lifecycle == StripePipelineSlotState::Released);
        stripe_idx = idx;
        lifecycle  = StripePipelineSlotState::Acquired;
    }

    // Active-range reset only: no allocation, resize, reserve, or BitMask alloc.
    void
    reset_for_stripe(size_t idx, size_t r0, size_t r1, size_t K_padded, size_t blocks_per_row) {
        const int16_t neg_inf = std::numeric_limits<int16_t>::min();
        assert(lifecycle == StripePipelineSlotState::Acquired);
        stripe_idx = idx;
        row_start  = r0;
        row_end    = r1;

        const size_t rows                  = r1 - r0;
        const size_t elem                  = rows * K_padded;
        const size_t blocks                = rows * blocks_per_row;
        const size_t block_mask_word_count = blocks * block_mask_words_per_block;

        stripe.row_start         = r0;
        stripe.row_end           = r1;
        stripe.e1                = neg_inf;
        stripe.e2                = neg_inf;
        stripe.e_s               = neg_inf;
        stripe.promote_top_block = false;
        assert(stripe.outlier_mask.rows >= rows);
        assert(stripe.outlier_mask.cols >= K_padded);
        stripe.outlier_mask.rows = rows;
        stripe.outlier_mask.cols = K_padded;
        stripe.scratch.reset();

        std::fill(q_wide.begin(), q_wide.begin() + elem, 0);
        std::fill(block_exp.begin(), block_exp.begin() + blocks, neg_inf);
        assert(block_mask_word_count <= block_mask_words.size());
        std::fill_n(block_mask_words.begin(), block_mask_word_count, uint64_t{0});
        active_block_count = blocks;
        rmd_packet.reset();
        direct_residual.reset();
        rmd_pack_ns           = 0;
        folding_commit_ns     = 0;
        quantization_start    = 0;
        quantization_end      = 0;
        quantization_start_ns = 0;
        quantization_end_ns   = 0;
#if EXSIA_BRANCH_COUNTS_ENABLED
        cycle_stats.reset();
#endif
    }

    BlockMask block_mask(size_t block_idx, size_t block_size) {
        assert(block_idx < active_block_count);
        assert(block_mask_words_per_block == BlockMask::word_count(block_size));
        const size_t offset = block_idx * block_mask_words_per_block;
        assert(offset + block_mask_words_per_block <= block_mask_words.size());
        return BlockMask(block_mask_words.data() + offset, block_size);
    }

    void mark_local_filled() {
        assert(lifecycle == StripePipelineSlotState::Acquired);
        lifecycle = StripePipelineSlotState::LocalFilled;
    }

    void mark_quantization_started(uint64_t start, uint64_t start_ns = 0) {
        assert(lifecycle == StripePipelineSlotState::Acquired);
        quantization_start    = start;
        quantization_start_ns = start_ns;
    }

    void mark_folding_committed(uint64_t commit_ns = 0, uint64_t quantization_end_tick = 0) {
        assert(lifecycle == StripePipelineSlotState::LocalFilled);
        folding_commit_ns   = commit_ns;
        quantization_end    = quantization_end_tick;
        quantization_end_ns = commit_ns;
        lifecycle           = StripePipelineSlotState::FoldingCommitted;
    }

    void release() {
        assert(lifecycle == StripePipelineSlotState::FoldingCommitted);
        lifecycle = StripePipelineSlotState::Released;
    }
};

struct LocalWorkerContext {
    size_t        worker_idx  = 0;
    size_t        row_start   = 0;
    size_t        row_end     = 0;
    size_t        block_start = 0;
    size_t        block_end   = 0;
    StripeScratch scratch; // worker-private reusable Local block state

    bool prepare(size_t block_size) {
        return scratch.prepare(block_size);
    }

    void reset_for_run() {
        row_start   = 0;
        row_end     = 0;
        block_start = 0;
        block_end   = 0;
        scratch.reset();
    }

    void reset_for_stripe(size_t stripe_idx, size_t r0, size_t r1, size_t blocks_per_row) {
        const size_t rows         = r1 - r0;
        const size_t worker_count = EXSIA_LOCAL_WORKER_COUNT;
        (void)stripe_idx;
        const size_t worker = worker_idx;
        row_start           = r0 + (rows * worker) / worker_count;
        row_end             = r0 + (rows * (worker + 1)) / worker_count;
        block_start         = (row_start - r0) * blocks_per_row;
        block_end           = (row_end - r0) * blocks_per_row;
        assert(row_start <= row_end);
        assert(block_start <= block_end);
        scratch.reset();
    }

    bool owns_row(size_t row) const {
        return row >= row_start && row < row_end;
    }
};

struct LocalTaskRuntime {
    bool completed = false;
#if EXSIA_BRANCH_COUNTS_ENABLED
    StripeCycleStats cycle_stats;
#endif

    void reset() noexcept {
        completed = false;
#if EXSIA_BRANCH_COUNTS_ENABLED
        cycle_stats.reset();
#endif
    }
};

struct LocalExecutionWorkspace {
    std::array<LocalWorkerContext, EXSIA_LOCAL_WORKER_COUNT> workers;
    std::array<LocalTaskRuntime, EXSIA_LOCAL_WORKER_COUNT>   local_tasks;
    size_t                                                   active_block_count = 0;

    bool prepare(size_t, size_t block_size) {
        for (size_t i = 0; i < workers.size(); ++i) {
            workers[i].worker_idx = i;
            if (!workers[i].prepare(block_size))
                return false;
        }
        return true;
    }

    void reset_for_run() {
        for (LocalWorkerContext & worker : workers)
            worker.reset_for_run();
        for (LocalTaskRuntime & task : local_tasks)
            task.reset();
        active_block_count = 0;
    }

    void reset_for_stripe(size_t stripe_idx, size_t r0, size_t r1, size_t blocks_per_row) {
        const size_t rows  = r1 - r0;
        active_block_count = rows * blocks_per_row;

        for (size_t i = 0; i < workers.size(); ++i) {
            workers[i].worker_idx = i;
            workers[i].reset_for_stripe(stripe_idx, r0, r1, blocks_per_row);
            local_tasks[i].reset();
        }
    }
};

struct LocalParallelTaskRecord {
#if EXSIA_OBSERVATION_ENABLED
    size_t task_id               = 0;
    size_t row_start             = 0;
    size_t row_end               = 0;
    size_t block_start           = 0;
    size_t block_end             = 0;
    size_t populated_block_count = 0;
    bool   empty                 = true;
    bool   short_task            = false;
    bool   completed             = false;
#endif
};

struct LocalParallelStripeObservation {
    size_t                                                        stripe_idx           = 0;
    size_t                                                        observed_team_size   = 0;
    size_t                                                        scheduled_task_count = 0;
    size_t                                                        completed_task_count = 0;
    std::array<LocalParallelTaskRecord, EXSIA_LOCAL_WORKER_COUNT> tasks;
};

struct ExSIAState {
    enum class ExecutionMode : uint8_t {
        Sequential,
        LocalParallel,
        LocalFoldingPipeline,
    };

    using FailureCode = exsia::FailureCode;

    struct ExecutionModeAvailability {
        ExecutionMode mode;
        const char *  label;
        bool          available;
        const char *  status;
        const char *  reason;
    };

#if EXSIA_VALIDATION && EXSIA_PROFILE_COLLECTION_ENABLED
    struct ProfileSnapshot {
        uint64_t                         run_id = 0;
        ExecutionMode                    mode   = ExecutionMode::Sequential;
        ProfileInterval                  run;
        std::vector<StripeProfileRecord> stripes;
    };
#endif

    ExecutionMode            mode              = ExecutionMode::Sequential;
    static constexpr size_t  no_failure_stripe = std::numeric_limits<size_t>::max();
    FailureCode              failure_code      = FailureCode::None;
    size_t                   failure_stripe    = no_failure_stripe;
    uint64_t                 run_id            = 0;
    size_t                   B_size            = BLOCK_SIZE;
    size_t                   K_logical         = 0;
    size_t                   K_padded          = 0;
    size_t                   blocks_per_row    = 0;
    std::vector<StripeState> stripe;    // validation mask snapshots and stripe row metadata
    std::vector<float>       x_f32;     // original activations, padded to I * K_padded
    std::vector<int32_t>     q_wide;    // unused public scratch; slot-owned q_wide is live
    std::vector<int16_t>     block_exp; // unused public scratch; slot-owned block_exp is live
    std::vector<int32_t>     residual;  // global residual output, I * K_padded
#if EXSIA_OBSERVATION_ENABLED
    std::vector<LocalParallelStripeObservation> local_parallel_observations;
#endif
#if EXSIA_VALIDATION
    std::array<uint64_t, 3> validation_p3_branch_counts{};
#endif
#if EXSIA_VALIDATION && EXSIA_PROFILE_COLLECTION_ENABLED
    ProfileSnapshot profile_snapshot;
#endif
};

} // namespace ggml::gemmini::quants::act::exsia
