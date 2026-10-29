#include "stripe.hpp"

#include "log.hpp"
#include "options.hpp"
#include "residual/stripe-residual.hpp"

#include "potal_hw_params.h"

#include "quants/act/quantize.hpp"
#include "quants/common/hp1_scu.hpp"

#include <gemmini/trace-context.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <new>
#include <utility>

namespace ggml::potal {

namespace {

namespace rmd = ggml::gemmini::rmd;

constexpr size_t   HW_DIM        = POTAL_HW_DIM; // not DIM: <gemmini_params.h> defines that macro
constexpr uint32_t SCALE_BLOCK_K = quants::weight::hp1::block_k;

#ifndef GGML_GEMMINI_EXSIA_DEFAULT_MODE_VALUE
#    define GGML_GEMMINI_EXSIA_DEFAULT_MODE_VALUE 0
#endif
// ExSIA SEQUENTIAL quantizes on the producer alone: pin it to one CPU. The
// OpenMP modes start their team from it: keep every CPU but the issuer's.
constexpr bool producer_on_one_cpu = GGML_GEMMINI_EXSIA_DEFAULT_MODE_VALUE == 0;

uint64_t now_ns() {
    return (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

// The producer's view of a run while ExSIA quantizes: StripeReadySink::user_data.
struct sink_binding {
    stripe_run *                run;
    const ggml_gemmini_args_t * args;

    static bool on_ready(void * user, const exsia::StripeReadyEvent & e) {
        const auto * b = static_cast<const sink_binding *>(user);
        return b->run->submit(e, *b->args);
    }
};

} // namespace

const char * slot_state_name(slot_state s) {
    switch (s) {
        case slot_state::free:              return "FREE";
        case slot_state::reserved:          return "RESERVED";
        case slot_state::ready:             return "READY";
        case slot_state::main_running:      return "MAIN_RUNNING";
        case slot_state::main_done:         return "MAIN_DONE";
        case slot_state::residual_prepare:  return "RESIDUAL_PREPARE";
        case slot_state::residual_running:  return "RESIDUAL_RUNNING";
        case slot_state::residual_done:     return "RESIDUAL_DONE";
        case slot_state::reconstruct_merge: return "RECONSTRUCT_MERGE";
        case slot_state::semantic_done:     return "SEMANTIC_DONE";
        case slot_state::paired_prepare:    return "PAIRED_PREPARE";
        case slot_state::paired_running:    return "PAIRED_RUNNING";
        case slot_state::paired_done:       return "PAIRED_DONE";
    }
    return "UNKNOWN";
}

const char * stripe_event_name(stripe_event_kind k) {
    switch (k) {
        case stripe_event_kind::quant_begin:             return "quant_begin";
        case stripe_event_kind::folding_commit:          return "folding_commit";
        case stripe_event_kind::queue_wait:              return "queue_wait";
        case stripe_event_kind::submit_accepted:         return "submit_accepted";
        case stripe_event_kind::main_begin:              return "main_begin";
        case stripe_event_kind::main_end:                return "main_end";
        case stripe_event_kind::residual_prepare_begin:  return "residual_prepare_begin";
        case stripe_event_kind::residual_prepare_end:    return "residual_prepare_end";
        case stripe_event_kind::residual_begin:          return "residual_begin";
        case stripe_event_kind::residual_end:            return "residual_end";
        case stripe_event_kind::radix_reconstruct_begin: return "radix_reconstruct_begin";
        case stripe_event_kind::radix_reconstruct_end:   return "radix_reconstruct_end";
        case stripe_event_kind::merge_begin:             return "merge_begin";
        case stripe_event_kind::merge_end:               return "merge_end";
        case stripe_event_kind::semantic_complete:       return "semantic_complete";
        case stripe_event_kind::capacity_release:        return "capacity_release";
        case stripe_event_kind::quant_end:               return "quant_end";
        case stripe_event_kind::run_commit:              return "run_commit";
        case stripe_event_kind::run_failed:              return "run_failed";
        case stripe_event_kind::paired_prepare_begin:    return "paired_prepare_begin";
        case stripe_event_kind::paired_prepare_end:      return "paired_prepare_end";
        case stripe_event_kind::paired_begin:            return "paired_begin";
        case stripe_event_kind::paired_end:              return "paired_end";
    }
    return "unknown";
}

struct stripe_run::slot {
    size_t     index      = 0;
    uint64_t   generation = 0; // completed reuses of this slot
    slot_state state      = slot_state::free;
    int8_t *   a          = nullptr; // [stripe rows][lda] DMA, the run's own copy of the stripe's codes
    int32_t *  c          = nullptr; // [stripe rows][ldc] DMA, Main GEMM result

    // The accepted stripe: written while RESERVED, read-only from READY to FREE.
    uint64_t                      run_id        = 0;
    size_t                        stripe_id     = 0;
    size_t                        row_begin     = 0;
    size_t                        row_end       = 0;
    size_t                        producer_slot = 0; // ExSIA pipeline slot
    exsia::StripeMetadataSnapshot meta;
    double                        activation_scale = 0; // float(2^theta), as ActivationMetadataView::scale
    rmd::StripePacketHandle       packet;
    gemmini_trace_context         trace{};
    uint64_t                      accepted_ns           = 0;
    uint64_t                      quantization_start_ns = 0;
    uint64_t                      quantization_end_ns   = 0;
    uint64_t                      folding_commit_ns     = 0;

    ~slot() {
        std::free(a);
        std::free(c);
    }
};

struct stripe_run::stripe_result {
    bool         ok     = false;
    bool         fatal  = false;
    potal_status status = POTAL_OK;
    const char * stage  = nullptr;
};

stripe_run::stripe_run(const pipeline_config & cfg, const ggml_gemmini_args_t & args, const hp1_weight & w,
                       const potal_tile_plan & plan) :
    cfg_(cfg),
    w_(w),
    plan_(plan),
    m_(args.I),
    n_(args.J),
    k_(args.K),
    stripe_rows_(plan.tile_I * HW_DIM) {}

stripe_run::~stripe_run() = default;

bool stripe_run::begin(std::string & why) {
    if (cfg_.job_capacity != 1 && cfg_.job_capacity != 2) {
        why = "job capacity must be 1 or 2";
        return false;
    }
    if (m_ == 0 || n_ == 0 || k_ == 0 || n_ > UINT32_MAX || k_ > UINT32_MAX || stripe_rows_ == 0 ||
        k_ % SCALE_BLOCK_K != 0 || m_ > SIZE_MAX / n_) {
        why = "shape outside the stripe pipeline";
        return false;
    }
    if (w_.n != n_ || w_.k != k_ || w_.b == nullptr || w_.carriers == nullptr || w_.scale == nullptr) {
        why = "prepared weight is not N x K";
        return false;
    }
    expected_ = (m_ + stripe_rows_ - 1) / stripe_rows_;
    lda_      = align_up(k_, POTAL_HW_MAX_BYTES);
    ldc_      = align_up(n_, HW_DIM);
    try {
        out_.assign(m_ * n_, 0.0f);
        for (size_t i = 0; i < cfg_.job_capacity; ++i) {
            auto s   = std::make_unique<slot>();
            s->index = i;
            s->a     = static_cast<int8_t *>(dma_alloc(stripe_rows_ * lda_));
            s->c     = static_cast<int32_t *>(dma_alloc(stripe_rows_ * ldc_ * sizeof(int32_t)));
            const bool ok = s->a != nullptr && s->c != nullptr;
            slots_.push_back(std::move(s));
            if (!ok) {
                why = "out of memory for the stripe slots";
                return false;
            }
        }
    } catch (const std::bad_alloc &) {
        why = "out of memory for the private output";
        return false;
    }
    return true;
}

std::string stripe_run::check_event(const exsia::StripeReadyEvent & e, size_t rows) const {
    if (!e.activation_metadata || e.activation_metadata->theta == std::numeric_limits<int16_t>::min()) {
        return "stripe without a post-fold theta";
    }
    const float scale = std::ldexp(1.0f, e.activation_metadata->theta);
    if (!std::isfinite(scale) || !(scale > 0.0f)) {
        return "post-fold theta gives no finite positive activation scale";
    }
    if (e.direct_residual) {
        return "cpu_direct residual payload in a STRIPE_PIPELINE run";
    }
    if (e.rmd_packet) {
        const rmd::StripePacket & p = *e.rmd_packet;
        if (!cfg_.residual) {
            return "residual packet in a run without compensation";
        }
        if (p.stripe_id != e.stripe_id || p.row_begin != e.row_begin || p.row_count != rows || p.logical_j != n_ ||
            p.logical_k != k_) {
            return "residual packet does not belong to this stripe";
        }
    }
    return {};
}

std::string stripe_run::check_order_locked(const exsia::StripeReadyEvent & e, size_t rows) const {
    if (run_bound_ && e.run_id != run_id_) {
        return "stripe of another ExSIA run";
    }
    if (e.stripe_id != next_stripe_) {
        return "stripe " + std::to_string(e.stripe_id) + " out of order or duplicated (expected " +
               std::to_string(next_stripe_) + ")";
    }
    if (e.row_begin != next_row_ || rows == 0 || e.row_end > m_) {
        return "stripe rows [" + std::to_string(e.row_begin) + ", " + std::to_string(e.row_end) +
               ") out of order or out of range";
    }
    if (e.stripe_id >= expected_ || (e.row_end != m_ ? rows != stripe_rows_ : rows > stripe_rows_)) {
        return "stripe height differs from the frozen Main plan (" + std::to_string(stripe_rows_) + " rows)";
    }
    return {};
}

void stripe_run::fail_locked(const char * stage, potal_status status, bool fatal, std::string detail) {
    if (!failed_) {
        failed_       = true;
        first_stage_  = stage;
        first_status_ = status;
        first_detail_ = std::move(detail);
    }
    fatal_ = fatal_ || fatal;
    cv_.notify_all();
}

void stripe_run::fail(const char * stage, potal_status status, bool fatal, std::string detail) {
    std::lock_guard<std::mutex> lock(mutex_);
    fail_locked(stage, status, fatal, std::move(detail));
}

void stripe_run::set_state(slot & s, slot_state state) {
    std::lock_guard<std::mutex> lock(mutex_);
    s.state = state;
}

void stripe_run::stamp_locked(stripe_event & ev) {
    ev.sequence    = next_event_sequence_++;
    ev.t_ns        = now_ns();
    ev.outstanding = outstanding_;
}

stripe_event stripe_run::slot_event_locked(stripe_event_kind kind, const slot & s) {
    stripe_event ev{ kind };
    ev.run_bound       = true;
    ev.run_id          = s.run_id;
    ev.stripe_id       = s.stripe_id;
    ev.slot            = s.index;
    ev.slot_generation = s.generation;
    ev.row_begin       = s.row_begin;
    ev.row_end         = s.row_end;
    stamp_locked(ev);
    return ev;
}

stripe_event stripe_run::event_of(stripe_event_kind kind, const slot & s) {
    std::lock_guard<std::mutex> lock(mutex_);
    return slot_event_locked(kind, s);
}

stripe_event stripe_run::advance(slot & s, slot_state state, stripe_event_kind kind) {
    std::lock_guard<std::mutex> lock(mutex_);
    s.state = state;
    return slot_event_locked(kind, s);
}

void stripe_run::emit(const stripe_event & ev) {
#if LOG_DEBUG
    {
        // The record of the step stamped just before this one comes first. Its
        // thread writes it right after releasing mutex_, so this waits for one
        // record write, never for a later step of either thread.
        std::unique_lock<std::mutex> lock(log_mutex_);
        log_cv_.wait(lock, [&] { return logged_sequence_ + 1 == ev.sequence; });
    }
    {
        struct next_record {
            stripe_run &   run;
            const uint64_t sequence;
            ~next_record() {
                {
                    std::lock_guard<std::mutex> lock(run.log_mutex_);
                    run.logged_sequence_ = sequence;
                }
                run.log_cv_.notify_all();
            }
        } written{ *this, ev.sequence }; // also when the record fails

        char run[32] = "-", stripe[32] = "-", slot[32] = "-", extra[768] = "";
        if (ev.run_bound) {
            std::snprintf(run, sizeof(run), "%llu", (unsigned long long) ev.run_id);
        }
        if (ev.stripe_id != no_stripe) {
            std::snprintf(stripe, sizeof(stripe), "%zu", ev.stripe_id);
        }
        if (ev.slot != no_stripe) {
            std::snprintf(slot, sizeof(slot), "%zu", ev.slot);
        }
        if (ev.gemm != nullptr) {
            const potal_stripe_gemm_report & g = *ev.gemm;
            std::snprintf(extra, sizeof(extra),
                          " gemm_rows=%zu N=%zu loop_ws=%llu validated_api=%llu fences=%llu scale_writes=%llu "
                          "scale_releases=%llu generation=%u tile_i=%u tile_j=%u tile_k=%u i0=%u j0=%u k0=%u "
                          "k_compact=%zu k_execution=%zu runs=%zu status=POTAL_%s%s%s",
                          ev.rows, ev.cols, (unsigned long long) g.loop_ws, (unsigned long long) g.validated_gemms,
                          (unsigned long long) g.fences, (unsigned long long) g.scale_writes,
                          (unsigned long long) g.scale_releases, g.last_generation, g.tile_i, g.tile_j, g.tile_k,
                          g.i0, g.j0, g.k0, ev.k_compact, ev.k_execution, ev.runs, potal_status_name(g.status),
                          g.stage ? " step=" : "", g.stage ? g.stage : "");
        } else if (ev.paired != nullptr) {
            const potal_paired_report & p = *ev.paired;
            const potal_paired_counts & c = p.counts;
            std::snprintf(extra, sizeof(extra),
                          " paired_status=%s step=%s hw_status=POTAL_%s issued=%d fatal=%d cleaned=%d main_loop_ws=%llu "
                          "residual_loop_ws=%llu index_configs=%llu index_writes=%llu scale_writes=%llu "
                          "scale_releases=%llu readbacks=%llu readback_tiles=%llu fences=%llu drains=%llu "
                          "tlb_flushes=%llu generation=%u tile_I=%zu tile_J=%zu tile_K=%zu k_panels=%zu j_groups=%zu "
                          "windows=%zu lane_rows=%zu",
                          potal_paired_status_name(p.status), p.stage ? p.stage : "-", potal_status_name(p.hw_status),
                          (int) p.issued, (int) p.fatal, (int) p.cleaned, (unsigned long long) c.main_loop_ws,
                          (unsigned long long) c.residual_loop_ws, (unsigned long long) c.index_configs,
                          (unsigned long long) c.index_writes, (unsigned long long) c.scale_writes,
                          (unsigned long long) c.scale_releases, (unsigned long long) c.readbacks,
                          (unsigned long long) c.readback_tiles, (unsigned long long) c.fences,
                          (unsigned long long) c.drains, (unsigned long long) c.tlb_flushes, p.last_generation,
                          p.tile.tile_I, p.tile.tile_J, p.tile.tile_K, p.tile.k_panels, p.tile.j_groups,
                          p.tile.residual_windows, p.tile.residual_lane_rows);
        } else if (ev.kind == stripe_event_kind::paired_prepare_end) {
            std::snprintf(extra, sizeof(extra), " route=%s active_lane_rows=%zu k_compact=%zu k_execution=%zu runs=%zu%s%s",
                          ev.route ? ev.route : "-", ev.rows, ev.k_compact, ev.k_execution, ev.runs,
                          ev.detail ? " refusal=" : "", ev.detail ? ev.detail : "");
        } else if (ev.kind == stripe_event_kind::residual_prepare_end) {
            std::snprintf(extra, sizeof(extra), " active_lane_rows=%zu k_compact=%zu k_execution=%zu runs=%zu",
                          ev.rows, ev.k_compact, ev.k_execution, ev.runs);
        } else if (ev.kind == stripe_event_kind::merge_end) {
            std::snprintf(extra, sizeof(extra), " corrections_nonzero=%zu", ev.nonzero);
        } else if (ev.kind == stripe_event_kind::submit_accepted) {
            std::snprintf(extra, sizeof(extra),
                          " producer_slot=%zu quantization_start_ns=%llu quantization_end_ns=%llu "
                          "folding_commit_ns=%llu",
                          ev.producer_slot, (unsigned long long) ev.quantization_start_ns,
                          (unsigned long long) ev.quantization_end_ns, (unsigned long long) ev.folding_commit_ns);
        } else if (ev.detail != nullptr) {
            std::snprintf(extra, sizeof(extra), " detail=%s", ev.detail);
        }
        log::debug(cfg_.layer.c_str(),
                   "[POTAL_STRIPE] event=%s event_sequence=%llu run_id=%s stripe=%s slot=%s slot_generation=%llu "
                   "rows=%zu..%zu outstanding=%zu capacity=%zu executor=POTAL/%s backend=%s t_ns=%llu%s %s",
                   stripe_event_name(ev.kind), (unsigned long long) ev.sequence, run, stripe, slot,
                   (unsigned long long) ev.slot_generation, ev.row_begin, ev.row_end, ev.outstanding,
                   cfg_.job_capacity, options::residual_executor_name(options::MatmulInvocationMode::stripe_pipeline),
                   cfg_.backend, (unsigned long long) ev.t_ns, extra, log::trace_fields().c_str());
    }
#endif
    // After the record and outside every lock of the run: an observer may
    // query or abort the run, and never holds up another step's record.
    if (cfg_.observer) {
        cfg_.observer(ev);
    }
}

void stripe_run::emit_run(stripe_event_kind kind, const char * detail) {
    stripe_event ev{ kind };
    {
        std::lock_guard<std::mutex> lock(mutex_);
        ev.run_bound = run_bound_;
        ev.run_id    = run_id_;
        stamp_locked(ev);
    }
    ev.row_end = m_;
    ev.detail  = detail;
    emit(ev);
}

bool stripe_run::submit(const exsia::StripeReadyEvent & e, const ggml_gemmini_args_t & producer_args) {
    try {
        const size_t rows = e.row_end > e.row_begin ? e.row_end - e.row_begin : 0;
        {
            stripe_event ev{ stripe_event_kind::folding_commit };
            ev.run_bound = true;
            ev.run_id    = e.run_id;
            ev.stripe_id = e.stripe_id;
            ev.row_begin = e.row_begin;
            ev.row_end   = e.row_end;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                stamp_locked(ev);
            }
            emit(ev);
        }
        std::string why = check_event(e, rows);

        std::unique_lock<std::mutex> lock(mutex_);
        if (failed_) {
            return false;
        }
        if (why.empty()) {
            why = closing_ ? std::string("stripe after the input was closed") : check_order_locked(e, rows);
        }
        if (!why.empty()) {
            fail_locked("submit", POTAL_ERR_INVALID_ARGUMENT, false, why);
            return false;
        }

        // Capacity: every non-FREE slot holds one semantic stripe.
        bool waited = false;
        while (!failed_ && outstanding_ >= cfg_.job_capacity) {
            if (!waited) {
                waited = true;
                counts_.producer_capacity_waits++;
                stripe_event ev{ stripe_event_kind::queue_wait };
                ev.run_bound = true;
                ev.run_id    = e.run_id;
                ev.stripe_id = e.stripe_id;
                ev.row_begin = e.row_begin;
                ev.row_end   = e.row_end;
                stamp_locked(ev);
                lock.unlock();
                emit(ev);
                lock.lock();
                continue; // the state may have changed while the lock was released
            }
            cv_.wait(lock);
        }
        if (failed_) {
            return false;
        }
        slot * s = nullptr;
        for (const auto & candidate : slots_) {
            if (candidate->state == slot_state::free) {
                s = candidate.get();
                break;
            }
        }
        if (s == nullptr) {
            fail_locked("submit", POTAL_OK, false, "no free slot below the capacity");
            return false;
        }
        s->state = slot_state::reserved;
        outstanding_++;
        counts_.max_outstanding_semantic = std::max<uint64_t>(counts_.max_outstanding_semantic, outstanding_);
        run_bound_ = true;
        run_id_    = e.run_id;
        next_stripe_++;
        next_row_ = e.row_end;
        lock.unlock();

        // RESERVED: this thread alone writes the slot until it is READY.
        s->run_id                = e.run_id;
        s->stripe_id             = e.stripe_id;
        s->row_begin             = e.row_begin;
        s->row_end               = e.row_end;
        s->producer_slot         = e.slot;
        s->meta                  = *e.activation_metadata;
        s->activation_scale      = (double) std::ldexp(1.0f, s->meta.theta);
        s->packet                = e.rmd_packet;
        s->trace                 = gemmini_trace_capture();
        s->quantization_start_ns = e.quantization_start_ns;
        s->quantization_end_ns   = e.quantization_end_ns;
        s->folding_commit_ns     = e.folding_commit_ns;
        const bool copied        = copy_activation_rows(producer_args, e.row_begin, rows, s->a, lda_);

        lock.lock();
        if (!copied || failed_) {
            s->state = slot_state::free;
            s->packet.reset();
            outstanding_--;
            if (!copied) {
                fail_locked("submit", POTAL_OK, false, "the stripe's activation codes are not readable");
            }
            return false;
        }
        // READY, its record stamped before the consumer can take it: the
        // stripe's main_begin always follows its submit_accepted. The queue
        // push (it may allocate) comes first, so nothing that can throw lies
        // between the stamp and the record.
        ready_.push_back(s);
        counts_.stripes_accepted++;
        s->state                 = slot_state::ready;
        stripe_event ev          = slot_event_locked(stripe_event_kind::submit_accepted, *s);
        ev.producer_slot         = s->producer_slot;
        ev.quantization_start_ns = s->quantization_start_ns;
        ev.quantization_end_ns   = s->quantization_end_ns;
        ev.folding_commit_ns     = s->folding_commit_ns;
        s->accepted_ns           = ev.t_ns;
        cv_.notify_all();
        lock.unlock();
        emit(ev);
        return true;
    } catch (const std::exception & ex) {
        fail("submit", POTAL_OK, false, std::string("exception: ") + ex.what());
        return false;
    }
}

bool stripe_run::finish() {
    std::unique_lock<std::mutex> lock(mutex_);
    closing_ = true;
    if (!failed_ && next_stripe_ != expected_) {
        fail_locked("finish", POTAL_OK, false,
                    "missing stripes: " + std::to_string(next_stripe_) + " of " + std::to_string(expected_) +
                        " accepted");
    }
    cv_.notify_all();
    cv_.wait(lock, [this] { return failed_ || completed_ == expected_; });
    return !failed_;
}

void stripe_run::abort(const char * stage, const std::string & detail) {
    std::lock_guard<std::mutex> lock(mutex_);
    closing_ = true;
    fail_locked(stage, POTAL_OK, false, detail);
}

consume_outcome stripe_run::consume(const planned_runner & exec) {
    owner_runners runners;
    runners.planned = exec;
    return consume(runners);
}

consume_outcome stripe_run::consume(const owner_runners & runners) {
    consume_outcome out;
    for (;;) {
        slot *       s = nullptr;
        stripe_event begin{ stripe_event_kind::main_begin };
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this] { return failed_ || !ready_.empty() || (closing_ && outstanding_ == 0); });
            if (failed_ || ready_.empty()) {
                break; // failed: READY stripes are dropped, nothing new starts
            }
            s = ready_.front();
            ready_.pop_front();
            s->state = slot_state::main_running;
            begin    = slot_event_locked(stripe_event_kind::main_begin, *s);
        }
        stripe_result r;
        try {
            r = execute(*s, begin, runners);
        } catch (...) {
            // Only a failure while reporting a failure gets here (no memory for
            // its text): the accelerator state is not known for sure, fail-stop.
            std::lock_guard<std::mutex> lock(mutex_);
            fail_locked("exception", POTAL_OK, true, std::string());
            r.fatal = true;
            r.stage = "exception";
        }
        if (!r.ok) {
            out.fatal  = r.fatal;
            out.status = r.status;
            out.stage  = r.stage;
            break;
        }
    }
    return out;
}

stripe_run::stripe_result stripe_run::execute(slot & s, const stripe_event & main_begin, const owner_runners & runners) {
    const planned_runner & exec = runners.planned;
    ggml::gemmini::trace::ScopedContext trace_scope(s.trace);
    stripe_result                       res;
    bool                                gemm_fatal = false; // a planned GEMM left the accelerator state unknown
    bool                                released   = false; // capacity returned: slot and counts are the run's
    pipeline_counts                     n; // this stripe's counts, merged into the run's under the lock
    const size_t                        rows = s.row_end - s.row_begin;
    stripe_completion                   done; // this stripe's record, published with its capacity
    done.stripe_id   = s.stripe_id;
    done.row_begin   = s.row_begin;
    done.row_end     = s.row_end;
    done.slot        = s.index;
    done.accepted_ns = s.accepted_ns;

    const auto merge_counts = [this, &n] {
        counts_.main_logical_gemms += n.main_logical_gemms;
        counts_.main_loop_ws += n.main_loop_ws;
        counts_.main_validated += n.main_validated;
        counts_.main_fences += n.main_fences;
        counts_.residual_nonempty_packets += n.residual_nonempty_packets;
        counts_.residual_logical_gemms += n.residual_logical_gemms;
        counts_.residual_loop_ws += n.residual_loop_ws;
        counts_.residual_validated += n.residual_validated;
        counts_.residual_fences += n.residual_fences;
        counts_.residual_compact_k += n.residual_compact_k;
        counts_.residual_execution_k += n.residual_execution_k;
        counts_.residual_active_lane_rows += n.residual_active_lane_rows;
        counts_.radix_reconstructions += n.radix_reconstructions;
        counts_.merges_completed += n.merges_completed;
        counts_.corrections_nonzero += n.corrections_nonzero;
        counts_.scale_writes += n.scale_writes;
        counts_.scale_releases += n.scale_releases;
        counts_.tlb_flushes += n.tlb_flushes;
        counts_.last_generation = n.last_generation != 0 ? n.last_generation : counts_.last_generation;
        counts_.paired_stripes += n.paired_stripes;
        counts_.paired_fallback_stripes += n.paired_fallback_stripes;
        counts_.paired_main_loop_ws += n.paired_main_loop_ws;
        counts_.paired_residual_loop_ws += n.paired_residual_loop_ws;
        counts_.paired_index_configs += n.paired_index_configs;
        counts_.paired_index_writes += n.paired_index_writes;
        counts_.paired_readbacks += n.paired_readbacks;
        counts_.paired_readback_tiles += n.paired_readback_tiles;
        counts_.paired_fences += n.paired_fences;
        counts_.paired_drains += n.paired_drains;
        counts_.paired_prefetch_loads += n.paired_prefetch_loads;
        counts_.paired_prefetch_mvins += n.paired_prefetch_mvins;
        counts_.paired_prefetch_bytes_declared += n.paired_prefetch_bytes_declared;
        counts_.paired_resident_main_loop_ws += n.paired_resident_main_loop_ws;
        counts_.paired_residual_prefetch_issue_s += n.paired_residual_prefetch_issue_s;
        counts_.scale_s += n.scale_s;
        counts_.issue_s += n.issue_s;
        counts_.fence_s += n.fence_s;
    };
    const auto account = [&n](const planned_result & pr, bool main) {
        const potal_stripe_gemm_report & g = pr.report;
        (main ? n.main_loop_ws : n.residual_loop_ws) += g.loop_ws;
        (main ? n.main_validated : n.residual_validated) += g.validated_gemms;
        (main ? n.main_fences : n.residual_fences) += g.fences;
        (main ? n.main_logical_gemms : n.residual_logical_gemms) += pr.ok ? 1 : 0;
        n.scale_writes += g.scale_writes;
        n.scale_releases += g.scale_releases;
        n.tlb_flushes += pr.tlb_flushes;
        n.last_generation = g.last_generation != 0 ? g.last_generation : n.last_generation;
        n.scale_s += g.scale_s;
        n.issue_s += g.issue_s;
        n.fence_s += g.fence_s;
    };
    // After the release (a failing capacity_release record) the slot may be
    // the producer's again: use this stripe's own id, merge nothing twice.
    const auto failed = [&](const char * stage, potal_status status, bool fatal, std::string detail) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!released) {
            merge_counts();
        }
        fail_locked(stage, status, fatal, "stripe " + std::to_string(done.stripe_id) + ": " + detail);
        res.fatal  = fatal;
        res.status = status;
        res.stage  = stage;
        return res;
    };
    const auto gemm_detail = [](const planned_result & pr) {
        return std::string("status=POTAL_") + potal_status_name(pr.status) + " step=" +
               (pr.stage != nullptr ? pr.stage : "-") + (pr.fatal ? " fatal=1" : " fatal=0");
    };

    try {
        // main_begin was stamped when the stripe left the queue.
        emit(main_begin);
        done.main_begin_ns   = main_begin.t_ns;
        done.slot_generation = main_begin.slot_generation;

        residual::request               req;        // serial route: the Residual GEMM's operands and C
        rmd::BlockScaledInt64Correction correction;
        residual::decoded_packet        decoded;    // paired route: the packet, decoded once
        potal_paired_execution_plan     routing;    // paired route: the plan that chose the route, executed as is
        potal_paired_report             paired;     // paired route: the executor's report (residual rows)
        potal_paired_geometry           diagnostic_hw;
        potal_tile_plan                 diagnostic_main;
        const potal_tile_plan *         paired_main = &plan_;
        bool                            diagnostic_index = false;
        struct dma_block {
            void * p = nullptr;
            ~dma_block() { std::free(p); }
            void abandon() { p = nullptr; } // the accelerator may still reach it
        } paired_rc;                                // paired route: Residual C, lane rows x ldc_

        // 0. Paired route (cfg_.paired): decode the packet, plan Main + Residual
        //    (potal_paired_plan_stripe, CPU, before any command) and choose the
        //    route. A capacity refusal runs the serial route below and is
        //    reported as serial_fallback; any other refusal fails the stripe.
        bool paired_route = false;
        if (cfg_.paired) {
            const char * profile_env = std::getenv("POTAL_PAIRED_PROFILE");
            const bool profile = profile_env != nullptr && std::strcmp(profile_env, "1") == 0;
            emit(advance(s, slot_state::paired_prepare, stripe_event_kind::paired_prepare_begin));
            if (!runners.paired) {
                return failed("paired", POTAL_ERR_UNSUPPORTED, false,
                              "paired route configured, but the accelerator owner runs no paired stripes");
            }
            const char * index_entries = std::getenv("POTAL_PAIRED_DIAG_INDEX_ENTRIES");
            if (index_entries != nullptr) {
                const char * window_diag = std::getenv("POTAL_PAIRED_WINDOW_DIAG");
                diagnostic_hw = potal_paired_hw_geometry();
                if (std::strcmp(index_entries, "16") != 0 || window_diag == nullptr ||
                    std::strcmp(window_diag, "1") != 0 || diagnostic_hw.dim != 16 || diagnostic_hw.index_entries < 16) {
                    return failed("paired_plan", POTAL_ERR_INVALID_ARGUMENT, false,
                                  "POTAL_PAIRED_DIAG_INDEX_ENTRIES requires value 16, POTAL_PAIRED_WINDOW_DIAG=1 and DIM16");
                }
                diagnostic_index = true;
            }
            const auto decode_start = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
            if (s.packet) {
                const rmd::RmdStatus dec = residual::decode_packet(*s.packet, w_, decoded);
                if (dec != rmd::RmdStatus::success) {
                    return failed("residual_prepare", POTAL_OK, false, rmd::rmd_status_message(dec));
                }
            }
            potal_paired_tile_plan    tile;
            std::string               why;
            const auto plan_start = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
            const double decode_ms = profile ? std::chrono::duration<double, std::milli>(plan_start - decode_start).count() : 0;
            potal_paired_status ps = potal_paired_plan_stripe(plan_, rows, n_, k_, s.packet ? &decoded : nullptr,
                                                             tile, routing, &why);
            if (ps == potal_paired_status::ok && diagnostic_index) {
                // Smaller Index Windows must not enlarge the normal paired Main K panel.
                diagnostic_main = tile.main;
                paired_main = &diagnostic_main;
                diagnostic_hw.index_entries = 16;
                ps = potal_paired_plan_stripe(*paired_main, rows, n_, k_, s.packet ? &decoded : nullptr,
                                              tile, routing, &why, diagnostic_hw);
            }
            if (ps != potal_paired_status::ok && !potal_paired_capacity_refusal(ps)) {
                return failed("paired_plan", POTAL_OK, false,
                              std::string("POTAL_PAIRED_") + potal_paired_status_name(ps) + ": " + why);
            }
            paired_route = ps == potal_paired_status::ok;
            if (!paired_route && cfg_.paired_ktile_scale_reuse) {
                return failed("paired_plan", POTAL_ERR_UNSUPPORTED, false,
                              "explicit K-tile request cannot use Serial fallback: " + why);
            }
            if (profile) {
                const double plan_ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - plan_start).count();
                std::fprintf(stderr, "[POTAL_PLAN_PROFILE] run_id=%llu stripe=%zu fallback=%d status=%s decode_ms=%.6f plan_ms=%.6f\n",
                             (unsigned long long) s.run_id, s.stripe_id, paired_route ? 0 : 1,
                             potal_paired_status_name(ps), decode_ms, plan_ms);
            }
            const std::string refusal =
                paired_route ? std::string() : std::string("POTAL_PAIRED_") + potal_paired_status_name(ps) + ": " + why;
            if (paired_route && !decoded.rows.empty()) {
                paired_rc.p = dma_alloc(align_up(decoded.rows.size(), HW_DIM) * ldc_ * sizeof(int32_t));
                if (paired_rc.p == nullptr) {
                    return failed("alloc", POTAL_OK, false, "out of memory for the paired Residual C");
                }
            }
            stripe_event ev = paired_route ? event_of(stripe_event_kind::paired_prepare_end, s)
                                           : advance(s, slot_state::main_running, stripe_event_kind::paired_prepare_end);
            ev.route       = paired_route ? "paired" : "serial_fallback";
            ev.detail      = paired_route ? nullptr : refusal.c_str();
            ev.rows        = decoded.rows.size();
            ev.cols        = n_;
            ev.k_compact   = decoded.k_compact;
            ev.k_execution = paired_route ? tile.fragments.size() * HW_DIM : 0;
            ev.runs        = decoded.runs.size();
            emit(ev);
            n.paired_fallback_stripes += paired_route ? 0 : 1;
        }

        if (paired_route) {
            // 1+2. Main and Residual GEMMs as one paired execution on the owner thread:
            //      planned again (deterministic), staged, issued, read back, drained.
            potal_paired_request pq;
            pq.run_id       = s.run_id;
            pq.stripe_id    = s.stripe_id;
            pq.row_begin    = s.row_begin;
            pq.rows         = rows;
            pq.n            = n_;
            pq.k            = k_;
            pq.main_plan    = paired_main;
            pq.a            = s.a;
            pq.lda          = lda_;
            pq.weight       = &w_;
            pq.residual     = decoded.rows.empty() ? nullptr : &decoded;
            pq.c_main       = s.c;
            pq.ldc_main     = ldc_;
            pq.c_residual   = static_cast<int32_t *>(paired_rc.p);
            pq.ldc_residual = ldc_;
            pq.tlb_flush    = true;
            pq.plan         = &routing; // validated again by the executor, never re-planned
            pq.hw           = diagnostic_index ? &diagnostic_hw : nullptr;
            pq.conservative_fences = cfg_.paired_conservative_fences;
            pq.ktile_scale_reuse = cfg_.paired_ktile_scale_reuse;
            pq.next_k_prefetch = cfg_.paired_next_k_prefetch;
            emit(advance(s, slot_state::paired_running, stripe_event_kind::paired_begin));
            const potal_paired_status pst = runners.paired(pq, paired);
            gemm_fatal                    = gemm_fatal || paired.fatal;
            if (paired.fatal) {
                paired_rc.abandon();
            }
            const potal_paired_counts & pc = paired.counts;
            n.paired_main_loop_ws += pc.main_loop_ws;
            n.paired_residual_loop_ws += pc.residual_loop_ws;
            n.paired_index_configs += pc.index_configs;
            n.paired_index_writes += pc.index_writes;
            n.paired_readbacks += pc.readbacks;
            n.paired_readback_tiles += pc.readback_tiles;
            n.paired_fences += pc.fences;
            n.paired_drains += pc.drains;
            n.paired_prefetch_loads += pc.prefetch_loads;
            n.paired_prefetch_mvins += pc.prefetch_mvins;
            n.paired_prefetch_bytes_declared += pc.prefetch_bytes_declared;
            n.paired_resident_main_loop_ws += pc.resident_main_loop_ws;
            n.paired_residual_prefetch_issue_s += pc.residual_prefetch_issue_s;
            n.scale_writes += pc.scale_writes;
            n.scale_releases += pc.scale_releases;
            n.tlb_flushes += pc.tlb_flushes;
            n.scale_s += pc.scale_s;
            n.issue_s += pc.issue_s;
            n.fence_s += pc.fence_s;
            n.last_generation = paired.last_generation != 0 ? paired.last_generation : n.last_generation;
            {
                stripe_event ev = pst == potal_paired_status::ok
                                      ? advance(s, slot_state::paired_done, stripe_event_kind::paired_end)
                                      : event_of(stripe_event_kind::paired_end, s);
                ev.paired      = &paired;
                ev.rows        = rows;
                ev.cols        = n_;
                ev.k_compact   = decoded.k_compact;
                ev.k_execution = paired.tile.fragments.size() * HW_DIM;
                ev.runs        = decoded.runs.size();
                if (pst == potal_paired_status::ok) {
                    ev.c             = s.c;
                    ev.ldc           = ldc_;
                    ev.residual_c    = static_cast<const int32_t *>(paired_rc.p);
                    ev.residual_ldc  = ldc_;
                    ev.residual_rows = paired.plan.residual_rows.size();
                }
                emit(ev);
            }
            if (pst != potal_paired_status::ok) {
                return failed("paired", paired.hw_status, paired.fatal,
                              std::string("paired status=POTAL_PAIRED_") + potal_paired_status_name(pst) + " step=" +
                                  (paired.stage != nullptr ? paired.stage : "-") + " hw=POTAL_" +
                                  potal_status_name(paired.hw_status) + (paired.fatal ? " fatal=1 " : " fatal=0 ") +
                                  paired.detail);
            }

            // Temporary FireSim numeric probe. Readbacks have already been fenced.
            const char * numeric_probe = std::getenv("POTAL_PAIRED_STAGE_DIAG");
            if (numeric_probe != nullptr && numeric_probe[0] == '1') {
                auto compare = [&](const char * stage,
                                   const int32_t * got,
                                   const std::vector<int32_t> & want,
                                   size_t count_rows) {
                    size_t bad = 0;
                    for (size_t rr = 0; rr < count_rows; ++rr) {
                        for (size_t col = 0; col < n_; ++col) {
                            const int32_t actual = got[rr * ldc_ + col];
                            const int32_t expected = want[rr * ldc_ + col];
                            if (actual != expected) {
                                if (bad < 8) {
                                    std::fprintf(stderr,
                                        "[POTAL_STAGE_DIAG] stage=%s row=%zu col=%zu "
                                        "got=%d want=%d\n",
                                        stage, rr, col, actual, expected);
                                }
                                ++bad;
                            }
                        }
                    }
                    std::fprintf(stderr,
                        "[POTAL_STAGE_DIAG] stage=%s bad=%zu total=%zu\n",
                        stage, bad, count_rows * n_);
                    std::fflush(stderr);
                };

                std::vector<int32_t> main_ref(rows * ldc_, 0);
                reference_gemm_i32(
                    s.a, lda_,
                    w_.b, w_.ldb,
                    w_.carriers, w_.cstride,
                    SCALE_BLOCK_K,
                    rows, n_, k_,
                    main_ref.data(), ldc_);
                compare("MAIN_INT32", s.c, main_ref, rows);

                if (s.packet && !decoded.rows.empty()) {
                    namespace hp1ref = ggml::gemmini::quants::hp1;

                    std::vector<int32_t> residual_ref(
                        decoded.rows.size() * ldc_, 0);

                    for (size_t rr = 0; rr < decoded.rows.size(); ++rr) {
                        for (size_t col = 0; col < n_; ++col) {
                            int32_t acc = 0;
                            bool first = true;
                            size_t compact_base = 0;

                            for (const auto & run : decoded.runs) {
                                const uint32_t carrier =
                                    w_.carriers[run.block_id * w_.cstride + col];

                                for (size_t f = 0; f < run.local_k.size();
                                     f += HW_DIM) {
                                    int32_t partial = 0;
                                    const size_t end =
                                        std::min(f + HW_DIM, run.local_k.size());

                                    for (size_t x = f; x < end; ++x) {
                                        const size_t original_k =
                                            (size_t) run.block_id * SCALE_BLOCK_K +
                                            run.local_k[x];
                                        const int32_t digit =
                                            decoded.digits[
                                                rr * decoded.k_compact +
                                                compact_base + x];
                                        const int32_t weight =
                                            w_.b[original_k * w_.ldb + col];
                                        partial += digit * weight;
                                    }

                                    const int32_t value =
                                        hp1ref::apply_validated(partial, carrier);

                                    acc = first ? value :
                                        hp1ref::accumulate(acc, value);
                                    first = false;
                                }
                                compact_base += run.local_k.size();
                            }
                            residual_ref[rr * ldc_ + col] = acc;
                        }
                    }

                    compare(
                        "RESIDUAL_INT32",
                        static_cast<const int32_t *>(paired_rc.p),
                        residual_ref, decoded.rows.size());
                }
            }

            n.paired_stripes++;
            // Packet properties, route-independent; residual_execution_k stays the serial
            // route's K32-padded K (a paired stripe executes fragments x DIM, see paired_end).
            n.residual_compact_k += decoded.k_compact;
            n.residual_active_lane_rows += decoded.rows.size();
            done.paired        = true;
            done.residual_rows = decoded.rows.size();
            done.k_compact     = decoded.k_compact;
            done.k_execution   = paired.tile.fragments.size() * HW_DIM;
        } else {
            potal_stripe_gemm_desc d = {};
            d.run_id                 = s.run_id;
            d.stripe_id              = s.stripe_id;
            d.row_begin              = s.row_begin;
            d.work_kind              = POTAL_STRIPE_WORK_MAIN;
            d.m                      = (uint32_t) rows;
            d.n                      = (uint32_t) n_;
            d.k                      = (uint32_t) k_;
            d.a                      = s.a;
            d.b                      = w_.b;
            d.c                      = s.c;
            d.lda                    = lda_;
            d.ldb                    = w_.ldb;
            d.ldc                    = ldc_;
            d.carriers               = w_.carriers;
            d.carrier_stride         = w_.cstride;
            d.scale_block_k          = SCALE_BLOCK_K;
            d.tile_i                 = (uint32_t) plan_.tile_I;
            d.tile_j                 = (uint32_t) plan_.tile_J;
            d.tile_k                 = (uint32_t) plan_.tile_K;
            const planned_result main = exec(d);
            gemm_fatal                = main.fatal;
            account(main, true);
            {
                stripe_event ev = main.ok ? advance(s, slot_state::main_done, stripe_event_kind::main_end)
                                          : event_of(stripe_event_kind::main_end, s);
                ev.gemm         = &main.report;
                ev.rows         = rows;
                ev.cols         = n_;
                if (main.ok) {
                    ev.c   = s.c;
                    ev.ldc = ldc_;
                }
                emit(ev);
            }
            if (!main.ok) {
                return failed("main", main.status, main.fatal, "Main GEMM " + gemm_detail(main));
            }

            // 2. Residual GEMM of the sealed packet, on the same accelerator.
            if (s.packet) {
                emit(advance(s, slot_state::residual_prepare, stripe_event_kind::residual_prepare_begin));
                // In this Serial branch, cfg_.paired means a recognized capacity
                // refusal after a successful decode of this same immutable packet.
                const rmd::RmdStatus built = cfg_.paired
                    ? residual::build_request_from_decoded(std::move(decoded), w_, req)
                    : residual::build_request(*s.packet, w_, req);
                if (built != rmd::RmdStatus::success) {
                    return failed("residual_prepare", POTAL_OK, false, rmd::rmd_status_message(built));
                }
                {
                    stripe_event ev = event_of(stripe_event_kind::residual_prepare_end, s);
                    ev.rows         = req.m;
                    ev.cols         = n_;
                    ev.k_compact    = req.k_compact;
                    ev.k_execution  = req.k_execution;
                    ev.runs         = req.runs.size();
                    emit(ev);
                }
                emit(advance(s, slot_state::residual_running, stripe_event_kind::residual_begin));
                potal_stripe_gemm_desc r = d;
                r.work_kind              = POTAL_STRIPE_WORK_RESIDUAL;
                r.m                      = (uint32_t) req.m;
                r.k                      = (uint32_t) req.k_execution;
                r.a                      = req.a;
                r.b                      = req.b;
                r.c                      = req.c;
                r.lda                    = req.lda;
                r.ldb                    = req.ldb;
                r.ldc                    = req.ldc;
                r.carriers               = req.carriers;
                r.carrier_stride         = req.cstride;
                r.tile_i                 = (uint32_t) req.plan.tile_I;
                r.tile_j                 = (uint32_t) req.plan.tile_J;
                r.tile_k                 = (uint32_t) req.plan.tile_K;
                const planned_result residual_gemm = exec(r);
                gemm_fatal                         = gemm_fatal || residual_gemm.fatal;
                if (residual_gemm.fatal) {
                    req.abandon(); // the accelerator may still reach them
                }
                account(residual_gemm, false);
                n.residual_compact_k += req.k_compact;
                n.residual_execution_k += req.k_execution;
                n.residual_active_lane_rows += req.m;
                done.residual_rows = req.m;
                done.k_compact     = req.k_compact;
                done.k_execution   = req.k_execution;
                {
                    stripe_event ev = residual_gemm.ok
                                          ? advance(s, slot_state::residual_done, stripe_event_kind::residual_end)
                                          : event_of(stripe_event_kind::residual_end, s);
                    ev.gemm         = &residual_gemm.report;
                    ev.rows         = req.m;
                    ev.cols         = n_;
                    ev.k_compact    = req.k_compact;
                    ev.k_execution  = req.k_execution;
                    ev.runs         = req.runs.size();
                    if (residual_gemm.ok) {
                        ev.c   = req.c;
                        ev.ldc = req.ldc;
                    }
                    emit(ev);
                }
                if (!residual_gemm.ok) {
                    return failed("residual", residual_gemm.status, residual_gemm.fatal,
                                  "Residual GEMM " + gemm_detail(residual_gemm));
                }
            }
        }

        // 3. CPU: Main reconstruction into the private output, then the
        //    residual radix reconstruction and checked merge.
        set_state(s, slot_state::reconstruct_merge);
        float * out = out_.data() + s.row_begin * n_;
        for (size_t r = 0; r < rows; ++r) {
            reconstruct_row(s.c + r * ldc_, n_, s.activation_scale, w_.scale, out + r * n_, 1);
        }
        if (s.packet) {
            emit(event_of(stripe_event_kind::radix_reconstruct_begin, s));
            const rmd::RmdStatus composed =
                paired_route ? residual::reconstruct_rows(*s.packet, paired.plan.residual_rows,
                                                          static_cast<const int32_t *>(paired_rc.p), ldc_, n_, correction)
                             : residual::reconstruct(*s.packet, req, correction);
            if (composed != rmd::RmdStatus::success) {
                return failed("radix_reconstruct", POTAL_OK, false, rmd::rmd_status_message(composed));
            }
            n.radix_reconstructions++;
            {
                stripe_event ev = event_of(stripe_event_kind::radix_reconstruct_end, s);
                ev.correction   = correction.values.data();
                ev.rows         = rows;
                ev.cols         = n_;
                emit(ev);
            }
            emit(event_of(stripe_event_kind::merge_begin, s));
            // The prepared weight (validated once, immutable) gives the channel
            // scales: no HP1 metadata pass per stripe.
            size_t               nonzero = 0;
            const rmd::RmdStatus merged  = residual::merge_prepared_hp1(w_, s.row_begin, s.row_end, s.meta, correction,
                                                                       out_.data(), m_, n_, 1, &nonzero);
            if (merged != rmd::RmdStatus::success) {
                return failed("merge", POTAL_OK, false, rmd::rmd_status_message(merged));
            }
            n.merges_completed++;
            n.residual_nonempty_packets++;
            n.corrections_nonzero += nonzero;
            done.residual            = true;
            done.corrections_nonzero = nonzero;
            {
                stripe_event ev = event_of(stripe_event_kind::merge_end, s);
                ev.out          = out;
                ev.out_stride   = n_;
                ev.rows         = rows;
                ev.cols         = n_;
                ev.nonzero      = nonzero;
                emit(ev);
            }
        }
        {
            stripe_event ev = advance(s, slot_state::semantic_done, stripe_event_kind::semantic_complete);
            ev.out          = out;
            ev.out_stride   = n_;
            ev.rows         = rows;
            ev.cols         = n_;
            emit(ev);
            done.semantic_done_ns = ev.t_ns;
        }

        // 4. Capacity back only now: after Main, Residual and the checked merge.
        //    Stamped with the release, before a waiting producer can reserve
        //    the slot: capacity_release(s) precedes submit_accepted(s + capacity).
        stripe_event release{ stripe_event_kind::capacity_release };
        {
            std::lock_guard<std::mutex> lock(mutex_);
            completions_.push_back(done); // may allocate: before anything changes
            merge_counts();
            counts_.stripes_semantic_completed++;
            s.state = slot_state::free;
            s.packet.reset();
            s.generation++;
            outstanding_--;
            completed_++;
            released = true;
            release  = slot_event_locked(stripe_event_kind::capacity_release, s);
            cv_.notify_all();
        }
        emit(release);
        res.ok = true;
        return res;
    } catch (const std::exception & ex) {
        // Drained unless a GEMM above reported fatal (then still unknown).
        return failed("exception", POTAL_OK, gemm_fatal, ex.what());
    }
}

std::vector<stripe_completion> stripe_run::poll_completed() {
    std::lock_guard<std::mutex>    lock(mutex_);
    std::vector<stripe_completion> polled;
    polled.swap(completions_);
    return polled;
}

pipeline_report stripe_run::report() const {
    std::lock_guard<std::mutex> lock(mutex_);
    pipeline_report r;
    r.ok               = !failed_ && closing_ && expected_ != 0 && completed_ == expected_;
    r.fatal            = fatal_;
    r.status           = first_status_;
    r.stage            = first_stage_;
    r.detail           = first_detail_;
    r.run_bound        = run_bound_;
    r.run_id           = run_id_;
    r.expected_stripes = expected_;
    r.counts           = counts_;
    return r;
}

void stripe_run::abandon() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto & s : slots_) {
        s->a = nullptr;
        s->c = nullptr;
    }
}

pipeline_report run_stripe_pipeline(const ggml_tensor * src1, ggml_gemmini_args_t & args, const hp1_weight & w,
                                    const potal_tile_plan & plan, const pipeline_config & cfg, stripe_owner & owner) {
    pipeline_report r;
    const auto refuse = [&r](const char * stage, std::string detail) {
        r.stage  = stage;
        r.detail = std::move(detail);
        return r;
    };
    // Supported combination (options.hpp) and the frozen Main plan.
    if (options::activation_quant != options::ActivationQuantAlgo::EXSIA) {
        return refuse("options", "STRIPE_PIPELINE needs GGML_GEMMINI_ACTIVATION_QUANT=EXSIA");
    }
    if (cfg.job_capacity != 1 && cfg.job_capacity != 2) {
        return refuse("options", "STRIPE_PIPELINE runs 1 or 2 semantic stripes in flight");
    }
    if (cfg.residual != (options::compensation && args.residual_route ==
                                                      ggml::gemmini::residual::ResidualRoute::ws_packet) ||
        (options::compensation && !cfg.residual)) {
        return refuse("options", "STRIPE_PIPELINE with compensation=ON needs the ws_packet residual route");
    }
    if (plan.tile_I == 0 || args.tile_I != plan.tile_I || args.tile_J != plan.tile_J || args.tile_K != plan.tile_K ||
        args.activation_rows_per_stripe != plan.tile_I * HW_DIM) {
        return refuse("geometry", "the arguments do not carry the frozen Main plan");
    }
    if (args.exsia_stripe_ready_sink != nullptr) {
        return refuse("options", "a stripe-ready sink is already attached");
    }
    if (cfg.paired && !cfg.residual) {
        return refuse("options", "the paired route needs compensation=ON (the ws_packet residual)");
    }
    if (cfg.paired_ktile_scale_reuse && (!cfg.paired || HW_DIM != 32)) {
        return refuse("options", "K-tile Scale reuse requires DIM32 Paired execution");
    }
    if (cfg.paired_next_k_prefetch && !cfg.paired_ktile_scale_reuse) {
        return refuse("options", "Next-K prefetch requires K-tile Scale reuse");
    }

    std::shared_ptr<stripe_run> run;
    std::string                 why;
    try {
        run = std::make_shared<stripe_run>(cfg, args, w, plan);
    } catch (const std::bad_alloc &) {
        return refuse("alloc", "out of memory for the stripe run");
    }
    if (!run->begin(why)) {
        return refuse("alloc", why);
    }
    r.expected_stripes = run->expected_stripes();

    // Consumer first, under the caller's unchanged affinity: the owner keeps
    // a strong reference to the run until its consumer returns.
    potal_status status = POTAL_OK;
    if (!owner.start([run](const owner_runners & runners) { return run->consume(runners); }, why, status)) {
        r.status = status;
        return refuse("issuer", why); // nothing was issued
    }
    double act_quant_s = 0;
    try {
        struct unpin {
            stripe_owner & owner;
            ~unpin() { owner.unpin_producer(); }
        } scoped_producer{ owner };
        if (!owner.pin_producer(producer_on_one_cpu, why)) {
            run->abort("producer_affinity", why);
        } else {
            sink_binding           binding{ run.get(), &args };
            exsia::StripeReadySink sink{ &binding, &sink_binding::on_ready };
            args.exsia_stripe_ready_sink = &sink;
            run->emit_run(stripe_event_kind::quant_begin);
            const auto quant_start = std::chrono::steady_clock::now();
            const bool quantized = ggml::gemmini::quants::quantize_activation(src1, args);
            act_quant_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - quant_start).count();
            args.exsia_stripe_ready_sink = nullptr;
            run->emit_run(stripe_event_kind::quant_end);
            if (quantized) {
                (void) run->finish();
            } else {
                run->abort("act_quant", "quantize_activation failed");
            }
        }
    } catch (const std::exception & ex) {
        args.exsia_stripe_ready_sink = nullptr;
        run->abort("exception", ex.what());
    }
    owner.join();

    r             = run->report();
    r.counts.act_quant_s = act_quant_s;
    r.completions = run->poll_completed();
    if (r.ok) {
        const float * out = run->output();
        for (size_t m = 0; m < args.I; ++m) {
            float * dst = args.f_out + m * args.stride_f_out;
            for (size_t n = 0; n < args.J; ++n) {
                dst[n * args.col_stride_f_out] = out[m * args.J + n];
            }
        }
        run->emit_run(stripe_event_kind::run_commit);
    } else {
        run->emit_run(stripe_event_kind::run_failed, r.detail.c_str());
        if (r.fatal) {
            run->abandon();
        }
    }
    return r;
}

} // namespace ggml::potal
