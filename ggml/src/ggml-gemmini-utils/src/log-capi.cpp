#include "../include/gemmini/log.hpp"
#include "../include/gemmini/log.h"

#include <cstdarg>
#include <cstdint>
#include <limits>
#include <mutex>
#include <new>

namespace {
std::mutex & hardware_counter_mutex()
{
    static std::mutex mutex;
    return mutex;
}

thread_local bool hardware_counter_c_lease_acquired = false;

void report_cycle_boundary_failure() noexcept
{
    ggml::gemmini::log::cycle.report_failure("serialization");
}
} // namespace

namespace ggml::gemmini::log
{
HardwareCounterLease::HardwareCounterLease()
{
    hardware_counter_mutex().lock();
}

HardwareCounterLease::~HardwareCounterLease()
{
    hardware_counter_mutex().unlock();
}
} // namespace ggml::gemmini::log

extern "C"
{
    gemmini_log_target gemmini_log_file(const char *path) noexcept
    {
        return gemmini_log_target{path};
    }

    int gemmini_log_truncate_file(const char *path) noexcept
    {
        try { return ggml::gemmini::log::truncate_file(path) ? 1 : 0; }
        catch (...) { return 0; }
    }

    int gemmini_log_debug_set_output_path(const char *path) noexcept
    {
        try { return ggml::gemmini::log::debug.set_output_path(path) ? 1 : 0; }
        catch (...) { return 0; }
    }

    void gemmini_log_debug_set_output(FILE *out) noexcept
    {
        try { ggml::gemmini::log::debug.set_output(out); } catch (...) {}
    }

    int gemmini_log_cycle_set_output_path(const char *path) noexcept
    {
        try { return ggml::gemmini::log::cycle.set_output_path(path) ? 1 : 0; }
        catch (...) { report_cycle_boundary_failure(); return 0; }
    }

    void gemmini_log_cycle_set_output(FILE *out) noexcept
    {
        try { ggml::gemmini::log::cycle.set_output(out); } catch (...) { report_cycle_boundary_failure(); }
    }

    void gemmini_log_debug(const char *fmt, ...) noexcept
    {
        va_list ap;
        va_start(ap, fmt);
        try { ggml::gemmini::log::debug.v(fmt, ap); } catch (...) {}
        va_end(ap);
    }

    void gemmini_log_debug_layer(const char *layer, const char *fmt, ...) noexcept
    {
        va_list ap;
        va_start(ap, fmt);
        try { ggml::gemmini::log::debug.v_layer(layer, fmt, ap); } catch (...) {}
        va_end(ap);
    }

    void gemmini_log_debug_loc(const char *file, int line, const char *func, const char *fmt, ...) noexcept
    {
        va_list ap;
        va_start(ap, fmt);
        try { ggml::gemmini::log::debug.v_loc(file, line, func, fmt, ap); } catch (...) {}
        va_end(ap);
    }

    void gemmini_log_debug_to(gemmini_log_target target, const char *fmt, ...) noexcept
    {
        va_list ap;
        va_start(ap, fmt);
        try { ggml::gemmini::log::debug.v_target({target.path}, fmt, ap); } catch (...) {}
        va_end(ap);
    }

    void gemmini_log_debug_to_layer(gemmini_log_target target, const char *layer, const char *fmt, ...) noexcept
    {
        va_list ap;
        va_start(ap, fmt);
        try { ggml::gemmini::log::debug.v_target_layer({target.path}, layer, fmt, ap); } catch (...) {}
        va_end(ap);
    }

    void gemmini_log_debug_to_loc(gemmini_log_target target, const char *file, int line, const char *func,
                                  const char *fmt, ...) noexcept
    {
        va_list ap;
        va_start(ap, fmt);
        try { ggml::gemmini::log::debug.v_target_loc({target.path}, file, line, func, fmt, ap); } catch (...) {}
        va_end(ap);
    }

    void gemmini_hardware_counter_lease_acquire(void) noexcept
    {
        hardware_counter_c_lease_acquired = false;
        try
        {
            hardware_counter_mutex().lock();
            hardware_counter_c_lease_acquired = true;
        }
        catch (...) {}
    }

    void gemmini_hardware_counter_lease_release(void) noexcept
    {
        if (!hardware_counter_c_lease_acquired) return;
        hardware_counter_c_lease_acquired = false;
        try { hardware_counter_mutex().unlock(); } catch (...) {}
    }

    void gemmini_log_ws_cycle(uint64_t containing_interval_cycles,
                              uint32_t load_occupancy_cycles,
                              uint32_t execute_occupancy_cycles,
                              uint32_t store_occupancy_cycles,
                              uint32_t loop_occupancy_cycles,
                              uint64_t dim_I, uint64_t dim_J, uint64_t dim_K,
                              uint64_t tile_I, uint64_t tile_J, uint64_t tile_K,
                              uint64_t I0, uint64_t J0, uint64_t K0,
                              uint64_t a_reuse, uint64_t b_reuse) noexcept
    {
#if LOG_CYCLE
        try
        {
            if (ggml::gemmini::log::detail::consume_fault(ggml::gemmini::log::testing::LogFault::allocation))
                throw std::bad_alloc();
            auto checked_product = [](uint64_t lhs, uint64_t rhs) {
                return lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs
                    ? std::numeric_limits<uint64_t>::max() : lhs * rhs;
            };
            (void) a_reuse;
            (void) b_reuse;
            const ggml::gemmini::log::WsCycleRecord record{
                containing_interval_cycles, load_occupancy_cycles, execute_occupancy_cycles,
                store_occupancy_cycles, loop_occupancy_cycles, dim_I, dim_J, dim_K,
                tile_I, tile_J, tile_K, I0, J0, K0,
                checked_product(checked_product(I0, J0), K0)};
            ggml::gemmini::log::cycle.write_json(ggml::gemmini::log::serialize_ws_cycle_record(record));
        }
        catch (...) { report_cycle_boundary_failure(); }
#else
        (void) containing_interval_cycles; (void) load_occupancy_cycles; (void) execute_occupancy_cycles;
        (void) store_occupancy_cycles; (void) loop_occupancy_cycles;
        (void) dim_I; (void) dim_J; (void) dim_K; (void) tile_I; (void) tile_J; (void) tile_K;
        (void) I0; (void) J0; (void) K0; (void) a_reuse; (void) b_reuse;
#endif
    }

    void gemmini_log_cycle_record(const gemmini_cycle_record *record) noexcept
    {
        if (!record) return;
        try
        {
            ggml::gemmini::log::cycle.write({record->layer, record->op, record->start, record->end,
                                             record->file, record->line, record->func});
        }
        catch (...) { report_cycle_boundary_failure(); }
    }

    void gemmini_log_cycle_record_v2(const gemmini_cycle_record_v2 *record) noexcept
    {
        if (!record) return;
        try
        {
            const gemmini_cycle_record & interval = record->interval;
            ggml::gemmini::log::cycle.write({
                interval.layer, interval.op, interval.start, interval.end,
                interval.file, interval.line, interval.func, nullptr, nullptr,
                record->identity_mask, record->run_id, record->stripe_id,
                record->slot, record->node_id, record->worker_id});
        }
        catch (...) { report_cycle_boundary_failure(); }
    }

    void gemmini_log_cycle(const char *layer, const char *op, uint64_t start, uint64_t end) noexcept
    {
        const gemmini_cycle_record record{layer, op, start, end, nullptr, 0, nullptr};
        gemmini_log_cycle_record(&record);
    }

    void gemmini_log_cycle_loc(const char *file, int line, const char *func,
                               const char *layer, const char *op, uint64_t start, uint64_t end) noexcept
    {
        const gemmini_cycle_record record{layer, op, start, end, file, line, func};
        gemmini_log_cycle_record(&record);
    }

    void gemmini_log_cycle_to(gemmini_log_target target, const char *layer, const char *op,
                              uint64_t start, uint64_t end) noexcept
    {
        try { ggml::gemmini::log::cycle({target.path}, layer, op, start, end); }
        catch (...) { report_cycle_boundary_failure(); }
    }

    void gemmini_log_cycle_to_loc(gemmini_log_target target, const char *file, int line, const char *func,
                                  const char *layer, const char *op, uint64_t start, uint64_t end) noexcept
    {
        try { ggml::gemmini::log::cycle({target.path}, file, line, func, layer, op, start, end); }
        catch (...) { report_cycle_boundary_failure(); }
    }
} // extern "C"
