#pragma once

// The parent target compiles this facility and adds its six GNU --wrap options
// only when diagnostics are enabled. An OFF build has no declarations or code.
#ifdef POTAL_PLANNING_DIAGNOSTICS

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>

namespace ggml::potal::planning_profile {

enum class Phase { route, executor, count };
enum class Component { validation, copy, allocation, deallocation, construction, other, instrumentation, count };
enum class Site {
    route_overhead,
    fragment_build,
    fragment_input,
    fragment_rows,
    tile_build,
    tile_input,
    tile_frozen,
    execution_build,
    execution_input,
    execution_validate,
    executor_request,
    supplied_match,
    builder_tile_copy,
    builder_rows_copy,
    report_plan_copy,
    report_tile_copy,
    payload_inventory,
    allocate,
    deallocate,
    count
};

struct ComponentStats {
    uint64_t calls = 0;
    uint64_t exclusive_ns = 0;
    uint64_t bytes = 0;
    uint64_t objects = 0;
};

struct SiteStats {
    uint64_t calls = 0;
    uint64_t exclusive_ns = 0;
    uint64_t inclusive_ns = 0;
    uint64_t bytes = 0;
    uint64_t objects = 0;
};

struct PhaseStats {
    uint64_t roots = 0;
    uint64_t total_ns = 0;
    uint64_t rows_sum = 0;
    uint64_t n = 0;
    uint64_t k = 0;
    // Mixed shapes are reported with n=0 and k=0, rather than a last-writer shape.
    bool mixed_shape = false;
    std::array<ComponentStats, static_cast<size_t>(Component::count)> components{};
    std::array<SiteStats, static_cast<size_t>(Site::count)> sites{};
};

struct Snapshot {
    std::array<PhaseStats, static_cast<size_t>(Phase::count)> phases{};
};

class Scope;

// Roots and their scopes are thread-affine and nonmovable. Only the outermost
// enabled Root owns accounting. A nested Root neither replaces nor finishes it.
// Separate route and executor roots must not overlap on the same thread. The
// normal supplied-plan path creates one of each per paired stripe. Aggregation
// spans the graph between reset() before its timer and dump() after its timer.
class Root {
public:
    Root(Phase phase, size_t rows, size_t n, size_t k);
    ~Root();
    void finish();

    Root(const Root &) = delete;
    Root & operator=(const Root &) = delete;

private:
    friend class Scope;
    using Clock = std::chrono::steady_clock;
    void charge(Clock::time_point now);
    void bookkeeping(Clock::time_point now, Site site);

    Phase phase_;
    PhaseStats stats_{};
    Clock::time_point start_{};
    Clock::time_point last_{};
    Scope * top_ = nullptr;
    uint64_t generation_ = 0;
    bool active_ = false;
};

// Scopes normally finish in stack order. Explicit finish also supports removing
// an enclosing scope first. Finishing a scope after its Root ends is harmless,
// even if another Root subsequently reuses the same stack address.
class Scope {
public:
    Scope(Site site, Component component);
    ~Scope();
    void finish();
    // Inventory may be attached after finish() while the same Root is active.
    // Only bytes/objects change on the destination scope. Time is charged to
    // the current scope, so payload_inventory/instrumentation can exclude the
    // traversal from the finished copy's timing without reviving that timer.
    void add_bytes(uint64_t bytes, uint64_t objects = 1);

    Scope(const Scope &) = delete;
    Scope & operator=(const Scope &) = delete;

private:
    friend class Root;
    bool attached() const;

    Root * root_ = nullptr;
    Scope * parent_ = nullptr;
    Root::Clock::time_point start_{};
    Site site_;
    Component component_;
    uint64_t generation_ = 0;
    bool finished_ = false;
};

// reset() requires idle graph workers and no active roots. snapshot() copies
// completed roots only. All storage is fixed-size; none of these APIs uses new.
void reset();
Snapshot snapshot();

// Call after the graph timer, not inside measured planning. Emission requires
// POTAL_PLANNING_PROFILE=1 at the time of this call. Names must be single tokens.
void dump(FILE * stream, const char * case_name, const char * route_name);

// Exclusive component and site sums each equal total_ns. Unscoped time belongs
// to other/route_overhead. Each root counts one such implicit scope, including
// its inclusive total. Scope inclusive times may overlap and must not be summed
// into exclusive totals. Bookkeeping intervals are charged to instrumentation
// at the affected site, with one instrumentation call per interval; nothing is
// calibrated away or silently subtracted. Runtime gating, initial zeroing, and
// global merging lie outside the root's clock interval.
//
// Link with --wrap=_Znwm, --wrap=_Znam, --wrap=_ZdlPv, --wrap=_ZdaPv,
// --wrap=_ZdlPvm, and --wrap=_ZdaPvm. Coverage is limited to references resolved
// in this target. It excludes malloc, aligned overloads, and external-DSO
// allocation. No allocator is replaced and no pointer ownership is tracked.
// Allocation calls/bytes count requests, including throwing requests. Unsized
// deletes count objects but have no known byte size; sized deletes count the
// size actually supplied. Callers must not treat this as all heap traffic.

} // namespace ggml::potal::planning_profile

#endif // POTAL_PLANNING_DIAGNOSTICS
