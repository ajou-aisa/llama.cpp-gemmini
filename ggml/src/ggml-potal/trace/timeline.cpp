#include "trace/timeline.hpp"

#include <pthread.h>
#include <sched.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <new>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

extern char ** environ;

namespace potal::timeline {

namespace detail {
bool                        g_on   = false;
thread_local thread_ring *  t_ring = nullptr;
void                        init() noexcept;
}  // namespace detail

namespace {

constexpr uint32_t kDefaultEvents  = 65536;
constexpr uint32_t kDefaultAnchors = 4096;
constexpr uint32_t kMissing        = UINT32_MAX;

enum anchor_kind : uint8_t { a_register = 0, a_run_begin, a_run_end, a_explicit, a_thread_exit, a_flush };
const char * const kAnchorNames[] = { "register", "run_begin", "run_end", "explicit", "thread_exit", "flush" };

struct anchor_t {
    uint64_t mono_ns;
    uint64_t mono_unc_ns;  // width of the CLOCK_MONOTONIC bracket around t_local
    uint64_t t_local;
    int32_t  cpu;
    uint32_t run_id;
    uint8_t  kind;
};

struct run_t {
    uint32_t run_id;
    uint32_t layer_id;
    uint32_t begin_anchor;
    uint32_t end_anchor;
    uint8_t  op;
    uint8_t  ph;
};

struct thread_info {
    detail::thread_ring   ring;
    role                  r;
    int                   req_hart;
    int                   req_cpu;
    int                   cpu_start;
    int                   hart_start;
    long                  tid;
    bool                  affinity_ok;
    anchor_t *            anchors;
    uint32_t              anchor_cap;
    std::atomic<uint32_t> anchor_head;
    std::atomic<bool>     anchor_overflow;
    run_t *               runs;
    uint32_t              run_cap;
    std::atomic<uint32_t> run_head;
    std::atomic<bool>     run_overflow;
    std::atomic<bool>     exited;
    std::atomic<int>      cpu_exit;
};

struct state {
    std::once_flag        once;
    std::string           path;
    uint32_t              events_cap  = kDefaultEvents;
    uint32_t              anchors_cap = kDefaultAnchors;
    std::string           meta;
    std::vector<int>      cpu_to_hart;
    std::mutex            reg_mu;
    std::atomic<uint32_t> nthreads{ 0 };
    thread_info *         threads[max_threads] = {};
    std::atomic<uint32_t> next_run{ 1 };
    std::atomic<uint32_t> next_edge{ 1 };
    std::atomic<uint64_t> unregistered{ 0 };
    std::atomic<uint32_t> rejected_threads{ 0 };
    std::mutex            flush_mu;
    uint64_t              flushed_signature = 0;
};

state & st() {
    static state * s = new state();  // never destroyed: flush runs from atexit
    return *s;
}

thread_local thread_info * t_info = nullptr;

uint64_t mono_ns() noexcept {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<uint64_t>(ts.tv_sec) * 1000000000ull + static_cast<uint64_t>(ts.tv_nsec);
}

void take_anchor(thread_info * ti, uint8_t kind, uint32_t run_id) noexcept {
    const uint32_t i = ti->anchor_head.load(std::memory_order_relaxed);
    if (i >= ti->anchor_cap) {
        ti->anchor_overflow.store(true, std::memory_order_relaxed);
        return;
    }
    const uint64_t m0 = mono_ns();
    const uint64_t c  = detail::now_local();
    const uint64_t m1 = mono_ns();
    anchor_t &     a  = ti->anchors[i];
    a.mono_ns         = m0 + (m1 - m0) / 2;
    a.mono_unc_ns     = m1 - m0;
    a.t_local         = c;
    a.cpu             = sched_getcpu();
    a.run_id          = run_id;
    a.kind            = kind;
    ti->anchor_head.store(i + 1, std::memory_order_relaxed);
}

struct exit_guard {
    thread_info * ti = nullptr;

    ~exit_guard() {
        if (ti != nullptr) {
            take_anchor(ti, a_thread_exit, 0);
            ti->cpu_exit.store(sched_getcpu(), std::memory_order_relaxed);
            ti->exited.store(true, std::memory_order_release);
        }
    }
};

thread_local exit_guard t_guard;

uint32_t parse_count(const char * name, uint32_t def) {
    const char * v = std::getenv(name);
    if (v == nullptr || *v == '\0') {
        return def;
    }
    char *              end = nullptr;
    errno                   = 0;
    const unsigned long n   = std::strtoul(v, &end, 10);
    if (errno != 0 || end == v || *end != '\0' || n == 0 || n > (1ul << 26)) {
        std::fprintf(stderr, "potal timeline: invalid %s=%s, using %u\n", name, v, def);
        return def;
    }
    return static_cast<uint32_t>(n);
}

std::vector<int> read_cpu_to_hart() {
    std::vector<int> map;
    FILE *           f = std::fopen("/proc/cpuinfo", "r");
    if (f == nullptr) {
        return map;
    }
    char line[256];
    int  processor = -1;
    while (std::fgets(line, sizeof(line), f) != nullptr) {
        int v;
        if (std::sscanf(line, "processor : %d", &v) == 1) {
            processor = v;
        } else if (std::sscanf(line, "hart : %d", &v) == 1 && processor >= 0) {
            if (map.size() <= static_cast<size_t>(processor)) {
                map.resize(static_cast<size_t>(processor) + 1, -1);
            }
            map[static_cast<size_t>(processor)] = v;
        }
    }
    std::fclose(f);
    return map;
}

int hart_of(int cpu) {
    const std::vector<int> & m = st().cpu_to_hart;
    return cpu >= 0 && static_cast<size_t>(cpu) < m.size() ? m[static_cast<size_t>(cpu)] : -1;
}

// Field 39 ("processor") of /proc/self/task/<tid>/stat: the CPU a live thread last ran on.
int task_last_cpu(long tid) {
    char path[64];
    std::snprintf(path, sizeof(path), "/proc/self/task/%ld/stat", tid);
    FILE * f = std::fopen(path, "r");
    if (f == nullptr) {
        return -1;
    }
    char   buf[1024];
    size_t n = std::fread(buf, 1, sizeof(buf) - 1, f);
    std::fclose(f);
    buf[n]          = '\0';
    const char * p  = std::strrchr(buf, ')');
    if (p == nullptr) {
        return -1;
    }
    int field = 2;
    for (++p; *p != '\0'; ++p) {
        if (*p == ' ') {
            ++field;
            if (field == 39) {
                return std::atoi(p + 1);
            }
        }
    }
    return -1;
}

std::string encode(const std::string & v) {
    std::string out;
    for (unsigned char c : v) {
        if (c == ',' || c == ';' || c == '%' || c == '=' || c < 0x20 || c == '"') {
            char b[4];
            std::snprintf(b, sizeof(b), "%%%02X", c);
            out += b;
        } else {
            out += static_cast<char>(c);
        }
    }
    return out;
}

std::string env_or(const char * name, const char * def) {
    const char * v = std::getenv(name);
    return v != nullptr && *v != '\0' ? std::string(v) : std::string(def);
}

std::string build_meta(const state & s) {
#ifdef POTAL_TIMELINE_BUILD_ID
    const char * build_default = POTAL_TIMELINE_BUILD_ID;
#else
    const char * build_default = "unknown";
#endif
    char exe[512];
    const ssize_t n = readlink("/proc/self/exe", exe, sizeof(exe) - 1);
    exe[n > 0 ? n : 0] = '\0';
    std::vector<std::string> env;
    for (char ** e = environ; e != nullptr && *e != nullptr; ++e) {
        const char * kv = *e;
        if (std::strncmp(kv, "GGML_POTAL_", 11) == 0 || std::strncmp(kv, "POTAL_", 6) == 0 ||
            std::strncmp(kv, "GGML_GEMMINI_", 13) == 0 || std::strncmp(kv, "GEMMINI_", 8) == 0) {
            const char * eq = std::strchr(kv, '=');
            if (eq != nullptr) {
                env.push_back(encode(std::string(kv, eq)) + "=" + encode(eq + 1));
            }
        }
    }
    std::sort(env.begin(), env.end());
    std::string joined;
    for (const std::string & kv : env) {
        joined += (joined.empty() ? "" : ";") + kv;
    }
    std::string m = "META,format=potal_timeline_v1";
    m += ",model_id=" + encode(env_or("POTAL_TIMELINE_MODEL_ID", "unknown"));
    m += ",build_id=" + encode(env_or("POTAL_TIMELINE_BUILD_ID", build_default));
    m += ",image_id=" + encode(env_or("POTAL_TIMELINE_IMAGE_ID", "unknown"));
    m += ",exe=" + encode(exe);
    m += ",pid=" + std::to_string(getpid());
#if defined(__riscv)
    m += ",clock_local=rdcycle";
#else
    m += ",clock_local=steady_ns";
#endif
    m += ",clock_common=CLOCK_MONOTONIC_ns";
    m += ",ring_events=" + std::to_string(s.events_cap);
    m += ",anchor_capacity=" + std::to_string(s.anchors_cap);
    m += ",env=" + joined;
    return m;
}

uint64_t signature(const state & s) {
    uint64_t       sig = 0;
    const uint32_t n   = s.nthreads.load(std::memory_order_acquire);
    for (uint32_t i = 0; i < n; ++i) {
        const thread_info * ti = s.threads[i];
        sig += ti->ring.head.load(std::memory_order_relaxed) + ti->ring.dropped.load(std::memory_order_relaxed) +
               ti->anchor_head.load(std::memory_order_relaxed) + ti->run_head.load(std::memory_order_relaxed) + 1;
    }
    return sig + s.unregistered.load(std::memory_order_relaxed);
}

void flush_at_exit() {
    state & s = st();
    bool    dirty;
    {
        std::lock_guard<std::mutex> lk(s.flush_mu);
        dirty = signature(s) != s.flushed_signature;
    }
    if (dirty) {
        (void) flush();
    }
}

void init_once() {
    state &      s = st();
    const char * p = std::getenv("POTAL_TIMELINE");
    if (p == nullptr || *p == '\0') {
        return;
    }
    s.path        = p;
    s.events_cap  = parse_count("POTAL_TIMELINE_EVENTS", kDefaultEvents);
    s.anchors_cap = parse_count("POTAL_TIMELINE_ANCHORS", kDefaultAnchors);
    s.cpu_to_hart = read_cpu_to_hart();
    s.meta        = build_meta(s);
    if (std::atexit(flush_at_exit) != 0) {
        std::fprintf(stderr, "potal timeline: atexit registration failed; call flush() explicitly\n");
    }
    detail::g_on = true;
}

struct load_time_init {
    load_time_init() { detail::init(); }
} g_load_time_init;

template <typename T> T * alloc_zeroed(size_t n) {
    void * p = nullptr;
    if (posix_memalign(&p, 64, n * sizeof(T)) != 0) {
        return nullptr;
    }
    std::memset(p, 0, n * sizeof(T));  // touch every page now, not on the hot path
    return static_cast<T *>(p);
}

const char * stage_name(uint16_t id) {
    switch (static_cast<stage>(id)) {
        case stage::node_total: return "node.total";
        case stage::exsia_prepare: return "exsia.prepare";
        case stage::exsia_local: return "exsia.local";
        case stage::exsia_local_seal: return "exsia.local_seal";
        case stage::exsia_mask_assembly: return "exsia.mask_assembly";
        case stage::exsia_exponent_reduction: return "exsia.exponent_reduction";
        case stage::exsia_folding: return "exsia.folding";
        case stage::exsia_packet_seal: return "exsia.packet_seal";
        case stage::stripe_submit: return "stripe.submit";
        case stage::issuer_ready_wait: return "issuer.ready_wait";
        case stage::stripe_residual_prepare: return "stripe.residual_prepare";
        case stage::acc_scale_write: return "acc.scale_write";
        case stage::acc_index_scalar: return "acc.index_scalar";
        case stage::acc_queue_install: return "acc.queue_install";
        case stage::acc_main_issue: return "acc.main_issue";
        case stage::acc_residual_issue: return "acc.residual_issue";
        case stage::acc_fence: return "acc.fence";
        case stage::acc_readback: return "acc.readback";
        case stage::acc_scale_release: return "acc.scale_release";
        case stage::acc_prefetch_issue: return "acc.prefetch_issue";
        case stage::final_handoff: return "final.handoff";
        case stage::final_wait: return "final.wait";
        case stage::final_reconstruct_int64: return "final.reconstruct_int64";
        case stage::final_dequant_merge: return "final.dequant_merge";
        case stage::final_slot_release: return "final.slot_release";
        case stage::node_commit: return "node.commit";
    }
    return nullptr;
}

const char * role_name(role r) {
    switch (r) {
        case role::exsia_local0: return "exsia_local0";
        case role::exsia_local1: return "exsia_local1";
        case role::exsia_fold: return "exsia_fold";
        case role::issuer: return "issuer";
        case role::finalizer: return "finalizer";
        case role::caller: return "caller";
    }
    return "unknown";
}

}  // namespace

void detail::init() noexcept {
    std::call_once(st().once, init_once);
}

void detail::note_unregistered() noexcept {
    st().unregistered.fetch_add(1, std::memory_order_relaxed);
}

bool enabled() noexcept {
    detail::init();
    return detail::g_on;
}

uint16_t register_thread(role r, int hart_id, int cpu_id) noexcept {
    if (!detail::g_on) {
        return no_slot;
    }
    if (t_info != nullptr) {
        return t_info->ring.slot;
    }
    state &       s  = st();
    thread_info * ti = new (std::nothrow) thread_info();
    const uint32_t cap = r == role::issuer ? parse_count("POTAL_TIMELINE_ISSUER_EVENTS", s.events_cap) : s.events_cap;
    if (ti != nullptr) {
        ti->ring.events = alloc_zeroed<record_t>(cap);
        ti->anchors     = alloc_zeroed<anchor_t>(s.anchors_cap);
        ti->runs        = alloc_zeroed<run_t>(s.anchors_cap);
    }
    if (ti == nullptr || ti->ring.events == nullptr || ti->anchors == nullptr || ti->runs == nullptr) {
        std::fprintf(stderr, "potal timeline: cannot allocate the ring of a %s thread; it records nothing\n",
                     role_name(r));
        s.rejected_threads.fetch_add(1, std::memory_order_relaxed);
        return no_slot;  // a partial allocation is leaked deliberately: diagnostics only, once per thread
    }
    ti->ring.capacity = cap;
    ti->ring.direct_all = cpu_id < 0;
    ti->anchor_cap    = s.anchors_cap;
    ti->run_cap       = s.anchors_cap;
    ti->r             = r;
    ti->req_hart      = hart_id;
    ti->req_cpu       = cpu_id;
    ti->tid           = static_cast<long>(syscall(SYS_gettid));
    ti->cpu_start     = sched_getcpu();
    ti->hart_start    = hart_of(ti->cpu_start);
    ti->cpu_exit.store(-1, std::memory_order_relaxed);
    ti->affinity_ok = false;
    if (cpu_id >= 0) {
        cpu_set_t got;
        CPU_ZERO(&got);
        ti->affinity_ok = pthread_getaffinity_np(pthread_self(), sizeof(got), &got) == 0 && CPU_COUNT(&got) == 1 &&
                          CPU_ISSET(cpu_id, &got) && ti->cpu_start == cpu_id &&
                          (hart_id < 0 || ti->hart_start == hart_id);
    }
    {
        std::lock_guard<std::mutex> lk(s.reg_mu);
        const uint32_t              n = s.nthreads.load(std::memory_order_relaxed);
        if (n >= max_threads) {
            s.rejected_threads.fetch_add(1, std::memory_order_relaxed);
            std::fprintf(stderr, "potal timeline: more than %u threads; a %s thread records nothing\n", max_threads,
                         role_name(r));
            return no_slot;
        }
        ti->ring.slot = static_cast<uint16_t>(n);
        s.threads[n]  = ti;
        s.nthreads.store(n + 1, std::memory_order_release);
    }
    t_info        = ti;
    t_guard.ti    = ti;
    detail::t_ring = &ti->ring;
    take_anchor(ti, a_register, 0);
    return ti->ring.slot;
}

uint32_t begin_run(op_type op, phase ph, uint32_t layer_id) noexcept {
    if (!detail::g_on) {
        return 0;
    }
    thread_info * ti = t_info;
    if (ti == nullptr) {
        detail::note_unregistered();
        return 0;
    }
    const uint32_t id = st().next_run.fetch_add(1, std::memory_order_relaxed);
    const uint32_t ai = ti->anchor_head.load(std::memory_order_relaxed);
    take_anchor(ti, a_run_begin, id);
    const uint32_t ri = ti->run_head.load(std::memory_order_relaxed);
    if (ri < ti->run_cap) {
        ti->runs[ri] = run_t{ id, layer_id, ai < ti->anchor_cap ? ai : kMissing, kMissing, static_cast<uint8_t>(op),
                              static_cast<uint8_t>(ph) };
        ti->run_head.store(ri + 1, std::memory_order_relaxed);
    } else {
        ti->run_overflow.store(true, std::memory_order_relaxed);
    }
    ti->ring.cur_op    = static_cast<uint8_t>(op);
    ti->ring.cur_phase = static_cast<uint8_t>(ph);
    return id;
}

void end_run(uint32_t run_id) noexcept {
    if (!detail::g_on || run_id == 0) {
        return;
    }
    thread_info * ti = t_info;
    if (ti == nullptr) {
        detail::note_unregistered();
        return;
    }
    const uint32_t ai = ti->anchor_head.load(std::memory_order_relaxed);
    take_anchor(ti, a_run_end, run_id);
    const uint32_t end_anchor = ai < ti->anchor_cap ? ai : kMissing;
    const uint32_t n          = ti->run_head.load(std::memory_order_relaxed);
    for (uint32_t i = n; i-- > 0;) {
        if (ti->runs[i].run_id == run_id) {
            ti->runs[i].end_anchor = end_anchor;
            ti->ring.cur_op        = 0;
            ti->ring.cur_phase     = 0;
            return;
        }
    }
    if (n < ti->run_cap) {  // ended on a thread that did not begin it
        ti->runs[n] = run_t{ run_id, kMissing, kMissing, end_anchor, 0, 0 };
        ti->run_head.store(n + 1, std::memory_order_relaxed);
    } else {
        ti->run_overflow.store(true, std::memory_order_relaxed);
    }
}

uint32_t next_edge_id() noexcept {
    if (!detail::g_on) {
        return 0;
    }
    return st().next_edge.fetch_add(1, std::memory_order_relaxed);
}

void anchor() noexcept {
    if (!detail::g_on) {
        return;
    }
    if (t_info == nullptr) {
        detail::note_unregistered();
        return;
    }
    take_anchor(t_info, a_explicit, 0);
}

stats_t stats() noexcept {
    stats_t r{};
    if (!detail::g_on) {
        return r;
    }
    const state & s = st();
    r.threads       = s.nthreads.load(std::memory_order_acquire);
    for (uint32_t i = 0; i < r.threads; ++i) {
        const thread_info * ti = s.threads[i];
        r.events += ti->ring.head.load(std::memory_order_relaxed);
        r.dropped += ti->ring.dropped.load(std::memory_order_relaxed);
        r.overflowed_threads += ti->ring.overflow.load(std::memory_order_relaxed) ? 1 : 0;
    }
    r.unregistered = s.unregistered.load(std::memory_order_relaxed);
    return r;
}

bool flush() noexcept {
    if (!detail::g_on) {
        return false;
    }
    state &                     s = st();
    std::lock_guard<std::mutex> lk(s.flush_mu);
    if (t_info != nullptr) {
        take_anchor(t_info, a_flush, 0);
    }
    const uint32_t                                         n = s.nthreads.load(std::memory_order_acquire);
    std::unordered_map<uint32_t, std::pair<uint8_t, uint8_t>> run_kind;
    for (uint32_t i = 0; i < n; ++i) {
        const thread_info * ti = s.threads[i];
        const uint32_t      rn = ti->run_head.load(std::memory_order_relaxed);
        for (uint32_t j = 0; j < rn; ++j) {
            if (ti->runs[j].layer_id != kMissing || ti->runs[j].begin_anchor != kMissing) {
                run_kind[ti->runs[j].run_id] = { ti->runs[j].op, ti->runs[j].ph };
            }
        }
    }

    const std::string tmp = s.path + ".tmp";
    FILE *            f   = std::fopen(tmp.c_str(), "w");
    if (f == nullptr) {
        std::fprintf(stderr, "potal timeline: cannot write %s: %s\n", tmp.c_str(), std::strerror(errno));
        return false;
    }
    std::fprintf(f, "# potal_timeline v1\n%s,unregistered_records=%" PRIu64 ",rejected_threads=%u\n", s.meta.c_str(),
                 s.unregistered.load(std::memory_order_relaxed), s.rejected_threads.load(std::memory_order_relaxed));
    std::fprintf(f, "#STAGE,id,name\n");
    for (uint16_t id = 0; id < 64; ++id) {
        if (const char * name = stage_name(id)) {
            std::fprintf(f, "S,%u,%s\n", id, name);
        }
    }
    std::fprintf(f,
                 "#THREAD,slot,role,req_hart,req_cpu,hart_id,linux_cpu_id,tid,pin_verified_start,pin_verified_end,"
                 "cpu_end,cpu_unverified,events,dropped,overflow,anchors,anchor_overflow,runs,run_overflow\n");
    uint64_t total_events = 0;
    for (uint32_t i = 0; i < n; ++i) {
        const thread_info * ti = s.threads[i];
        int                 cpu_end;
        if (ti->exited.load(std::memory_order_acquire)) {
            cpu_end = ti->cpu_exit.load(std::memory_order_relaxed);
        } else if (ti == t_info) {
            cpu_end = sched_getcpu();
        } else {
            cpu_end = task_last_cpu(ti->tid);
        }
        const uint32_t an       = ti->anchor_head.load(std::memory_order_relaxed);
        bool           migrated = cpu_end < 0 || cpu_end != ti->cpu_start;
        for (uint32_t j = 0; j < an; ++j) {
            migrated = migrated || ti->anchors[j].cpu != ti->cpu_start;
        }
        const bool     pin_end = ti->affinity_ok && !migrated && cpu_end == ti->req_cpu;
        const uint32_t ev      = ti->ring.head.load(std::memory_order_relaxed);
        total_events += ev;
        std::fprintf(f, "T,%u,%s,%d,%d,%d,%d,%ld,%d,%d,%d,%d,%u,%u,%d,%u,%d,%u,%d\n", ti->ring.slot, role_name(ti->r),
                     ti->req_hart, ti->req_cpu, ti->hart_start, ti->cpu_start, ti->tid, ti->affinity_ok ? 1 : 0,
                     pin_end ? 1 : 0, cpu_end, migrated ? 1 : 0, ev, ti->ring.dropped.load(std::memory_order_relaxed),
                     ti->ring.overflow.load(std::memory_order_relaxed) ? 1 : 0, an,
                     ti->anchor_overflow.load(std::memory_order_relaxed) ? 1 : 0,
                     ti->run_head.load(std::memory_order_relaxed),
                     ti->run_overflow.load(std::memory_order_relaxed) ? 1 : 0);
    }
    std::fprintf(f, "#ANCHOR,slot,idx,kind,run_id,mono_ns,mono_unc_ns,t_local,cpu\n");
    for (uint32_t i = 0; i < n; ++i) {
        const thread_info * ti = s.threads[i];
        const uint32_t      an = ti->anchor_head.load(std::memory_order_relaxed);
        for (uint32_t j = 0; j < an; ++j) {
            const anchor_t & a = ti->anchors[j];
            std::fprintf(f, "A,%u,%u,%s,%u,%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%d\n", ti->ring.slot, j,
                         kAnchorNames[a.kind], a.run_id, a.mono_ns, a.mono_unc_ns, a.t_local, a.cpu);
        }
    }
    std::fprintf(f, "#RUN,run_id,slot,op_type,phase,layer_id,begin_anchor,end_anchor\n");
    for (uint32_t i = 0; i < n; ++i) {
        const thread_info * ti = s.threads[i];
        const uint32_t      rn = ti->run_head.load(std::memory_order_relaxed);
        for (uint32_t j = 0; j < rn; ++j) {
            const run_t & r = ti->runs[j];
            std::fprintf(f, "R,%u,%u,%u,%u,%u,%u,%u\n", r.run_id, ti->ring.slot, r.op, r.ph, r.layer_id,
                         r.begin_anchor, r.end_anchor);
        }
    }
    std::fprintf(f,
                 "#EVENT,t_local,seq,slot,stage,event_type,status,op_type,phase,run_id,layer_id,stripe_id,tile_id,"
                 "buffer_id,edge_id,work,mono_direct\n");
    for (uint32_t i = 0; i < n; ++i) {
        const thread_info * ti = s.threads[i];
        const uint32_t      ev = ti->ring.head.load(std::memory_order_relaxed);
        for (uint32_t j = 0; j < ev; ++j) {
            const record_t & e  = ti->ring.events[j];
            uint8_t          op = e.op_type;
            uint8_t          ph = e.phase;
            if (op == 0 && ph == 0) {
                const auto it = run_kind.find(e.run_id);
                if (it != run_kind.end()) {
                    op = it->second.first;
                    ph = it->second.second;
                }
            }
            std::fprintf(f, "E,%" PRIu64 ",%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%" PRIu64 ",%" PRIu64 "\n", e.t_local,
                         e.seq, e.thread_slot, e.stage, e.event_type, e.status, op, ph, e.run_id, e.layer_id,
                         e.stripe_id, e.tile_id, e.buffer_id, e.edge_id, e.work, e.reserved);
        }
    }
    std::fprintf(f, "#END,events=%" PRIu64 "\n", total_events);
    const bool write_error = std::ferror(f) != 0;
    if (std::fclose(f) != 0 || write_error || std::rename(tmp.c_str(), s.path.c_str()) != 0) {
        std::fprintf(stderr, "potal timeline: writing %s failed: %s\n", s.path.c_str(), std::strerror(errno));
        return false;
    }
    s.flushed_signature = signature(s);
    return true;
}

}  // namespace potal::timeline
