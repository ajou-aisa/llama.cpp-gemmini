// potal-timeline-test: the opt-in timeline trace (trace/timeline.hpp).
//
// POTAL_TIMELINE is read once at load time, so each mode runs in its own child
// process (this executable re-spawned with --child=<mode> and its own env):
//   disabled  POTAL_TIMELINE unset: nothing recorded, no slot, no flush
//   overflow  POTAL_TIMELINE_EVENTS=64: drops past capacity, overflow flag
//   main      4 recording threads (caller, fold, issuer, finalizer) with
//             handoff edges; flushed CSV content checked
// The main trace goes to $POTAL_TIMELINE when set (for timeline_parse.py),
// else to a fresh temporary directory.
#include "trace/timeline.hpp"

#include <pthread.h>
#include <sched.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

extern char ** environ;

namespace tl = potal::timeline;

namespace {

int g_failures = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                        \
        }                                                                        \
    } while (0)

constexpr uint32_t kStripes = 8;
constexpr auto     kWait    = std::chrono::seconds(10);

struct handoff_queue {
    std::mutex                                  mu;
    std::condition_variable                     cv;
    std::deque<std::pair<uint32_t, uint32_t>>   items;  // (stripe, edge)

    void push(uint32_t stripe, uint32_t edge) {
        {
            std::lock_guard<std::mutex> lk(mu);
            items.emplace_back(stripe, edge);
        }
        cv.notify_one();
    }

    bool pop(std::pair<uint32_t, uint32_t> & out) {
        std::unique_lock<std::mutex> lk(mu);
        if (!cv.wait_for(lk, kWait, [&] { return !items.empty(); })) {
            return false;
        }
        out = items.front();
        items.pop_front();
        return true;
    }
};

std::vector<std::vector<std::string>> rows_of(const std::string & path, const std::string & tag) {
    std::vector<std::vector<std::string>> rows;
    std::ifstream                         in(path);
    std::string                           line;
    while (std::getline(in, line)) {
        std::vector<std::string> cols;
        std::stringstream        ss(line);
        std::string              c;
        while (std::getline(ss, c, ',')) {
            cols.push_back(c);
        }
        if (!cols.empty() && cols[0] == tag) {
            rows.push_back(cols);
        }
    }
    return rows;
}

std::string slurp(const std::string & path) {
    std::ifstream      in(path);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

using tl::event;
using tl::stage;

void fold_thread(uint32_t run, handoff_queue & to_issuer) {
    tl::register_thread(tl::role::exsia_fold, -1, -1);
    for (uint32_t s = 0; s < kStripes; ++s) {
        for (uint32_t t = 0; t < 2; ++t) {
            tl::record(stage::exsia_local, event::begin, 0, run, 7, s, t, s % 2, 0, 0);
            tl::record(stage::exsia_local, event::end, 0, run, 7, s, t, s % 2, 0, 128);
        }
        tl::record(stage::exsia_folding, event::begin, 0, run, 7, s, 0, s % 2, 0, 0);
        tl::record(stage::exsia_folding, event::end, 0, run, 7, s, 0, s % 2, 0, 256);
        const uint32_t edge = tl::next_edge_id();
        tl::record(stage::stripe_submit, event::handoff_send, 0, run, 7, s, 0, s % 2, edge, 0);
        to_issuer.push(s, edge);
    }
    tl::anchor();
}

void issuer_thread(uint32_t run, handoff_queue & from_fold, handoff_queue & to_final, bool & pinned, bool & ok) {
    const int cpu = sched_getcpu();
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    pinned = cpu >= 0 && pthread_setaffinity_np(pthread_self(), sizeof(set), &set) == 0 && sched_getcpu() == cpu;
    tl::register_thread(tl::role::issuer, -1, pinned ? cpu : -1);
    for (uint32_t i = 0; i < kStripes; ++i) {
        std::pair<uint32_t, uint32_t> item;
        tl::record(stage::issuer_ready_wait, event::wait_begin, 0, run, 7, tl::no_id, 0, 0, 0, 0);
        const bool got = from_fold.pop(item);
        tl::record(stage::issuer_ready_wait, event::wait_end, got ? 0 : 1, run, 7, tl::no_id, 0, 0, 0, 0);
        if (!got) {
            ok = false;
            return;
        }
        const uint32_t s = item.first;
        tl::record(stage::stripe_submit, event::handoff_recv, 0, run, 7, s, 0, s % 2, item.second, 0);
        tl::record(stage::acc_main_issue, event::begin, 0, run, 7, s, 0, s % 2, 0, 0);
        tl::record(stage::acc_main_issue, event::end, 0, run, 7, s, 0, s % 2, 0, 4096);
        tl::record(stage::acc_fence, event::begin, 0, run, 7, s, 0, s % 2, 0, 0);
        tl::record(stage::acc_fence, event::end, 0, run, 7, s, 0, s % 2, 0, 0);
        const uint32_t edge = tl::next_edge_id();
        tl::record(stage::final_handoff, event::handoff_send, 0, run, 7, s, 0, s % 2, edge, 0);
        to_final.push(s, edge);
    }
    tl::anchor();
}

void finalizer_thread(uint32_t run, handoff_queue & from_issuer, bool & ok) {
    tl::register_thread(tl::role::finalizer, -1, -1);
    for (uint32_t i = 0; i < kStripes; ++i) {
        std::pair<uint32_t, uint32_t> item;
        tl::record(stage::final_wait, event::wait_begin, 0, run, 7, tl::no_id, 0, 0, 0, 0);
        const bool got = from_issuer.pop(item);
        tl::record(stage::final_wait, event::wait_end, got ? 0 : 1, run, 7, tl::no_id, 0, 0, 0, 0);
        if (!got) {
            ok = false;
            return;
        }
        const uint32_t s = item.first;
        tl::record(stage::final_handoff, event::handoff_recv, 0, run, 7, s, 0, s % 2, item.second, 0);
        tl::record(stage::final_reconstruct_int64, event::begin, 0, run, 7, s, 0, s % 2, 0, 0);
        tl::record(stage::final_reconstruct_int64, event::end, 0, run, 7, s, 0, s % 2, 0, 64);
        tl::record(stage::final_slot_release, event::instant, 0, run, 7, s, 0, s % 2, 0, 1);
    }
    tl::anchor();
}

int child_disabled() {
    CHECK(!tl::enabled());
    CHECK(tl::register_thread(tl::role::caller, 0, 0) == tl::no_slot);
    CHECK(tl::begin_run(tl::op_type::linear, tl::phase::decode, 1) == 0);
    CHECK(tl::next_edge_id() == 0);
    for (int i = 0; i < 1000; ++i) {
        tl::record(stage::node_total, event::instant, 0, 0, 0, 0, 0, 0, 0, 0);
    }
    tl::anchor();
    tl::end_run(1);
    const tl::stats_t s = tl::stats();
    CHECK(s.threads == 0 && s.events == 0 && s.dropped == 0 && s.unregistered == 0);
    CHECK(!tl::flush());
    return g_failures == 0 ? 0 : 1;
}

int child_overflow() {
    const char * path = std::getenv("POTAL_TIMELINE");
    CHECK(tl::enabled() && path != nullptr);
    if (path == nullptr) {
        return 1;
    }
    CHECK(tl::register_thread(tl::role::caller, -1, -1) == 0);
    for (uint32_t i = 0; i < 100; ++i) {
        tl::record(stage::node_total, event::instant, 0, 0, 0, tl::no_id, i, 0, 0, i);
    }
    const tl::stats_t s = tl::stats();
    CHECK(s.threads == 1 && s.events == 64 && s.dropped == 36 && s.overflowed_threads == 1);
    CHECK(tl::flush());
    const auto t = rows_of(path, "T");
    CHECK(t.size() == 1);
    if (t.size() == 1) {
        CHECK(t[0][12] == "64" && t[0][13] == "36" && t[0][14] == "1");
    }
    const auto e = rows_of(path, "E");
    CHECK(e.size() == 64);
    for (size_t i = 0; i < e.size(); ++i) {
        CHECK(e[i][2] == std::to_string(i) && e[i][12] == std::to_string(i));
    }
    return g_failures == 0 ? 0 : 1;
}

int child_main() {
    const char * path = std::getenv("POTAL_TIMELINE");
    CHECK(tl::enabled() && path != nullptr);
    if (path == nullptr) {
        return 1;
    }
    CHECK(tl::register_thread(tl::role::caller, -1, -1) == 0);
    CHECK(tl::register_thread(tl::role::caller, -1, -1) == 0);
    const uint32_t run = tl::begin_run(tl::op_type::linear, tl::phase::decode, 7);
    CHECK(run != 0);
    tl::record(stage::node_total, event::begin, 0, run, 7, tl::no_id, 0, 0, 0, 0);

    handoff_queue to_issuer;
    handoff_queue to_final;
    bool          issuer_ok = true;
    bool          final_ok  = true;
    bool          pinned    = false;
    std::thread   fold([&] { fold_thread(run, to_issuer); });
    std::thread   issuer([&] { issuer_thread(run, to_issuer, to_final, pinned, issuer_ok); });
    std::thread   finalizer([&] { finalizer_thread(run, to_final, final_ok); });
    std::thread   stray([] { tl::record(stage::node_total, event::instant, 0, 0, 0, 0, 0, 0, 0, 0); });
    fold.join();
    issuer.join();
    finalizer.join();
    stray.join();
    CHECK(issuer_ok && final_ok);

    tl::record(stage::node_commit, event::begin, 0, run, 7, tl::no_id, 0, 0, 0, 0);
    tl::record(stage::node_commit, event::end, 0, run, 7, tl::no_id, 0, 0, 0, 0);
    tl::record(stage::node_total, event::end, 0, run, 7, tl::no_id, 0, 0, 0, 0);
    tl::end_run(run);

    const uint32_t    expected = kStripes * (7 + 8 + 6) + 4;
    const tl::stats_t s        = tl::stats();
    CHECK(s.threads == 4 && s.events == expected && s.dropped == 0 && s.overflowed_threads == 0 &&
          s.unregistered == 1);
    CHECK(tl::flush());

    const std::string text = slurp(path);
    CHECK(text.rfind("# potal_timeline v1\nMETA,format=potal_timeline_v1,", 0) == 0);
    CHECK(text.find(",model_id=timeline-test,") != std::string::npos);
    CHECK(text.find("POTAL_TIMELINE=") != std::string::npos);
    CHECK(text.find("unregistered_records=1") != std::string::npos);
    CHECK(text.find("#END,events=" + std::to_string(expected) + "\n") != std::string::npos);

    const auto threads = rows_of(path, "T");
    CHECK(threads.size() == 4);
    std::map<std::string, std::vector<std::string>> by_role;
    for (const auto & t : threads) {
        CHECK(t.size() == 19);
        by_role[t[2]] = t;
    }
    CHECK(by_role.count("caller") && by_role.count("exsia_fold") && by_role.count("issuer") &&
          by_role.count("finalizer"));
    if (pinned && by_role.count("issuer")) {
        CHECK(by_role["issuer"][8] == "1" && by_role["issuer"][9] == "1" && by_role["issuer"][11] == "0");
    }
    if (by_role.count("caller")) {
        CHECK(by_role["caller"][8] == "0");
    }

    std::map<std::string, int> anchors_per_slot;
    for (const auto & a : rows_of(path, "A")) {
        ++anchors_per_slot[a[1]];
    }
    for (const auto & t : threads) {
        CHECK(anchors_per_slot[t[1]] >= 2);
    }

    std::map<std::string, bool> unpinned;
    for (const auto & t : threads) {
        unpinned[t[1]] = std::atoi(t[4].c_str()) < 0;
    }
    const auto events = rows_of(path, "E");
    CHECK(events.size() == expected);
    std::map<std::string, long>                   last_seq;
    std::map<std::string, std::pair<int, int>>    edges;
    for (const auto & e : events) {
        CHECK(e.size() == 17);
        const bool wait_or_handoff = e[5] == "3" || e[5] == "4" || e[5] == "5" || e[5] == "6";
        CHECK((e[16] != "0") == (wait_or_handoff || unpinned[e[3]]));
        const long seq = std::atol(e[2].c_str());
        CHECK(!last_seq.count(e[3]) ? seq == 0 : seq == last_seq[e[3]] + 1);
        last_seq[e[3]] = seq;
        CHECK(e[7] == "1" && e[8] == "2");
        if (e[5] == "5") {
            ++edges[e[14]].first;
        } else if (e[5] == "6") {
            ++edges[e[14]].second;
        }
    }
    CHECK(edges.size() == 2 * kStripes);
    for (const auto & kv : edges) {
        CHECK(kv.first != "0" && kv.second.first == 1 && kv.second.second == 1);
    }
    return g_failures == 0 ? 0 : 1;
}

int spawn_child(const char * mode, const std::vector<std::string> & extra_env) {
    std::vector<std::string> env;
    for (char ** e = environ; *e != nullptr; ++e) {
        if (std::strncmp(*e, "POTAL_TIMELINE", 14) != 0) {
            env.emplace_back(*e);
        }
    }
    env.insert(env.end(), extra_env.begin(), extra_env.end());
    std::vector<char *> envp;
    for (auto & s : env) {
        envp.push_back(s.data());
    }
    envp.push_back(nullptr);
    std::string         arg = std::string("--child=") + mode;
    char                exe[] = "/proc/self/exe";
    std::vector<char *> argv  = { exe, arg.data(), nullptr };
    pid_t               pid;
    if (posix_spawn(&pid, exe, nullptr, nullptr, argv.data(), envp.data()) != 0) {
        std::fprintf(stderr, "FAIL: cannot spawn the %s child\n", mode);
        return -1;
    }
    int status = 0;
    if (waitpid(pid, &status, 0) != pid || !WIFEXITED(status)) {
        std::fprintf(stderr, "FAIL: %s child did not exit normally (status %d)\n", mode, status);
        return -1;
    }
    return WEXITSTATUS(status);
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc == 2 && std::strcmp(argv[1], "--child=disabled") == 0) {
        return child_disabled();
    }
    if (argc == 2 && std::strcmp(argv[1], "--child=overflow") == 0) {
        return child_overflow();
    }
    if (argc == 2 && std::strcmp(argv[1], "--child=main") == 0) {
        return child_main();
    }

    std::string main_path;
    if (const char * p = std::getenv("POTAL_TIMELINE"); p != nullptr && *p != '\0') {
        main_path = p;
    } else {
        const char * tmp = std::getenv("TMPDIR");
        std::string  tmpl = std::string(tmp != nullptr && *tmp != '\0' ? tmp : "/tmp") + "/potal-timeline-test-XXXXXX";
        if (mkdtemp(tmpl.data()) == nullptr) {
            std::fprintf(stderr, "FAIL: mkdtemp %s\n", tmpl.c_str());
            return 1;
        }
        main_path = tmpl + "/timeline.csv";
    }
    const std::string overflow_path = main_path + ".overflow.csv";

    const int rc_disabled = spawn_child("disabled", {});
    const int rc_overflow =
        spawn_child("overflow", { "POTAL_TIMELINE=" + overflow_path, "POTAL_TIMELINE_EVENTS=64" });
    const int rc_main =
        spawn_child("main", { "POTAL_TIMELINE=" + main_path, "POTAL_TIMELINE_MODEL_ID=timeline-test" });
    CHECK(rc_disabled == 0);
    CHECK(rc_overflow == 0);
    CHECK(rc_main == 0);

    const std::string text     = slurp(main_path);
    const uint32_t    expected = kStripes * (7 + 8 + 6) + 4;
    CHECK(text.find("#END,events=" + std::to_string(expected) + "\n") != std::string::npos);

    std::printf("potal-timeline-test: disabled=%d overflow=%d main=%d trace=%s\n", rc_disabled, rc_overflow, rc_main,
                main_path.c_str());
    if (g_failures != 0) {
        std::printf("potal-timeline-test: FAILED (%d)\n", g_failures);
        return 1;
    }
    std::printf("potal-timeline-test: PASSED\n");
    return 0;
}
