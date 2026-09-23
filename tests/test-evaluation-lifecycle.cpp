#include "../tools/eval/evaluation-lifecycle.h"
#include <cassert>
#include <iostream>
#include <fstream>

template<class Action> static void must_reject(Action action) {
    bool failed = false;
    try { action(); } catch (const std::runtime_error &) { failed = true; }
    assert(failed);
}

int main(int argc, char ** argv) {
    evaluation_lifecycle trace("potal_collection", "candidate", 2);
    trace.phase("prefill", {}, 0);
    trace.begin_dispatch(0);
    trace.end_dispatch(1, 0);
    trace.sample(0, 41);
    trace.phase("decode", 0, 1);
    trace.begin_dispatch(1);
    trace.end_dispatch(2, 0);
    trace.sample(1, 42);
    const auto events = trace.finish(true, 2);
    assert(events.front().at("kind") == "RUN");
    assert(events.back().at("samples") == 2);
    assert(events.back().at("graphs") == 2);
    assert(events[3].at("graph_end") == 1);
    assert(events[4].at("token_id") == 41);
    bool refused = false;
    try { evaluation_lifecycle bad("potal_collection", "candidate", 2); bad.phase("decode", 0, 0); }
    catch (const std::runtime_error &) { refused = true; }
    assert(refused);
    evaluation_lifecycle forced("full_cpu", "candidate", 128, true);
    for (uint64_t index = 0; index != 128; ++index) {
        forced.phase(index == 0 ? "prefill" : "decode", index == 0 ? std::nullopt : std::optional<uint64_t>(index - 1), index);
        forced.begin_dispatch(index);
        forced.end_dispatch(index + 1, 0);
        must_reject([&] { forced.sample(index, 100 + index); });
        forced.forced_token(index, 100 + index);
        if (index == 126) must_reject([&] { forced.finish(true, 127); });
    }
    const auto forced_events = forced.finish(true, 128);
    assert(forced_events.front().at("execution_kind") == "FORCED_CPU_COST_ONLY");
    assert(forced_events.front().at("trajectory_source") == "POTAL");
    assert(forced_events.front().at("version") == 2);
    assert(forced_events.back().at("samples") == 0);
    assert(forced_events.back().at("forced_tokens") == 128);
    assert(forced_events.back().at("completed_tokens") == 128);
    uint64_t observed_forced = 0;
    for (const auto & event : forced_events) {
        assert(event.at("kind") != "SAMPLE");
        if (event.at("kind") == "FORCED_TOKEN") ++observed_forced;
    }
    assert(observed_forced == 128);
    must_reject([&] { evaluation_lifecycle bad("potal_collection", "candidate", 128, true); });
    must_reject([&] { evaluation_lifecycle bad("full_cpu", "candidate", 127, true); });
    must_reject([&] { trace.forced_token(2, 43); });
    if (argc >= 2) {
        std::ofstream output(argv[1]);
        for (const auto & event : events) output << event.dump() << '\n';
        output.close();
        assert(output);
    }
    if (argc == 3) {
        std::ofstream output(argv[2]);
        for (const auto & event : forced_events) output << event.dump() << '\n';
        output.close();
        assert(output);
    }
    std::cout << "actual lifecycle recorder: sampled and forced128 cost-only coverage PASS\n";
}
