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
    evaluation_lifecycle pipeline("potal_collection", "candidate", 1, false, true);
    pipeline.phase("prefill", {}, 0);
    pipeline.request_start(0);
    pipeline.prefill_batch_ready(0, 0);
    pipeline.begin_dispatch(0);
    pipeline.end_dispatch(1, 0);
    pipeline.sample(0, 41);
    pipeline.pipeline_parent({{"operation_id", 7}, {"parent_id", 3},
        {"required_work_ids", {10, 11}}, {"fence_call_id", 9},
        {"fence_required_work_ids", {10, 11}}, {"phase_id", 0},
        {"production_geometry_version", 1}, {"scope", "STREAM"},
        {"activation_bits", 8}, {"weight_bits", 8}, {"dim", 16},
        {"parent_m", 65}, {"n", 32}, {"k", 96},
        {"tile_i_count", 5}, {"tile_j_count", 2}, {"tile_k_count", 6},
        {"residual_bindings", {{{"work_id", 11}, {"call_id", 8},
            {"child_parent_id", 4}, {"dense_work_id", 10},
            {"dense_parent_id", 3}, {"stripe_id", 0},
            {"row_begin", 0}, {"row_end", 65},
            {"source_row_begin", 0}, {"source_row_count", 65}}}}});
    pipeline.pipeline_owner({{"producer_sequence", 0}, {"phase_id", 0},
        {"operation_id", 7}, {"parent_id", 3}, {"work_id", 10},
        {"producer_run_id", 42}, {"stripe_id", 0}, {"workspace_slot", 0},
        {"target_npu_slot", nullptr}, {"row_begin", 0}, {"row_end", 64},
        {"resource", "FRONTEND_QUEUE"}, {"transition", "ENQUEUE"},
        {"required_work_ids", nlohmann::json::array()},
        {"required_call_ids", nlohmann::json::array()},
        {"source_location", "frontend/src/im2p_gemmini_frontend.cpp:submit_stripe_planned"}});
    const auto pipeline_events = pipeline.finish(true, 1);
    assert(pipeline_events.front().at("version") == 3);
    assert(pipeline_events.front().at("target_mode_scope") == "STRIPE_PIPELINE_ONLY");
    assert(pipeline_events[pipeline_events.size() - 3].at("kind") == "PIPELINE_PARENT");
    assert(pipeline_events[pipeline_events.size() - 2].at("kind") == "PIPELINE_OWNER");
    assert(pipeline_events.back().at("kind") == "RUN_END");
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
    if (argc >= 4) {
        std::ofstream output(argv[3]);
        for (const auto &event : pipeline_events) output << event.dump() << '\n';
        output.close();
        assert(output);
    }
    std::cout << "actual lifecycle recorder: sampled and forced128 cost-only coverage PASS\n";
}
