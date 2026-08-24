#include "../ggml/src/ggml-gemmini/ggml-gemmini-matmul.hpp"
#include "../ggml/src/ggml-gemmini/ggml-gemmini-telemetry.hpp"
#include "../ggml/src/ggml-gemmini/quants/act/exsia/exsia.hpp"
#include <gemmini/cycle_reader.hpp>
#include <gemmini/log.hpp>
#include "../ggml/src/ggml-gemmini/residual/residual-capture.hpp"
#include "../ggml/src/ggml-gemmini/residual/rmd/rmd-builder.hpp"
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <type_traits>
using namespace ggml::gemmini;
namespace {
bool expect(bool condition, const char * message) {
    if (!condition) std::fprintf(stderr, "FAIL: %s\n", message);
    return condition;
}
RmdTelemetryRecord cpu_record() {
    RmdTelemetryRecord record{};
    record.runtime_bundle_id = "bundle-7"; record.model_id = "model-hash"; record.run_id = "run-42";
    record.backend = RmdBackend::cpu_direct; record.source = MatmulOptionSource::explicit_override;
    record.units = cycle::units(); record.work = true;
    record.counters.direct_events = 9; record.counters.direct_calls = 2;
    record.timing.prep = 11; record.timing.backend_service = 31; record.timing.merge = 7;
    record.timing.residual_total = 47; record.timing.queue = 3;
    record.timing.dense_end = 120; record.timing.residual_start = 120; record.invocation_total = 101;
    return record;
}
RmdTelemetryRecord ws_record() {
    RmdTelemetryRecord record = cpu_record(); record.backend = RmdBackend::gemmini_ws_compact;
    record.counters = {}; record.counters.packet_calls = 2; record.counters.ws_calls = 6;
    record.geometry.packet_count = 2; record.geometry.active_blocks = 3;
    record.geometry.compact_k_count = 17; record.geometry.padded_k_count = 32;
    record.geometry.physical_tile_count = 12; return record;
}
std::string read_file(const std::filesystem::path & path) {
    std::ifstream input(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}
std::size_t count_occurrences(const std::string & value, const std::string & needle) {
    std::size_t count = 0;
    for (std::size_t pos = 0; (pos = value.find(needle, pos)) != std::string::npos; pos += needle.size()) ++count;
    return count;
}

bool aggregate_serializer_fixtures() {
    static_assert(std::is_same_v<decltype(WsLoopTelemetry::load_occupancy_cycles), std::uint32_t>);
    static_assert(std::is_same_v<decltype(Im2pExecutionTelemetry::run_id), std::uint64_t>);
    static_assert(std::is_same_v<decltype(Im2pExecutionTelemetry::rtl_work_total_cycles), std::uint64_t>);
    static_assert(std::is_same_v<decltype(Im2pStripeTelemetry::publish_cycle), std::uint64_t>);
    static_assert(std::is_same_v<decltype(Im2pStripeTelemetry::completion_cycle), std::uint64_t>);
    CycleIntervalTelemetry interval{};
    interval.layer = "ffn\"norm"; interval.name = "dense";
    interval.start = 10; interval.end = 34;
    const std::string interval_json = serialize_cycle_telemetry(interval);
    const std::string expected_interval =
        "{\"schema\":\"gemmini.cycle\",\"version\":1,\"record_type\":\"CYCLE_INTERVAL\","
        "\"source\":\"host_tick\",\"unit\":\"tick\",\"layer\":\"ffn\\\"norm\","
        "\"name\":\"dense\",\"start\":10,\"end\":34,\"delta\":24,\"valid\":true}";

    WsLoopTelemetry ws{};
    ws.problem_i = 256; ws.problem_j = 768; ws.problem_k = 768;
    ws.tile_i = 5; ws.tile_j = 3; ws.tile_k = 6;
    ws.gemmini_outer_i = 4; ws.gemmini_outer_j = 29; ws.gemmini_outer_k = 1;
    ws.ws_inner_calls = 116;
    ws.containing_interval_cycles = 1000;
    ws.load_occupancy_cycles = 101; ws.execute_occupancy_cycles = 202;
    ws.store_occupancy_cycles = 303; ws.loop_occupancy_cycles = 999;
    const std::string ws_json = serialize_cycle_telemetry(ws);
    const std::string expected_ws =
        "{\"schema\":\"gemmini.cycle\",\"version\":1,\"record_type\":\"WS_LOOP_TELEMETRY\","
        "\"source\":\"gemmini_hw_counter\",\"unit\":\"cycle\",\"problem_i\":256,\"problem_j\":768,\"problem_k\":768,"
        "\"tile_i\":5,\"tile_j\":3,\"tile_k\":6,\"gemmini_outer_i\":4,\"gemmini_outer_j\":29,\"gemmini_outer_k\":1,"
        "\"ws_inner_calls\":116,\"containing_interval_cycles\":1000,\"containing_interval_counter_bits\":64,"
        "\"load_occupancy_cycles\":101,\"execute_occupancy_cycles\":202,\"store_occupancy_cycles\":303,"
        "\"loop_occupancy_cycles\":999,\"occupancy_counter_bits\":32,\"valid\":true}";
    WsLoopTelemetry invalid_ws = ws;
    invalid_ws.load_occupancy_cycles = 1001;
    WsLoopTelemetry wrapped_ws = ws;
    wrapped_ws.containing_interval_cycles = static_cast<std::uint64_t>(UINT32_MAX) + 1;

    Im2pExecutionTelemetry rtl{};
    rtl.layer = "blk.15.mlp.down_proj"; rtl.run_id = 17;
    rtl.mode = "stripe_pipeline"; rtl.activation_bits = 8; rtl.weight_bits = 8; rtl.dim = 16;
    rtl.problem_i = 256; rtl.problem_j = 768; rtl.problem_k = 768;
    rtl.tile_i = 5; rtl.tile_j = 3; rtl.tile_k = 6;
    rtl.rtl_work_total_cycles = 5000; rtl.rtl_compute_cycles = 3000;
    rtl.rtl_drain_cycles = 120; rtl.rtl_activation_wait_cycles = 41;
    rtl.rtl_weight_wait_cycles = 42; rtl.rtl_scale_wait_cycles = 43;
    rtl.rtl_output_wait_cycles = 44; rtl.rtl_overlap_cycles = 900;
    rtl.rtl_activation_overlap_cycles = 300; rtl.rtl_weight_overlap_cycles = 400;
    rtl.rtl_scale_overlap_cycles = 250; rtl.rtl_completed_output_works = 16;
    rtl.rtl_completed_fragments = 48; rtl.rtl_scheduler_groups_completed = 4;
    rtl.rtl_stripes_published = 4; rtl.rtl_stripe_rows_published = 256;
    const std::string rtl_json = serialize_cycle_telemetry(rtl);
    const std::string expected_rtl =
        "{\"schema\":\"gemmini.cycle\",\"version\":1,\"record_type\":\"IM2P_EXECUTION_TELEMETRY\","
        "\"source\":\"im2p_rtl\",\"unit\":\"rtl_cycle\",\"layer\":\"blk.15.mlp.down_proj\","
        "\"run_id\":17,\"rtl_work_total_cycles\":5000}";
    Im2pExecutionTelemetry malformed_rtl{};
    malformed_rtl.layer = "bad\"\\\n";
    malformed_rtl.layer.push_back('\x01');
    malformed_rtl.rtl_work_total_cycles = 1;
    const std::string expected_malformed_rtl =
        "{\"schema\":\"gemmini.cycle\",\"version\":1,\"record_type\":\"IM2P_EXECUTION_TELEMETRY\","
        "\"source\":\"im2p_rtl\",\"unit\":\"rtl_cycle\",\"layer\":\"bad\\\"\\\\\\n\\u0001\","
        "\"run_id\":0,\"rtl_work_total_cycles\":1}";
    const std::string malformed_rtl_json = serialize_cycle_telemetry(malformed_rtl);
    Im2pExecutionTelemetry empty_rtl{};
    const std::string expected_empty_rtl =
        "{\"schema\":\"gemmini.cycle\",\"version\":1,\"record_type\":\"IM2P_EXECUTION_TELEMETRY\","
        "\"source\":\"im2p_rtl\",\"unit\":\"rtl_cycle\",\"layer\":\"\",\"run_id\":0,\"rtl_work_total_cycles\":0}";
    const std::string empty_rtl_json = serialize_cycle_telemetry(empty_rtl);

    Im2pStripeTelemetry stripe{};
    stripe.layer = "blk.15.mlp.down_proj"; stripe.run_id = 17; stripe.stripe_id = 2;
    stripe.slot = 1; stripe.row_begin = 80; stripe.row_end = 160;
    stripe.publish_cycle = 100; stripe.completion_cycle = 116;
    const std::string stripe_json = serialize_cycle_telemetry(stripe);
    const std::string expected_stripe =
        "{\"schema\":\"gemmini.cycle\",\"version\":1,\"record_type\":\"IM2P_STRIPE_TELEMETRY\","
        "\"source\":\"im2p_rtl\",\"unit\":\"rtl_cycle\",\"layer\":\"blk.15.mlp.down_proj\","
        "\"run_id\":17,\"stripe_id\":2,\"slot\":1,\"row_begin\":80,\"row_end\":160,"
        "\"publish_cycle\":100,\"completion_cycle\":116,\"latency_cycles\":16,\"additive\":false}";
    Im2pStripeTelemetry overlap_stripe = stripe;
    overlap_stripe.stripe_id = 3; overlap_stripe.slot = 0; overlap_stripe.row_begin = 160;
    overlap_stripe.row_end = 256; overlap_stripe.publish_cycle = 108; overlap_stripe.completion_cycle = 124;
    const std::string expected_overlap_stripe =
        "{\"schema\":\"gemmini.cycle\",\"version\":1,\"record_type\":\"IM2P_STRIPE_TELEMETRY\","
        "\"source\":\"im2p_rtl\",\"unit\":\"rtl_cycle\",\"layer\":\"blk.15.mlp.down_proj\","
        "\"run_id\":17,\"stripe_id\":3,\"slot\":0,\"row_begin\":160,\"row_end\":256,"
        "\"publish_cycle\":108,\"completion_cycle\":124,\"latency_cycles\":16,\"additive\":false}";
    Im2pStripeTelemetry zero_stripe{};
    const std::string expected_zero_stripe =
        "{\"schema\":\"gemmini.cycle\",\"version\":1,\"record_type\":\"IM2P_STRIPE_TELEMETRY\","
        "\"source\":\"im2p_rtl\",\"unit\":\"rtl_cycle\",\"layer\":\"\",\"run_id\":0,\"stripe_id\":0,"
        "\"slot\":0,\"row_begin\":0,\"row_end\":0,\"publish_cycle\":0,\"completion_cycle\":0,"
        "\"latency_cycles\":0,\"additive\":false}";
    Im2pStripeTelemetry wrapped_stripe = stripe;
    wrapped_stripe.publish_cycle = UINT64_MAX - 2; wrapped_stripe.completion_cycle = 3;
    const std::string expected_wrapped_stripe =
        "{\"schema\":\"gemmini.cycle\",\"version\":1,\"record_type\":\"IM2P_STRIPE_TELEMETRY\","
        "\"source\":\"im2p_rtl\",\"unit\":\"rtl_cycle\",\"layer\":\"blk.15.mlp.down_proj\","
        "\"run_id\":17,\"stripe_id\":2,\"slot\":1,\"row_begin\":80,\"row_end\":160,"
        "\"publish_cycle\":18446744073709551613,\"completion_cycle\":3,\"latency_cycles\":6,\"additive\":false}";
    const std::string overlap_stripe_json = serialize_cycle_telemetry(overlap_stripe);
    const std::string zero_stripe_json = serialize_cycle_telemetry(zero_stripe);
    const std::string wrapped_stripe_json = serialize_cycle_telemetry(wrapped_stripe);

    PipelineStripeTelemetry pipeline{};
    pipeline.layer = "ffn"; pipeline.run_id = 7; pipeline.stripe_id = 2;
    pipeline.slot = 1; pipeline.row_begin = 80; pipeline.row_end = 160;
    pipeline.queue_start_ns = 10; pipeline.queue_end_ns = 12;
    pipeline.dense_start_ns = 12; pipeline.dense_end_ns = 30;
    pipeline.rmd_start_ns = 30; pipeline.rmd_end_ns = 40;
    pipeline.compose_start_ns = 40; pipeline.compose_end_ns = 44;
    pipeline.finalize_start_ns = 44; pipeline.finalize_end_ns = 48;
    const std::string pipeline_json = serialize_cycle_telemetry(pipeline);
    const std::string expected_pipeline =
        "{\"schema\":\"gemmini.cycle\",\"version\":1,\"record_type\":\"PIPELINE_STRIPE_SUMMARY\","
        "\"source\":\"steady_clock\",\"unit\":\"nanosecond\",\"layer\":\"ffn\",\"run_id\":7,\"stripe_id\":2,\"slot\":1,"
        "\"row_begin\":80,\"row_end\":160,\"queue_start_ns\":10,\"queue_end_ns\":12,\"dense_start_ns\":12,\"dense_end_ns\":30,"
        "\"rmd_start_ns\":30,\"rmd_end_ns\":40,\"compose_start_ns\":40,\"compose_end_ns\":44,"
        "\"finalize_start_ns\":44,\"finalize_end_ns\":48,\"valid\":true}";

    if (std::getenv("GEMMINI_TELEMETRY_PRINT_ALL") != nullptr) {
        std::printf("%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n", interval_json.c_str(), ws_json.c_str(),
                    rtl_json.c_str(), malformed_rtl_json.c_str(), empty_rtl_json.c_str(), stripe_json.c_str(),
                    overlap_stripe_json.c_str(), zero_stripe_json.c_str(), wrapped_stripe_json.c_str(), pipeline_json.c_str());
    }
#if !LOG_CYCLE
    return expect(interval_json.empty() && ws_json.empty() && rtl_json.empty() && stripe_json.empty() &&
                      overlap_stripe_json.empty() && zero_stripe_json.empty() && wrapped_stripe_json.empty() &&
                      pipeline_json.empty(),
                  "cycle-off suppresses every aggregate serializer");
#else
    return expect(interval_json == expected_interval, "cycle interval exact schema") &&
        expect(ws_json == expected_ws, "WS exact schema and 32-bit occupancy") &&
        expect(serialize_cycle_telemetry(invalid_ws).find("\"valid\":false") != std::string::npos,
               "hardware occupancy outside containing interval is invalid") &&
        expect(serialize_cycle_telemetry(wrapped_ws).find("\"valid\":false") != std::string::npos,
               "hardware containing interval wider than occupancy counter is invalid") &&
        expect(rtl_json == expected_rtl, "RTL aggregate has run correlation and exactly one cycle payload") &&
        expect(count_occurrences(rtl_json, "\":") == 8, "RTL aggregate schema has exactly eight top-level fields") &&
        expect(malformed_rtl_json == expected_malformed_rtl,
               "RTL semantic layer escapes quotes, backslashes, newline, and control bytes") &&
        expect(empty_rtl_json == expected_empty_rtl,
               "empty RTL semantic layer remains deterministic valid JSON") &&
        expect(stripe_json == expected_stripe, "RTL stripe normal schema is exact") &&
        expect(overlap_stripe_json == expected_overlap_stripe,
               "RTL stripe overlap preserves raw endpoints and non-additive latency") &&
        expect(zero_stripe_json == expected_zero_stripe,
               "RTL stripe zero endpoints are serialized rather than treated as missing") &&
        expect(wrapped_stripe_json == expected_wrapped_stripe,
               "RTL stripe wrapped latency uses unsigned endpoint subtraction") &&
        expect(pipeline_json == expected_pipeline, "pipeline exact schema remains host-nanosecond-only") &&
        [&] {
            const std::uint64_t first_generic_run_id = quants::act::exsia::next_exsia_run_id();
            const std::uint64_t second_generic_run_id = quants::act::exsia::next_exsia_run_id();
            return expect(first_generic_run_id != second_generic_run_id,
                          "generic and ExSIA runs share a distinct atomic run-ID sequence");
        }();
#endif
}

bool aggregate_cycle_sink_fixtures() {
    const auto root = std::filesystem::temp_directory_path() / "gemmini-aggregate-cycle-sink";
    std::error_code error;
    std::filesystem::remove_all(root, error);
    std::filesystem::create_directory(root, error);
    const auto cycle_path = root / "cycle-log.jsonl";
    const auto debug_path = root / "debug-log.jsonl";
#if LOG_CYCLE
    if (!expect(ggml::gemmini::log::cycle.set_output_path(cycle_path.c_str(), true),
                "aggregate cycle sink setup")) return false;
#endif
#if LOG_DEBUG
    if (!expect(ggml::gemmini::log::debug.set_output_path(debug_path.c_str(), true),
                "aggregate debug sink setup")) return false;
#endif

    CycleIntervalTelemetry interval{}; interval.layer = "driver"; interval.name = "interval";
    interval.start = 1; interval.end = 2;
    WsLoopTelemetry ws{}; ws.containing_interval_cycles = 4; ws.loop_occupancy_cycles = 3;
    Im2pExecutionTelemetry rtl{}; rtl.layer = "blk.15.mlp.down_proj"; rtl.run_id = 17; rtl.mode = "full";
    rtl.rtl_work_total_cycles = 9; rtl.rtl_compute_cycles = 7;
    Im2pStripeTelemetry stripe{}; stripe.layer = "blk.15.mlp.down_proj"; stripe.run_id = 17;
    stripe.row_end = 1; stripe.completion_cycle = 1;
    PipelineStripeTelemetry pipeline{}; pipeline.layer = "driver"; pipeline.row_end = 1;
    const RmdTelemetryRecord rmd = cpu_record();
    emit_cycle_telemetry(interval);
    emit_cycle_telemetry(ws);
    emit_cycle_telemetry(rtl);
    emit_cycle_telemetry(stripe);
    emit_cycle_telemetry(pipeline);
    emit_cycle_telemetry(rmd);
    ggml::gemmini::log::cycle.set_output(stderr);
    ggml::gemmini::log::debug.set_output(stderr);

    const std::string cycle_output = read_file(cycle_path);
    const std::string debug_output = read_file(debug_path);
    bool ok = true;
#if LOG_CYCLE
    const char * types[] = {"CYCLE_INTERVAL", "WS_LOOP_TELEMETRY", "IM2P_EXECUTION_TELEMETRY",
                            "IM2P_STRIPE_TELEMETRY", "PIPELINE_STRIPE_SUMMARY", "RMD_BACKEND_TELEMETRY"};
    for (const char * type : types) {
        ok &= expect(cycle_output.find(std::string("\"record_type\":\"") + type + "\"") != std::string::npos,
                     "aggregate record reaches cycle sink");
    }
    ok &= expect(count_occurrences(cycle_output, "\"record_type\":") == 6 &&
                 cycle_output.find("\"additive\":false") != std::string::npos,
                 "cycle sink receives one non-additive RTL stripe row");
#else
    ok &= expect(cycle_output.empty(), "cycle-off suppresses aggregate records");
#endif
#if LOG_DEBUG
    ok &= expect(debug_output.find("\"layer\":\"blk.15.mlp.down_proj\"") != std::string::npos &&
                 debug_output.find("\"layer\":\"im2p_rtl\"") == std::string::npos &&
                 count_occurrences(debug_output, "IM2P_EXECUTION_TELEMETRY_DETAIL") == 1 &&
                 debug_output.find("rtl_work_total_cycles=9") != std::string::npos &&
                 debug_output.find("rtl_compute_cycles=7") != std::string::npos &&
                 debug_output.find("IM2P_STRIPE_TELEMETRY") == std::string::npos,
                 "IM2P stripe telemetry stays out of the debug sink");
#else
    ok &= expect(debug_output.empty(), "debug-off suppresses IM2P execution detail");
#endif
    ok &= expect(!std::filesystem::exists(root / "log-ws-loop.jsonl"), "legacy WS aggregate is absent");
    if (std::getenv("GEMMINI_TELEMETRY_KEEP_ARTIFACTS") == nullptr) std::filesystem::remove_all(root, error);
    return ok && !error;
}

bool run_aggregate_driver(const std::filesystem::path & cycle_path) {
    cycle::reset_read_count_for_test();
    (void) cycle::read();
    (void) cycle::timestamp_ns();
#if LOG_CYCLE
    if (!ggml::gemmini::log::cycle.set_output_path(cycle_path.c_str(), true)) return false;
#endif
    CycleIntervalTelemetry interval{}; interval.layer = "driver"; interval.name = "interval";
    interval.start = 10; interval.end = 11;
    WsLoopTelemetry ws{}; ws.containing_interval_cycles = 10; ws.loop_occupancy_cycles = 4;
    Im2pExecutionTelemetry rtl{}; rtl.mode = "full";
    Im2pStripeTelemetry stripe{}; stripe.row_end = 1; stripe.completion_cycle = 1;
    PipelineStripeTelemetry pipeline{}; pipeline.layer = "driver"; pipeline.row_end = 1;
    emit_cycle_telemetry(interval); emit_cycle_telemetry(ws); emit_cycle_telemetry(rtl);
    emit_cycle_telemetry(stripe); emit_cycle_telemetry(pipeline); emit_cycle_telemetry(cpu_record());
    ggml::gemmini::log::cycle.set_output(stderr);
#if LOG_CYCLE
    const std::string output = read_file(cycle_path);
    return cycle::read_count_for_test() == 2 &&
        output.find("\"record_type\":\"CYCLE_INTERVAL\"") != std::string::npos &&
        output.find("\"record_type\":\"WS_LOOP_TELEMETRY\"") != std::string::npos &&
        output.find("\"record_type\":\"IM2P_EXECUTION_TELEMETRY\"") != std::string::npos &&
        output.find("\"record_type\":\"IM2P_STRIPE_TELEMETRY\"") != std::string::npos &&
        output.find("\"record_type\":\"PIPELINE_STRIPE_SUMMARY\"") != std::string::npos &&
        output.find("\"record_type\":\"RMD_BACKEND_TELEMETRY\"") != std::string::npos;
#else
    return cycle::read_count_for_test() == 0 && !std::filesystem::exists(cycle_path);
#endif
}

bool residual_capture_timer_seam() {
    cycle::reset_read_count_for_test();
    residual::TimedResidualCapture capture(residual::ResidualRoute::cpu_direct);
    capture.reset(0, 0, 1, 4, 2);
    if (!expect(capture.add_residual(0, 1, 7), "residual timer fixture accepts work")) return false;
    const residual::ResidualStripePayload payload = capture.finish();
    if (!expect(payload.direct != nullptr, "residual timer fixture produces payload")) return false;
#if LOG_CYCLE
    return expect(cycle::read_count_for_test() == 2,
                  "residual capture routes both profiling reads through timer seam");
#else
    return expect(cycle::read_count_for_test() == 0 && payload.capture_ns == 0,
                  "cycle-off residual capture performs zero profiling reads");
#endif
}

bool negative_fixtures() {
    RmdTelemetryRecord malformed = cpu_record(); malformed.schema = "wrong";
    RmdTelemetryRecord zero = cpu_record(); zero.work = false; zero.counters = {};
    RmdTelemetryRecord wrong_unit = cpu_record(); wrong_unit.units = wrong_unit.units == "ticks" ? "cycles" : "ticks";
    RmdTelemetryRecord ordering = cpu_record(); ordering.timing.dense_end = ordering.timing.residual_start + 1;
    RmdTelemetryRecord containment = cpu_record(); containment.timing.residual_total = containment.timing.backend_service - 1;
    return expect(!check_rmd_telemetry(malformed, cycle::units(), true).ok(), "malformed schema rejected") &&
        expect(!check_rmd_telemetry(zero, cycle::units(), true).ok(), "zero work rejected for comparison") &&
        expect(check_rmd_telemetry(zero, cycle::units(), false).ok(), "zero work explicit outside comparison") &&
        expect(!check_rmd_telemetry(wrong_unit, cycle::units(), true).ok(), "wrong units rejected") &&
        expect(!check_rmd_telemetry(ordering, cycle::units(), true).ok(), "ordering violation rejected") &&
        expect(!check_rmd_telemetry(containment, cycle::units(), true).ok(), "service containment enforced");
}

bool hash_parity_and_mismatch_fixtures() {
    residual::DirectStripePayload direct{};
    direct.stripe_id = 3; direct.row_begin = 7; direct.row_count = 2;
    direct.logical_k = 64; direct.logical_j = 3;
    direct.events = {{0, 1, 129}, {0, 33, -257}, {1, 2, 65537}};
    rmd::RmdStripeBuilder builder;
    builder.reset(3, 7, 2, 64, 3);
    for (const auto & event : direct.events) {
        if (!builder.add_residual(event.local_row, event.original_k, event.residual)) return false;
    }
    const auto packet = builder.finish();
    if (!expect(packet != nullptr, "WS parity packet built")) return false;
    const std::string cpu_input = rmd_input_hash(direct);
    const std::string ws_input = rmd_input_hash(*packet);
    if (!expect(cpu_input.size() == 16 && cpu_input == ws_input,
                "CPU direct and WS packet canonical input hashes match")) return false;

    RmdTelemetryRecord cpu = cpu_record();
    cpu.stripes = {{0,0,1,{1,2,2,3,4,5,6,7},cpu_input,"1111111111111111","2222222222222222"}};
    RmdTelemetryRecord ws = ws_record(); ws.stripes = cpu.stripes;
    if (!expect(compare_rmd_telemetry_proofs(cpu, ws).ok(), "equal-work proofs compare")) return false;
    RmdTelemetryRecord mismatch = ws;
    mismatch.stripes[0].input_hash[0] ^= 1;
    const bool input = expect(compare_rmd_telemetry_proofs(cpu, mismatch).code ==
                              RmdTelemetryCheckCode::input_hash_mismatch,
                              "input hash mismatch rejected");
    mismatch = ws; mismatch.stripes[0].correction_hash[0] ^= 1;
    const bool correction = expect(compare_rmd_telemetry_proofs(cpu, mismatch).code ==
                                   RmdTelemetryCheckCode::correction_hash_mismatch,
                                   "correction hash mismatch rejected");
    mismatch = ws; mismatch.stripes[0].correction_nonzero_count = 1;
    const bool nonzero = expect(compare_rmd_telemetry_proofs(cpu, mismatch).code ==
                                  RmdTelemetryCheckCode::correction_nonzero_count_mismatch,
                                  "correction nonzero count mismatch rejected");
    mismatch = ws; mismatch.stripes[0].output_hash[0] ^= 1;
    const bool output = expect(compare_rmd_telemetry_proofs(cpu, mismatch).code ==
                               RmdTelemetryCheckCode::output_hash_mismatch,
                               "output hash mismatch rejected");
    return input && correction && nonzero && output;
}
}
int main(int argc, char ** argv) {
    if (argc == 3 && std::string(argv[1]) == "--aggregate-driver") {
        return run_aggregate_driver(argv[2]) ? 0 : 1;
    }
    if (argc != 1) return 2;
    if (!aggregate_serializer_fixtures() || !aggregate_cycle_sink_fixtures() ||
        !residual_capture_timer_seam()) return 1;
    if (!expect(resolve_rmd_model_id("model-id-env", "model-arch") == "model-id-env",
                "model ID environment value wins") ||
        !expect(resolve_rmd_model_id("", "model-arch").empty(),
                "set empty model ID does not fall back") ||
        !expect(resolve_rmd_model_id(nullptr, "model-arch") == "model-arch",
                "unset model ID falls back to model architecture")) return 1;
    cycle::reset_read_count_for_test(); (void) cycle::read();
#if !LOG_CYCLE
    const std::string json = serialize_cycle_telemetry(cpu_record());
    return !(expect(cycle::read_count_for_test() == 0, "OFF performs zero clock reads") &&
             expect(json.empty(), "OFF emits no cycle or detail JSON keys") && negative_fixtures());
#else
    const uint64_t first = cycle::read(); const uint64_t second = cycle::read();
    if (!expect(second >= first && cycle::read_count_for_test() == 3, "enabled clock reads are observable")) return 1;
    RmdTelemetryRecord cpu = cpu_record(); RmdTelemetryRecord ws = ws_record();
    const std::string expected_rmd_summary =
        "{\"schema\":\"gemmini.cycle\",\"version\":1,\"record_type\":\"RMD_BACKEND_TELEMETRY\","
        "\"runtime_bundle_id\":\"bundle-7\",\"model_id\":\"model-hash\",\"run_id\":\"run-42\","
        "\"source\":\"host_tick\",\"unit\":\"tick\",\"backend\":\"cpu_direct\",\"option_source\":\"explicit_override\","
        "\"work\":true,\"invocation_total\":101,\"dispatch\":{\"direct_events\":9,\"direct_calls\":2,\"packet_calls\":0,\"ws_calls\":0},"
        "\"timing\":{\"prep\":11,\"backend_service\":31,\"merge\":7,\"residual_total\":47,\"queue\":3,\"dense_end\":120,\"residual_start\":120},"
        "\"geometry\":{\"packet_count\":0,\"active_blocks\":0,\"compact_k_count\":0,\"padded_k_count\":0,\"physical_tile_count\":0}}";
#if CYCLE_DETAIL
    cpu.stripes = {
        {0,0,4,{10,20,20,26,27,31,31,38},"0123456789abcdef","1123456789abcdef","2123456789abcdef",1},
        {1,4,8,{40,50,50,59,60,65,65,72},"3123456789abcdef","4123456789abcdef","5123456789abcdef",0},
    };
    ws.stripes = cpu.stripes;
    const std::string json = serialize_cycle_telemetry(cpu);
    const std::string expected_rmd = expected_rmd_summary.substr(0, expected_rmd_summary.size() - 1) +
        ",\"stripes\":[{\"stripe_id\":0,\"row_begin\":0,\"row_end\":4,\"stages\":{\"dense_start\":10,\"dense_end\":20,"
        "\"residual_start\":20,\"backend_start\":26,\"backend_end\":27,\"merge_start\":31,\"merge_end\":31,\"residual_end\":38},"
        "\"input_hash\":\"0123456789abcdef\",\"correction_hash\":\"1123456789abcdef\",\"correction_nonzero_count\":1,"
        "\"output_hash\":\"2123456789abcdef\"},{\"stripe_id\":1,\"row_begin\":4,\"row_end\":8,\"stages\":{\"dense_start\":40,"
        "\"dense_end\":50,\"residual_start\":50,\"backend_start\":59,\"backend_end\":60,\"merge_start\":65,\"merge_end\":65,"
        "\"residual_end\":72},\"input_hash\":\"3123456789abcdef\",\"correction_hash\":\"4123456789abcdef\","
        "\"correction_nonzero_count\":0,\"output_hash\":\"5123456789abcdef\"}]}";
    const bool detail = expect(json == expected_rmd, "RMD detail serialized bytes remain exact baseline") &&
        expect(json.find("\"stripes\"") != std::string::npos, "DETAIL emits per-stripe attribution") &&
        expect(json.find("input_hash") != std::string::npos &&
               json.find("correction_hash") != std::string::npos &&
               json.find("\"correction_nonzero_count\":1") != std::string::npos &&
               json.find("\"correction_nonzero_count\":0") != std::string::npos &&
               json.find("output_hash") != std::string::npos,
               "DETAIL emits exact correction nonzero counts") &&
        expect(check_rmd_telemetry(cpu, cycle::units(), true).ok(), "ordered CPU detail accepted") &&
        expect(check_rmd_telemetry(ws, cycle::units(), true).ok(), "route-exclusive WS detail accepted") &&
        hash_parity_and_mismatch_fixtures();
#else
    const std::string json = serialize_cycle_telemetry(cpu);
    const bool detail = expect(json == expected_rmd_summary, "RMD summary serialized bytes remain exact baseline") &&
        expect(json.find("\"stripes\"") == std::string::npos, "SUMMARY has no per-stripe detail") &&
        expect(json.find("input_hash") == std::string::npos &&
               json.find("correction_hash") == std::string::npos &&
               json.find("output_hash") == std::string::npos &&
               json.find("correction_nonzero_count") == std::string::npos,
               "SUMMARY has no proof hashes or nonzero counts") &&
        expect(check_rmd_telemetry(cpu, cycle::units(), true).ok(), "route-exclusive CPU summary accepted") &&
        expect(check_rmd_telemetry(ws, cycle::units(), true).ok(), "route-exclusive WS summary accepted");
#endif
    if (std::getenv("GEMMINI_TELEMETRY_PRINT") != nullptr) std::printf("%s\n", json.c_str());
    const size_t total = json.find("\"invocation_total\"");
    const bool once = expect(total != std::string::npos && json.find("\"invocation_total\"", total + 1) == std::string::npos,
                             "exactly one invocation total is serialized");
#ifdef __riscv
    const char * expected_source = "\"source\":\"riscv_cycle\"";
    const char * expected_unit = "\"unit\":\"cycle\"";
#else
    const char * expected_source = "\"source\":\"host_tick\"";
    const char * expected_unit = "\"unit\":\"tick\"";
#endif
    const bool fields = expect(json.find("\"schema\":\"gemmini.cycle\"") != std::string::npos &&
                               json.find("\"record_type\":\"RMD_BACKEND_TELEMETRY\"") != std::string::npos &&
                               json.find(expected_source) != std::string::npos &&
                               json.find(expected_unit) != std::string::npos,
                               "RMD common discriminator, source, and unit serialized") &&
        expect(json.find("\"runtime_bundle_id\":\"bundle-7\"") != std::string::npos,
               "runtime bundle identifier serialized") &&
        expect(json.find("\"direct_calls\":2") != std::string::npos && json.find("\"packet_calls\":0") != std::string::npos &&
               json.find("\"ws_calls\":0") != std::string::npos, "CPU dispatch counters are exclusive") &&
        expect(json.find("\"tiles\"") == std::string::npos && json.find("stripes=") == std::string::npos,
               "ambiguous bare tiles and stripes progress fields are absent");
    return !(detail && once && fields && negative_fixtures());
#endif
}
