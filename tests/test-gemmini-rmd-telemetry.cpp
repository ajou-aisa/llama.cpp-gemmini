#include "../ggml/src/ggml-gemmini/ggml-gemmini-matmul.hpp"
#include <gemmini/cycle_reader.hpp>
#include "../ggml/src/ggml-gemmini/residual/rmd/rmd-builder.hpp"
#include <cstdio>
#include <string>
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
int main() {
    if (!expect(resolve_rmd_model_id("model-id-env", "model-arch") == "model-id-env",
                "model ID environment value wins") ||
        !expect(resolve_rmd_model_id("", "model-arch").empty(),
                "set empty model ID does not fall back") ||
        !expect(resolve_rmd_model_id(nullptr, "model-arch") == "model-arch",
                "unset model ID falls back to model architecture")) return 1;
    cycle::reset_read_count_for_test(); (void) cycle::read();
#if !LOG_CYCLE
    const std::string json = serialize_rmd_telemetry(cpu_record());
    return !(expect(cycle::read_count_for_test() == 0, "OFF performs zero clock reads") &&
             expect(json.empty(), "OFF emits no cycle or detail JSON keys") && negative_fixtures());
#else
    const uint64_t first = cycle::read(); const uint64_t second = cycle::read();
    if (!expect(second >= first && cycle::read_count_for_test() == 3, "enabled clock reads are observable")) return 1;
    RmdTelemetryRecord cpu = cpu_record(); RmdTelemetryRecord ws = ws_record();
#if CYCLE_DETAIL
    cpu.stripes = {
        {0,0,4,{10,20,20,26,27,31,31,38},"0123456789abcdef","1123456789abcdef","2123456789abcdef",1},
        {1,4,8,{40,50,50,59,60,65,65,72},"3123456789abcdef","4123456789abcdef","5123456789abcdef",0},
    };
    ws.stripes = cpu.stripes;
    const std::string json = serialize_rmd_telemetry(cpu);
    const bool detail = expect(json.find("\"stripes\"") != std::string::npos, "DETAIL emits per-stripe attribution") &&
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
    const std::string json = serialize_rmd_telemetry(cpu);
    const bool detail = expect(json.find("\"stripes\"") == std::string::npos, "SUMMARY has no per-stripe detail") &&
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
    const bool fields = expect(json.find("\"runtime_bundle_id\":\"bundle-7\"") != std::string::npos,
                               "runtime bundle identifier serialized") &&
        expect(json.find("\"direct_calls\":2") != std::string::npos && json.find("\"packet_calls\":0") != std::string::npos &&
               json.find("\"ws_calls\":0") != std::string::npos, "CPU dispatch counters are exclusive");
    return !(detail && once && fields && negative_fixtures());
#endif
}
