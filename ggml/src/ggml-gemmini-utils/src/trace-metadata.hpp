#pragma once

#include <gemmini/performance.hpp>
#include <gemmini/cpu-timing.h>
#include <gemmini/host-timing.hpp>
#include "../../../../common/json.hpp"

namespace ggml::gemmini::trace {
nlohmann::json metadata_fields(const nlohmann::json &        row_facts,
                               const gemmini_trace_context & context,
                               uint64_t                      segment_id,
                               bool                          structural_envelope = false);
nlohmann::json origin_fields(const nlohmann::json &        row_facts,
                             const gemmini_trace_context & context,
                             uint64_t                      segment_id,
                             bool                          structural_envelope = false,
                             bool                          explicit_empty      = false);
std::string    metadata_suffix(const nlohmann::json & fields, bool compact = false);
nlohmann::json
host_timing_facts(uint64_t start_ns, uint64_t end_ns, uint64_t start_tid, uint64_t end_tid);
nlohmann::json cpu_timing_facts(const gemmini_cpu_sample & start, const gemmini_cpu_sample & end);
} // namespace ggml::gemmini::trace

namespace ggml::gemmini::performance {
std::string serialize_measurement(const Measurement &    measurement,
                                  const nlohmann::json & metadata_fields);
}

namespace ggml::gemmini::log {
enum class ProfileRowKind { timeline, stage, workload };
struct ProfileRow {
    std::string       body;
    ProfileRowKind    kind;
    const char *      source;
    cycle::HostSample start{}, end{};
};
struct ProfileRows {
    std::vector<ProfileRow> rows;
    bool                    reserve_additional = false;
};
struct CycleRecord;
struct WsCycleRecord;
std::string serialize_cycle_record(const CycleRecord & record, const std::string & tail);
std::string serialize_cycle_record(const CycleRecord &           record,
                                   const gemmini_trace_context & origin,
                                   uint64_t                      segment_id);
std::string serialize_checked_cycle_record(const CycleRecord & record,
                                           bool                valid,
                                           const char *        reason,
                                           const char *        sample_reason,
                                           const std::string & tail);
std::string serialize_ws_cycle_record(const WsCycleRecord & record, const std::string & tail);
} // namespace ggml::gemmini::log
