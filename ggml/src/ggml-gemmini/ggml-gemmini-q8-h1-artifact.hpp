#pragma once

#include "ggml.h"

#include <array>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace ggml::gemmini {

struct q8_h1_artifact_tensor {
    std::array<int64_t, GGML_MAX_DIMS> dims{};
    size_t logical_rows = 0;
    size_t k = 0;
    size_t blocks_per_row = 0;
    std::vector<int8_t> qs;
    std::vector<uint8_t> subs;
    std::vector<float> sups_f32;
    std::vector<uint16_t> z;
};

using q8_h1_artifact_store = std::map<std::string, q8_h1_artifact_tensor>;

enum class q8_h1_artifact_runtime_status : uint8_t {
    disabled_no_env = 0,
    ready = 1,
    disabled_load_error = 2,
};

enum class q8_h1_artifact_lookup_status : uint8_t {
    disabled = 0,
    ready = 1,
    missing_tensor = 2,
    shape_mismatch = 3,
};

struct q8_h1_artifact_runtime_state {
    q8_h1_artifact_runtime_status status = q8_h1_artifact_runtime_status::disabled_no_env;
    std::string path;
    std::string error;
    q8_h1_artifact_store store;
};

bool load_q8_h1_artifact(
        const std::string & path,
        q8_h1_artifact_store & store,
        std::string * error = nullptr);

void load_q8_h1_artifact_from_env(q8_h1_artifact_runtime_state & state);

q8_h1_artifact_lookup_status lookup_q8_h1_artifact_tensor(
        const q8_h1_artifact_runtime_state & state,
        const char * tensor_name,
        const std::array<int64_t, GGML_MAX_DIMS> & dims,
        const q8_h1_artifact_tensor ** tensor = nullptr);

}
