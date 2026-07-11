#pragma once

#include "ggml-gemmini-q8-h1-artifact.hpp"

namespace ggml::gemmini::detail {

bool dims_to_geometry(
        const std::array<int64_t, GGML_MAX_DIMS> & dims,
        size_t & logical_rows,
        size_t & k,
        size_t & blocks_per_row);

bool load_q8_h1_artifact_impl(
        const std::string & path,
        q8_h1_artifact_store & store,
        std::string * error);

}
