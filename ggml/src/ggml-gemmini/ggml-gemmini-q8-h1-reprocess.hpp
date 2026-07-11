#pragma once

#include "ggml-gemmini-args.h"

#include <cstddef>
#include <vector>

namespace ggml::gemmini {

bool prepare_q8_0_rows_for_q8_h1(
    const ggml_tensor * src,
    std::vector<block_q8_h1> & dst,
    size_t * blocks_per_row,
    size_t * logical_rows);

}
