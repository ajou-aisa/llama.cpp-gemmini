#pragma once

#include <cstdio>

#include "dump_tensor.hpp"

namespace ggml::gemmini::log::dump_detail
{
    const char *dump_phase_to_string(DumpPhase phase);
    void write_json_escaped(FILE *out, const char *s);
}
