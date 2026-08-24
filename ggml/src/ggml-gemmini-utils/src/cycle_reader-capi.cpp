#include "../include/gemmini/cycle_reader.hpp"
#include "../include/gemmini/cycle_reader.h"

extern "C" uint64_t gemmini_read_cycles(void) noexcept
{
    try { return ggml::gemmini::cycle::read(); }
    catch (...) { return 0; }
}
