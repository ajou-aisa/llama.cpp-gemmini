#include "../include/gemmini/cycle_reader.hpp"
#include "../include/gemmini/cycle_reader.h"

extern "C" uint64_t gemmini_read_cycles(void)
{
    return ggml::gemmini::cycle::read();
}
