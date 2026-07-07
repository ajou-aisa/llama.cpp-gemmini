#pragma once

#include "ggml.h"

#include <cstdint>
#include <fstream>
#include <string>

class gemmini_q8_0_r_artifact_writer {
public:
    explicit gemmini_q8_0_r_artifact_writer(const char * path);

    void add_tensor(const std::string & name, const ggml_tensor * tensor, const void * data);
    void finish();

private:
    std::ofstream file;
    uint64_t tensor_count = 0;
};
