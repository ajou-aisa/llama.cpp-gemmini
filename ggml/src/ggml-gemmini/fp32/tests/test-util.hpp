#pragma once
#include "../exec.hpp"
#include "golden.hpp"
#include "ggml-gemmini.h"
#include "../../ggml-gemmini-buffer.hpp"
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

namespace fp32 = ggml::gemmini::fp32;
inline ggml_tensor tensor(ggml_type type, void * data, size_t k, size_t rows, size_t heads = 1) {
    ggml_tensor t{};
    t.type = type; t.data = data;
    t.ne[0] = int64_t(k); t.ne[1] = int64_t(rows); t.ne[2] = int64_t(heads); t.ne[3] = 1;
    t.nb[0] = ggml_type_size(type);
    for (size_t i = 1; i < 4; ++i) t.nb[i] = t.nb[i - 1] * size_t(t.ne[i - 1]);
    return t;
}
inline void start_test_runtime(int argc, char ** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    auto * init = ggml_init({1024 * 1024, nullptr, true});
    golden_fp32::check(init != nullptr, "ggml initialization");
    ggml_free(init);
    if (argc > 1) fp32::set_order(fp32::Order(std::stoul(argv[1])));
    if (argc > 3) {
        std::vector<int> lanes;
        std::istringstream in(argv[3]);
        for (std::string s; std::getline(in, s, ',');) lanes.push_back(std::stoi(s));
        fp32::start_runtime(std::stoi(argv[2]), lanes);
    }
}
inline void finish_test_runtime() {
    const auto r = fp32::runtime_report();
    for (size_t i = 0; i < r.size(); ++i) {
        std::printf("FP32_THREAD lane=%zu cpu=%d jobs=%llu elements=%llu mismatch=%llu\n", i, r[i].cpu,
                    (unsigned long long)r[i].jobs, (unsigned long long)r[i].elements, (unsigned long long)r[i].mismatch);
        golden_fp32::check(r[i].jobs > 0 && r[i].mismatch == 0, "five-core work/affinity");
    }
    fp32::stop_runtime();
}
