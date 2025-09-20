// ggml-gemmini-util.h
#pragma once

#ifndef DEBUG
#define DEBUG 0
#endif

#include "ggml-impl.h"
#include "ggml-gemmini.h"
#include "ggml-backend-impl.h"

#include <future>
#include <vector>
#include <map>
#include <memory>

#include "gemmini_tensor/baseline_tensor/baseline_tensor.h"
#include "gemmini_tensor/bench_tensor/bench_tensor.h"
#include "gemmini_tensor/gemmini_tensor_interface.h"

namespace aisa {
    constexpr size_t GEMMINI_ALIGN = 16; // 16-byte align
}

#include "gemmini_tensor/gemmini_tensor_interface.h"

struct ggml_backend_gemmini_context
{
    int n_threads = GGML_DEFAULT_N_THREADS;
    std::unique_ptr<char[]> work_data;
    size_t work_size = 0;
    std::map<ggml_tensor *, ggml_tensor *> bias_map;
    std::map<const ggml_tensor *, std::unique_ptr<aisa::BenchTensor<int8_t>>> tensor_cache;
    std::vector<std::unique_ptr<aisa::BaselineTensor<int8_t>>> temp_tensors;
#ifndef GGML_USE_OPENMP
    std::vector<std::future<void>> tasks;
#endif
};

#ifndef PRINT_TILE
#define PRINT_TILE 0
#endif

#if DEBUG 
    #define DBG(fmt, ...) \
        fprintf(stderr, "[%s:%d] %s(): " fmt "\n", __FILE__, __LINE__, __func__, ##__VA_ARGS__)
    // simple debug    
    #define DBG0(fmt, ...) \
        fprintf(stderr, fmt, ##__VA_ARGS__)
#else
    #define DBG(fmt, ...)  ((void)0)
    #define DBG0(fmt, ...) ((void)0)
#endif
