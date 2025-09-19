#pragma once

#if defined(GEMMINI_BENCH)
    // Target(Bench) Tensor 사용
    #include "bench_tensor/bench_tensor.h"
    namespace aisa {
        template<typename T>
        using GemminiTensor = BenchTensor<T>;
    }
#else
    // Baseline Tensor 사용
    #include "baseline_tensor/baseline_tensor.h" 
    namespace aisa {
        template<typename T>
        using GemminiTensor = BaselineTensor<T>;
    }
#endif