#pragma once
#include "../bench_tensor/bench_tensor.h"
#include "ggml.h"
#include "vector"

namespace aisa
{
    // Dynamic Error Compensation for activations (input-side), Top-K based
    class ActivationDEC
    {
    public:
        void computeResidual();
        void computeCompensation(int8_t *W, size_t J, float *y_out);
        static void compensate(const ggml_tensor *A,
                        const BenchTensor<int8_t> *qA,
                        BenchTensor<int8_t> *W,
                        ggml_tensor *C_out);

        size_t K() const { return alpha_; }
        const std::vector<int> &indices() const { return S_; }

    private:
        const ggml_tensor *A_;          // F32 activation
        const BenchTensor<int8_t> *qA_; // quantized activation

        size_t alpha_;             // number of salient input channels
        std::vector<int> S_;       // outlier 인덱스 집합
        std::vector<float> delta_; // residual
        std::vector<std::pair<float, size_t>> sorted_;

        ActivationDEC(const ggml_tensor *A, const BenchTensor<int8_t> *qA, double ratio = 0.05);
    };
}

#include "activation_DEC.tpp"