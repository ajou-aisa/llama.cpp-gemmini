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
        void computeCompensation(const int8_t *W, size_t J, float *y_out);
        static void compensate(const ggml_tensor *A,
                        const BenchTensor<int8_t> *qA,
                        const BenchTensor<int8_t> *W,
                        ggml_tensor *C_out);

        size_t K() const { return alpha_; }

    private:
        const ggml_tensor *A_;          // F32 activation
        const BenchTensor<int8_t> *qA_; // quantized activation

        size_t I_;  // number of rows
        size_t K_;  // columns per row
        size_t alpha_;             // number of salient input channels

        std::vector<std::vector<int>> S_;       // row 별 outlier 인덱스 집합
        std::vector<std::vector<float>> delta_; // row 별 residual
        std::vector<std::pair<float, size_t>> sorted_;

        ActivationDEC(const ggml_tensor *A, const BenchTensor<int8_t> *qA, double ratio = 0.05);
        void selectTopKForRow(size_t row_idx, const float* row_data, double ratio);
    };
}

#include "activation_DEC.tpp"