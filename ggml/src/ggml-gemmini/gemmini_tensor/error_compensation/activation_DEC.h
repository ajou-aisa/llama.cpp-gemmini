#pragma once
#include "../bench_tensor/bench_tensor.h"
#include "../../labeling/label.h"
#include "ggml.h"
#include <vector>
#define SCALE_W 1.0f // 임시
#ifndef DEC_ALPHA_RATIO
#define DEC_ALPHA_RATIO 0.05    // 5% 
#endif

namespace aisa
{
    // Dynamic Error Compensation for activations (input-side), Top-K based
    class ActivationDEC
    {
    public:
        static void compensate(const ggml_tensor *A,
                        const BenchTensor<int8_t> *qA,
                        const BenchTensor<int8_t> *qW,
                        ggml_tensor *C_out);

    private:
        const ggml_tensor *A_;          // F32 activation
        const BenchTensor<int8_t> *qA_; // quantized activation
        const BenchTensor<int8_t> *qW_; // quantized weight

        const char* layer_ = "others"; // layer
        size_t I_;  // number of rows  
        size_t K_;  // columns per row
        size_t J_;  // number of output columns
        size_t alpha_;             // number of salient input channels

        std::vector<std::vector<int>> S_;       // row 별 outlier 인덱스 집합
        std::vector<std::vector<float>> delta_; // row 별 residual
        std::vector<std::pair<float, size_t>> sorted_;

        ActivationDEC(const ggml_tensor *A, 
                      const BenchTensor<int8_t> *qA, 
                      const BenchTensor<int8_t> *qW) : A_(A), qA_(qA), qW_(qW){}
        void prepare();
        void selectTopKandComputeResidual(size_t row_idx, const float *row_fp, const int8_t *row_q);
        void computeCompensation(const int8_t *W, size_t J, float *y_out);
        void computeCompensation_CSC(const int8_t *W, size_t J, float *y_com);
        void applyCompensation(ggml_tensor *C_out, const std::vector<float> &y_com);
    };
}

#include "activation_DEC.tpp"