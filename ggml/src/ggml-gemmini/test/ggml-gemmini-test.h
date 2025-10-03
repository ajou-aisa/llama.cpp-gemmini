// test/ggml-gemmini-test.h
#pragma once

#include "../ggml-gemmini-util.h"
#include "../gemmini_tensor/gemmini_tensor_interface.h"
#include <vector>

#ifndef OPTION
#define OPTION CPU
#endif
// 1: src0(JxK) -> 변환 시 물리 전치(KxJ)로 배치(현재 기본)
// 0: src0(JxK) -> 전치 없이 JxK로 배치
#ifndef TRANSPOSE_B
#define TRANSPOSE_B 1
#endif
#ifndef TEST_SHAPE
#define TEST_SHAPE 0
#endif
#ifndef TEST_SORT
#define TEST_SORT 0
#endif
#ifndef TEST_TYPE
#define TEST_TYPE 0
#endif
#ifndef TEST_SLICE
#define TEST_SLICE 0
#endif
#ifndef TEST_CPU_REF
#define TEST_CPU_REF 0
#endif
#ifndef TEST_GEMMINI
#define TEST_GEMMINI 0
#endif
#ifndef TEST_COMPARE
#define TEST_COMPARE 0
#endif
#ifndef TEST_DEQUANTIZE_OUT
#define TEST_DEQUANTIZE_OUT 0
#endif
#ifndef DUMP
#define DUMP (TEST_SLICE || TEST_CPU_REF || TEST_COMPARE)
#endif

// slice 크기
#ifndef SLICE_I
#define SLICE_I 1 // A/C는 행 1만 출력
#endif
#ifndef SLICE_K
#define SLICE_K 4 // K 방향 최대 출력 열
#endif
#ifndef SLICE_J
#define SLICE_J 6 // J 방향 최대 출력 열(논리 J 기준)
#endif

namespace aisa
{
    // 1. 테스트 설정을 위한 구조체
    struct TestConfig
    {
        bool test_shape = TEST_SHAPE;
        bool test_slice = TEST_SLICE;
        bool test_sort = TEST_SORT;
        bool run_cpu_ref = TEST_CPU_REF;
        bool run_gemmini = TEST_GEMMINI;
        bool compare_results = TEST_CPU_REF && TEST_GEMMINI && TEST_COMPARE;
        bool dequantize_output = TEST_GEMMINI &&TEST_DEQUANTIZE_OUT;
    };

    // 2. 테스트 로직을 캡슐화하는 클래스
    class GemminiTestbench
    {
    public:
        GemminiTestbench(ggml_backend_gemmini_context *ctx, ggml_tensor *dst, const TestConfig &config);
        void run();

    private:
        // 각 테스트 단계를 위한 private 멤버 함수들
        void setUpDimensions();
        void debugShapes();
        void createTensors();
        void dumpTensorSlices();
        void prepareBias();
        void runCpuReference();
        void runGemminiComputation();
        void compareAndReport();
        void dequantizeAndFinalize();
        void sortActivation();


        // 멤버 변수들
        ggml_backend_gemmini_context *ctx_;
        ggml_tensor *dst_;
        const ggml_tensor *src0_;
        const ggml_tensor *src1_;
        const TestConfig &config_;

        const aisa::GemminiTensor<int8_t> *tA_ = nullptr;
        const aisa::GemminiTensor<int8_t> *tB_ = nullptr;
        const aisa::GemminiTensor<int8_t> *tC_ = nullptr;

        // Bias, CPU 참조 결과 등 테스트 중에 필요한 상태 변수들
        std::vector<int8_t> cpu_ref_c_;
        const int32_t *bias_data_ = nullptr;

        // 차원 정보
        int I_ = 0, J_ = 0, K_ = 0;
    };

    // 3. 외부에서 호출하는 유일한 진입점 함수
    void ggml_gemmini_test(ggml_backend_gemmini_context *ctx, struct ggml_tensor *dst);

    // utils
    static inline int8_t saturationToInt8(int x);

    template <typename T>
    static inline void dumpMatrix(const char *name, const T *m, int r, int c, int s);
}