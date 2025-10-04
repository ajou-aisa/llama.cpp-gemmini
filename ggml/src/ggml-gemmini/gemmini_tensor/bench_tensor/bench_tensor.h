// gemmini_tensor/bench_tensor/bench_tensor.h
#pragma once

#include <cstdint>
#include <cstddef>
#include <type_traits>
#include <cstdlib>
#include <string>

#include "ggml.h"

#define SCALE 0.002441f

// Forward declarations
struct ggml_backend_gemmini_context;

namespace aisa
{
    template <typename T>
    class BenchTensor
    {
    public:
        /* Weight용: 완전 캐싱 (값 포함)
         * 포인터 + dimension + transpose 기반 캐싱
         * Weight는 고정값(0-fill)이므로 한번 생성 후 계속 재사용 */
        static BenchTensor<T> *getOrCreate(ggml_backend_gemmini_context *ctx,
                                           const char *layer,  
                                           const ggml_tensor *src,
                                           const char *suffix = ".bench",
                                           bool acc = false,
                                           bool transpose = false);

        /* Activation/Output용: 버퍼만 재사용
         * dimension 기반으로 버퍼 풀에서 재사용
         * 값은 매번 quantize(Activation) 또는 Gemmini 연산(Output)으로 갱신 */
        static BenchTensor<T> *getOrCreateTransient(ggml_backend_gemmini_context *ctx,
                                                    const char *layer,
                                                    const ggml_tensor *src,
                                                    const char *suffix = ".transient",
                                                    bool transpose = false);

        // 이동 전용 구현
        BenchTensor(BenchTensor &&) noexcept;            // 이동 생성자
        BenchTensor &operator=(BenchTensor &&) noexcept; // 이동 대입

        BenchTensor(const BenchTensor &) = delete;            // 복사 생성자 금지
        BenchTensor &operator=(const BenchTensor &) = delete; // 복사 대입 금지

        // Gemmini 커널용 데이터 버퍼
        void *get() noexcept { return data_; }
        const void *get() const noexcept { return data_; }

        // dimension 접근
        size_t getRows() const noexcept { return rows_; }
        size_t getCols() const noexcept { return cols_; }

        // stride 접근
        size_t getStride() const noexcept { return stride_; }

        // 소멸자 & 버퍼 해제
        ~BenchTensor() { freeBuffer(); }

    private:
        BenchTensor(const char* layer,
                    const ggml_tensor *src,
                    const char *suffix = ".bench",
                    bool acc = false,
                    bool transpose = false);
        void freeBuffer();

        void quantizeActivation(const ggml_tensor *src);

        std::string name_;
        const ggml_type type_;

        void *data_ = nullptr; // data 버퍼
        size_t buf_bytes_ = 0;          // 할당된 바이트 수
        
        int64_t rows_ = 0;
        int64_t cols_ = 0;
        int64_t stride_ = 0; // stride in elements
    };
}

#include "bench_tensor.tpp"

