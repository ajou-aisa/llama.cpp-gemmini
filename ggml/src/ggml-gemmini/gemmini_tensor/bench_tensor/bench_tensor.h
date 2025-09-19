#pragma once

#include "../../ggml-gemmini-util.h"
#include <cstdint>
#include <cstddef>
#include <type_traits>
#include <cstdlib>
#include <string>

#include "ggml.h"

namespace aisa
{
    template <typename T>
    class BenchTensor
    {
    public:
        static BenchTensor<T> *getOrCreate(ggml_backend_gemmini_context *ctx,
                                           const ggml_tensor *src,
                                           const char *suffix = ".bench",
                                           bool acc = false,
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
        BenchTensor(const ggml_tensor *src,
                    const char *suffix = ".bench",
                    bool acc = false,
                    bool transpose = false);
        void freeBuffer();

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

