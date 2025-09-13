#include "include/gemmini.h"
#include "gemmini_bench_tensor.h"

namespace zerogod
{
    template <typename T>
    BenchTensor<T>::BenchTensor(ggml_context *ctx,
                              const ggml_tensor *src,
                              const char *suffix,
                              bool acc,
                              bool transpose)
    {
        /* 1. ____________________원본 행/열____________________
              ggml 네이티브: ne[0] = columns(X), ne[1] = rows(Y) */
        const int64_t cols_ = transpose ? src->ne[1] : src->ne[0];
        const int64_t rows_ = transpose ? src->ne[0] : src->ne[1];
        ggml_type type_ = src->type;

        /* 3. ___________________tensor 이름___________________ */
        snprintf(name_, sizeof(src->name), "%s%s", src->name, suffix);


        /* 4. __________________buffer 할당____________________ */
        const size_t elem_size = sizeof(T);
        const size_t row_bytes = cols_ * elem_size;
        buf_bytes_ = row_bytes * rows_;

        if (buf_bytes_ == 0)
            buf_bytes_ = GEMMINI_ALIGN; // 최소 16 B 확보

        data_ = std::aligned_alloc(GEMMINI_ALIGN, buf_bytes_); // buffer을 16B 경계에 할당
        GGML_ASSERT(this->data_ != nullptr);

        stride_ = row_bytes / elem_size;

        /* 5. _______________ 0-fill _________________ */
            std::memset(data_, 0, buf_bytes_);
    }

    template <typename T>
    void BenchTensor<T>::freeBuffer()
    {
        if (data_)
        {
            std::free(data_);
            data_ = nullptr;
        }
        tensor_ = nullptr;
        buf_bytes_ = 0;
    }

    // 이동 생성자 & 이동 대입 연산자 오버라이딩
    // other: 기존 객체
    template <typename T>
    BenchTensor<T>::BenchTensor(BenchTensor &&other) noexcept
        : name_{other.name_}, type_{other.type_}, 
          data_{other.data_}, buf_bytes_(other.buf_bytes_), 
          rows_(other.rows_), cols_(other.cols_), stride_(other.stride_)
    {
        other.name_ = "";
        other.type_ = "";
        
        other.data_ = nullptr;
        other.buf_bytes_ = 0;
        other.rows_ = other.cols_ = other.stride_ = 0;
    }

    template <typename T>
    BenchTensor<T> &
    BenchTensor<T>::operator=(BenchTensor &&other) noexcept
    {
        if (this != &other)
        {
            freeBuffer();
            name_ = other.name_;
            type_ = other.type_;
            data_ = other.data_;
            buf_bytes_ = other.buf_bytes_;
            rows_ = other.rows_;
            cols_ = other.cols_;
            stride_ = other.stride_;

            other.name_ = "";
            other.type_ = "";
            other.data_ = nullptr;
            other.buf_bytes_ = 0;
            other.rows_ = other.cols_ = other.stride_ = 0;
        }
        return *this;
    }
}