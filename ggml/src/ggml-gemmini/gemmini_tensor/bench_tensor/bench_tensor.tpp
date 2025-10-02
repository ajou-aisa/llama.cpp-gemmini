// gemmini_tensor/bench_tensor/bench_tensor.tpp
#include "include/gemmini.h"
#include "bench_tensor.h"
#include "../tensor_cache_key.h"
#include "../transient_key.h"
#include <memory>
#include <cstring>

namespace aisa
{
    // Weight용: 완전 캐싱
    template <typename T>
    BenchTensor<T> *BenchTensor<T>::getOrCreate(ggml_backend_gemmini_context *ctx,
                                                const ggml_tensor *src,
                                                const char *suffix,
                                                bool acc,
                                                bool transpose)
    {
        // 캐시 키 생성 (포인터, dimension, transpose, acc)
        TensorCacheKey key{src, src->ne[0], src->ne[1], transpose, acc};

        // weight_cache 사용
        auto it = ctx->weight_cache.find(key);
        if (it != ctx->weight_cache.end())
        {
#if DEBUG
            DBG0("[WEIGHT CACHE HIT] ptr=%p, ne=[%lld,%lld]\n",
                 (void *)src, (long long)src->ne[0], (long long)src->ne[1]);
#endif
            return it->second.get();
        }

#if DEBUG
        DBG0("[WEIGHT CACHE MISS] Creating: ptr=%p, ne=[%lld,%lld]\n",
             (void *)src, (long long)src->ne[0], (long long)src->ne[1]);
#endif

        auto new_tensor = std::unique_ptr<BenchTensor<T>>(
            new BenchTensor<T>(src, suffix, acc, transpose));
        BenchTensor<T> *ptr = new_tensor.get();
        ctx->weight_cache[key] = std::move(new_tensor);
        return ptr;
    }

    // Activation/Output용: 버퍼만 재사용
    template <typename T>
    BenchTensor<T> *BenchTensor<T>::getOrCreateTransient(ggml_backend_gemmini_context *ctx,
                                                         const ggml_tensor *src,
                                                         const char *suffix,
                                                         bool transpose)
    {
        int64_t rows = transpose ? src->ne[0] : src->ne[1];
        int64_t cols = transpose ? src->ne[1] : src->ne[0];
        TransientKey key{rows, cols};

        auto it = ctx->transient_pool.find(key);
        if (it != ctx->transient_pool.end())
        {
#if DEBUG
            DBG0("[TRANSIENT POOL HIT] dims=%lldx%lld\n",
                 (long long)rows, (long long)cols);
#endif
            return it->second.get();
        }

#if DEBUG
        DBG0("[TRANSIENT POOL MISS] Creating buffer: %lldx%lld\n",
             (long long)rows, (long long)cols);
#endif

        auto new_tensor = std::unique_ptr<BenchTensor<T>>(
            new BenchTensor<T>(src, suffix, false, transpose));
        BenchTensor<T> *ptr = new_tensor.get();
        ctx->transient_pool[key] = std::move(new_tensor);
        return ptr;
    }

    template <typename T>
    BenchTensor<T>::BenchTensor(const ggml_tensor *src,
                                const char *suffix,
                                bool acc,
                                bool transpose)
        : name_{std::string(src->name) + (suffix ? suffix : "")}, type_{src->type}
    {
        /* 1. ____________________원본 행/열____________________
        ggml 네이티브: ne[0] = columns(X), ne[1] = rows(Y) */
        cols_ = transpose ? src->ne[1] : src->ne[0];
        rows_ = transpose ? src->ne[0] : src->ne[1];

        /* 4. __________________buffer 할당____________________ */
        const size_t elem_size = sizeof(T);
        const size_t row_bytes = static_cast<size_t>(cols_) * elem_size;
        buf_bytes_ = row_bytes * static_cast<size_t>(rows_);

        if (buf_bytes_ == 0)
            buf_bytes_ = GEMMINI_ALIGN; // 최소 16 B 확보

        this->data_ = std::aligned_alloc(GEMMINI_ALIGN, buf_bytes_); // buffer을 16B 경계에 할당
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
        buf_bytes_ = 0;
    }

    // 이동 생성자 & 이동 대입 연산자 오버라이딩
    // other: 기존 객체
    template <typename T>
    BenchTensor<T>::BenchTensor(BenchTensor &&other) noexcept
        : name_{std::move(other.name_)}, type_{other.type_},
          data_{other.data_}, buf_bytes_(other.buf_bytes_),
          rows_(other.rows_), cols_(other.cols_), stride_(other.stride_)
    {
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
            GGML_ASSERT(type_ == other.type_);

            freeBuffer();
            name_ = std::move(other.name_);
            data_ = other.data_;
            buf_bytes_ = other.buf_bytes_;
            rows_ = other.rows_;
            cols_ = other.cols_;
            stride_ = other.stride_;

            other.data_ = nullptr;
            other.buf_bytes_ = 0;
            other.rows_ = other.cols_ = other.stride_ = 0;
        }
        return *this;
    }
}