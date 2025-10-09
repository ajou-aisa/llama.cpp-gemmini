// gemmini_tensor/baseline_tensor/baseline_tensor.tpp
#include "include/gemmini.h"
#include "baseline_tensor.h"
#include "../dequantize_weight.h"
#include <memory>
#include <cstring>

#define DEQUANTIZE 1

namespace aisa
{
    template <typename T>
    BaselineTensor<T> *BaselineTensor<T>::getOrCreate(ggml_backend_gemmini_context *ctx,
                                                      const ggml_tensor *src,
                                                      const char *suffix,
                                                      bool acc,
                                                      bool transpose)
    {
        // 1. initiate
        auto new_tensor = std::unique_ptr<BaselineTensor<T>>(new BaselineTensor<T>(src, suffix, acc, transpose));
        BaselineTensor<T> *ptr = new_tensor.get();

        // 2. move
        ctx->temp_tensors.push_back(std::move(new_tensor));
        return ptr;
    }

    /* Static Polymorphism용 추가
     * BenchTensor와 인터페이스 통일을 위해 추가
     * BaselineTensor는 캐싱 구분이 불필요하므로 단순히 getOrCreate 호출 */
    template <typename T>
    BaselineTensor<T> *BaselineTensor<T>::getOrCreateTransient(
        ggml_backend_gemmini_context *ctx,
        const ggml_tensor *src,
        const char *suffix,
        bool transpose)
    {
        // BaselineTensor는 캐싱하지 않으므로 getOrCreate와 동일
        return getOrCreate(ctx, src, suffix, false, transpose);
    }

    template <typename T>
    BaselineTensor<T>::BaselineTensor(const ggml_tensor *src,
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
        size_t row_bytes = static_cast<size_t>(cols_) * elem_size;

        //more data allocation for DEQUANTIZE option.
        if (src->type == GGML_TYPE_Q8_0 && DEQUANTIZE)
            row_bytes = static_cast<size_t>(cols_) * elem_size *2;


        buf_bytes_ = row_bytes * static_cast<size_t>(rows_);

        if (buf_bytes_ == 0)
            buf_bytes_ = GEMMINI_ALIGN; // 최소 16 B 확보

        this->data_ = std::aligned_alloc(GEMMINI_ALIGN, buf_bytes_); // buffer을 16B 경계에 할당
        GGML_ASSERT(this->data_ != nullptr);

        stride_ = row_bytes / elem_size; // element 단위


        /* 5. _______________casting & 0-fill _________________ */


        if (acc)
            std::memset(data_, 0, buf_bytes_);
        else
            castBaselineData(src, transpose);

    }

    template <typename T>
    void BaselineTensor<T>::castBaselineData(const ggml_tensor *src,
                                             bool transpose) const
    {
        const size_t src_row_bytes = src->nb[1]; // 행 간 byte-stride
        const size_t src_col_bytes = src->nb[0]; // 열 간 byte-stride

        /* _____________________2. dst 정보______________________*/
        T *dst_base = static_cast<T *>(this->data_);
        const size_t dst_row_bytes = this->stride_ * sizeof(T);
        const size_t elem_size = sizeof(T);

        /* ___________________3. 원본 타입별 분기__________________*/
        switch (src->type)
        {
        case GGML_TYPE_F32:
        {
            const uint8_t *src_base = static_cast<const uint8_t *>(src->data);

            for (size_t r = 0; r < rows_; ++r)
            {
                T *dst_elem_row = dst_base + r * stride_;
                if (!transpose)
                    // src 행 r 를 그대로 복사 : 주소 = base + r*src_row_bytes + c*src_col_bytes
                    for (size_t c = 0; c < cols_; ++c)
                    {
                        const float *p = reinterpret_cast<const float *>(src_base + r * src_row_bytes + c * src_col_bytes);
                        dst_elem_row[c] = static_cast<T>(*p);
                    }
                else
                    // 전치 복사 : src( c , r ) -> dst( r , c )
                    for (size_t c = 0; c < cols_; ++c)
                    {
                        const float *p = reinterpret_cast<const float *>(src_base + c * src_row_bytes + r * src_col_bytes);
                        dst_elem_row[c] = static_cast<T>(*p);
                    }
            }
            break;
        }
        case GGML_TYPE_Q8_0:
        {
            uint64_t start, end;

            start = read_cycles();
            dequantizingWithGgml(src);
            end = read_cycles();
            printf("[dequantizing cycles] start = %lu, end = %lu, elapsed = %lu\n", start, end, end - start);

            break;
        }
        default:
        {
            GGML_ASSERT(false && "castBaselineData: unsupported src type");
            break;
        }
        }
    }

    template <typename T>
    void BaselineTensor<T>::freeBuffer()
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
    BaselineTensor<T>::BaselineTensor(BaselineTensor &&other) noexcept
        : name_{std::move(other.name_)}, type_{other.type_},
          data_{other.data_}, buf_bytes_(other.buf_bytes_),
          rows_(other.rows_), cols_(other.cols_), stride_(other.stride_)
    {
        other.data_ = nullptr;
        other.buf_bytes_ = 0;
        other.rows_ = other.cols_ = other.stride_ = 0;
    }

    template <typename T>
    BaselineTensor<T> &
    BaselineTensor<T>::operator=(BaselineTensor &&other) noexcept
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