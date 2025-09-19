#include "include/gemmini.h"
#include "bench_tensor.h"
#include "dequantize_weight.h"
#include <memory>
#include <cstring>

namespace aisa
{
    template <typename T>
    BaselineTensor<T> *BaselineTensor<T>::getOrCreate(ggml_backend_gemmini_context *ctx,
                                                      const ggml_tensor *src,
                                                      const char *suffix = ".base",
                                                      bool acc = false,
                                                      bool transpose = false)
    {
        // 1. initiate
        auto new_tensor = std::unique_ptr<BaselineTensor<T>>(new BaselineTensor<T>(src, suffix, acc, transpose));
        BaselineTensor<T> *ptr = new_tensor.get();

        // 2. move
        ctx->temp_tensors.push_back(std::move(new_tensor));
        return ptr;
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
        const size_t row_bytes = static_cast<size_t>(cols_) * elem_size;
        buf_bytes_ = row_bytes * static_cast<size_t>(rows_);

        if (buf_bytes_ == 0)
            buf_bytes_ = GEMMINI_ALIGN; // 최소 16 B 확보

        this->data_ = std::aligned_alloc(GEMMINI_ALIGN, buf_bytes_); // buffer을 16B 경계에 할당
        GGML_ASSERT(this->data_ != nullptr);

        stride_ = row_bytes / elem_size; // element 단위

        /* 5. _______________casting & 0-fill _________________ */
        uint64_t start, end;
        start = read_cycles();
        if (acc)
            std::memset(data_, 0, buf_bytes_);
        else
            castBaselineData(src, transpose);
        end = read_cycles();

        if (!acc)
        {
            start = read_cycles();
            if (src->type == GGML_TYPE_Q8_0)
            {
                const int64_t rows = src_rows;
                const int64_t cols = src_cols;

                // 목적지: 패딩 반영된 행 스트라이드(요소 단위)
                T *dst_base = reinterpret_cast<T *>(this->data_);
                const size_t dst_stride_elems = this->stride_;
                if (!transpose)
                {
                    // src의 x축 방향(Q8_0 블록)이 그대로 열이며, 행 단위 직복사
                    q80_to_T_rowwise<T>(src, dst_base, dst_stride_elems, rows, cols);
                }
                else
                {
                    // 전치: dst(r,c) = src(x=r, y=c). 1원소 gather
                    q80_to_T_transposed<T>(src, dst_base, dst_stride_elems, rows, cols);
                }
                DBG("checking bp4\n");
                // 패딩 영역은 0으로 채우기(열 패딩분)
                if (padded_cols > src_cols)
                {
                    DBG0("padding...");
                    for (int64_t r = 0; r < rows; ++r)
                    {
                        T *rowp = dst_base + r * dst_stride_elems;
                        std::memset(rowp + src_cols, 0, (padded_cols - src_cols) * sizeof(T));
                    }
                }
            }
            else
                ggml_gemmini_cast(src, transpose);
            end = read_cycles();
            DBG("[casting data] start = %lu, end = %lu, elapsed = %lu\n", start, end, end - start);
        }

        else
        {
            std::memset(data_, 0, buf_bytes_);
        }
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
            if (!transpose)
                q80_to_T_rowwise<T>(src, dst_base, stride_, this->rows_, this->cols_);
            else
                q80_to_T_transposed<T>(src, dst_base, stride_, this->rows_, this->cols_);

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