// ggml-gemmini-tensor.cpp
#define DEBUG 1

#include "ggml-gemmini-tensor.h"

namespace zerogod
{
    template <typename T>
    static inline ggml_type ggml_type_of()
    {
        return std::is_same<T, int8_t>::value ? GGML_TYPE_I8
                                              : GGML_TYPE_I32;
    }

    // 생성자
    template <typename T>
    ggml_gemmini_tensor<T>::ggml_gemmini_tensor(ggml_context *ctx,
                                                const ggml_tensor *src,
                                                const char *suffix,
                                                bool acc,
                                                bool transpose)
    {

        DBG("\ngenerate ggml_gemmini_tensor from: %s, type=%s transpose=%d\n", src->name, ggml_type_name(src->type), transpose);

        /* 1. ____________________원본 행/열____________________
              ggml 네이티브: ne[0] = columns(X), ne[1] = rows(Y) */
        const int src_cols = transpose ? src->ne[1] : src->ne[0];
        const int src_rows = transpose ? src->ne[0] : src->ne[1];
        ggml_type type = ggml_type_of<T>();

        /* 2. _____16-byte row-stride 정렬을 위한 colum 패딩_____ */
        const size_t elem_size = sizeof(T);
        const size_t align_elems = GEMMINI_ALIGN / elem_size;
        const int padded_cols = align_up(src_cols, align_elems);

        /* 3. ___________________tensor 생성___________________ */
        tensor_ = ggml_new_tensor_2d(ctx, type, padded_cols, src_rows);
        snprintf(tensor_->name, sizeof(tensor_->name), "%s%s", src->name, suffix);

        this->rows_ = tensor_->ne[1];
        this->cols_ = tensor_->ne[0];

        /* 4. __________________buffer 할당____________________ */
        const size_t row_bytes = align_up(this->cols_ * elem_size, GEMMINI_ALIGN);
        buf_bytes_ = row_bytes * src_rows;

        if (buf_bytes_ == 0)
            buf_bytes_ = GEMMINI_ALIGN; // 최소 16 B 확보

        this->data_ = std::aligned_alloc(GEMMINI_ALIGN, buf_bytes_); // buffer을 16B 경계에 할당
        GGML_ASSERT(this->data_ != nullptr);

        tensor_->data = this->data_;
        tensor_->nb[0] = elem_size;
        tensor_->nb[1] = row_bytes;
        stride_ = row_bytes / elem_size;

        DBG("\ngenerated tensor: type=%s, cols=%d, rows=%d, buf_bytes=%zu\n", ggml_type_name(type), tensor_->ne[0], tensor_->ne[1], buf_bytes_);

        /* 5. _______________casting & 0-fill _________________ */
        if (!acc)
            ggml_gemmini_cast(src, transpose);
        else
            std::memset(data_, 0, buf_bytes_);

        /* 6. _________________stride 업데이트__________________ */
        update_stride();
    }

    // 소멸자 & 버퍼 해제
    template <typename T>
    ggml_gemmini_tensor<T>::~ggml_gemmini_tensor() { free_buffer(); }

    template <typename T>
    void ggml_gemmini_tensor<T>::free_buffer()
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
    ggml_gemmini_tensor<T>::ggml_gemmini_tensor(ggml_gemmini_tensor &&other) noexcept
        : tensor_(other.tensor_), data_(other.data_), buf_bytes_(other.buf_bytes_), rows_(other.rows_), cols_(other.cols_), stride_(other.stride_)
    {
        other.tensor_ = nullptr;
        other.data_ = nullptr;
        other.buf_bytes_ = 0;
        other.rows_ = other.cols_ = other.stride_ = 0;
    }

    template <typename T>
    ggml_gemmini_tensor<T> &
    ggml_gemmini_tensor<T>::operator=(ggml_gemmini_tensor &&other) noexcept
    {
        if (this != &other)
        {
            free_buffer();
            tensor_ = other.tensor_;
            data_ = other.data_;
            buf_bytes_ = other.buf_bytes_;
            rows_ = other.rows_;
            cols_ = other.cols_;
            stride_ = other.stride_;

            other.tensor_ = nullptr;
            other.data_ = nullptr;
            other.buf_bytes_ = 0;
            other.rows_ = other.cols_ = other.stride_ = 0;
        }
        return *this;
    }

    template <typename T>
    void ggml_gemmini_tensor<T>::ggml_gemmini_cast(const ggml_tensor *src,
                                                   bool transpose) const
    {
        /* _________________1. 원본 shape/stride_________________*/
        const int src_cols = transpose ? src->ne[1] : src->ne[0];
        const int src_rows = transpose ? src->ne[0] : src->ne[1];

        const size_t src_row_bytes = src->nb[1]; // 행 간 byte-stride
        const size_t src_col_bytes = src->nb[0]; // 열 간 byte-stride

        /* _____________________2. dst 정보______________________*/
        uint8_t *dst_row = static_cast<uint8_t *>(this->data_);
        const size_t dst_row_bytes = tensor_->nb[1]; // 16B align된 값
        const size_t elem_size = static_cast<size_t>(ggml_type_size(ggml_type_of<T>()));

        /* ___________________3. 원본 타입별 분기__________________*/
        switch (src->type)
        {
        case GGML_TYPE_F32:
        {
            const uint8_t *src_base = static_cast<const uint8_t *>(src->data);

            for (size_t r = 0; r < src_rows; ++r)
            {
                T *dst_elem = reinterpret_cast<T *>(dst_row);
                if (!transpose)
                    // src 행 r 를 그대로 복사 : 주소 = base + r*src_row_bytes + c*src_col_bytes
                    for (size_t c = 0; c < src_cols; ++c)
                    {
                        const float *p = reinterpret_cast<const float *>(src_base + r * src_row_bytes + c * src_col_bytes);
                        dst_elem[c] = static_cast<T>(*p);
                    }
                else
                    // 전치 복사 : src( c , r ) -> dst( r , c )
                    for (size_t c = 0; c < src_cols; ++c)
                    {
                        const float *p = reinterpret_cast<const float *>(src_base + c * src_row_bytes + r * src_col_bytes);
                        dst_elem[c] = static_cast<T>(*p);
                    }

                // 0-fill
                if (src_cols < this->cols_)
                    std::memset(dst_elem + src_cols, 0, (this->cols_ - src_cols) * elem_size);

                dst_row += dst_row_bytes;
            }
            break;
        }
        case GGML_TYPE_Q8_0:
        {
            /* TODO: copy */
            std::memset(dst_row, 0, buf_bytes_); // 임시 패딩
            break;
        }
        default:
        {
            GGML_ASSERT(false && "ggml_gemmini_cast: unsupported src type");
            break;
        }
        }
    }

    template <typename T>
    void ggml_gemmini_tensor<T>::update_stride()
    {
        for (int d = 2; d < GGML_MAX_DIMS; ++d)
            tensor_->nb[d] = tensor_->nb[d - 1] * tensor_->ne[d - 1];
    }

    // explicit instantiation : 지원 타입 한정
    template class ggml_gemmini_tensor<int8_t>;
    template class ggml_gemmini_tensor<int32_t>;

} // namespace zerogod