// ggml-gemmini-tensor.cpp
#define DEBUG 1

#include "ggml-gemmini-tensor.h"

namespace zerogod
{
    struct block_q8_0 {
        int8_t       qs[QK8_0];  // 우리가 추출할 대상
        ggml_fp16_t  d;          // 스케일(이번 작업에서는 폐기)
    };


    inline const block_q8_0 *get_q80_row_ptr(const ggml_tensor *src,
                                         int64_t iy, int64_t iz, int64_t iw) 
    {
        const char *base = (const char *)(src->view_src ? src->view_src->data : src->data);
        const size_t offs = src->view_src ? src->view_offs : 0;
        return reinterpret_cast<const block_q8_0 *>(
            base + offs + iw * src->nb[3] + iz * src->nb[2] + iy * src->nb[1]
        );
    }
    template<typename T>
    inline void copy_qs_block_to_T(const block_q8_0 &blk, T *dst) 
    {
        if constexpr (std::is_same_v<T, int8_t>) {
            std::memcpy(dst, blk.qs, QK8_0 * sizeof(int8_t));
        } else {
            for (int j = 0; j < QK8_0; ++j) dst[j] = static_cast<T>(blk.qs[j]);
        }
    }   

    template<typename T>
    inline void copy_qs_block_to_T(const block_q8_0 &blk, T *dst) {
        if constexpr (std::is_same_v<T, int8_t>) {
            std::memcpy(dst, blk.qs, QK8_0 * sizeof(int8_t));
        } else {
            for (int j = 0; j < QK8_0; ++j) dst[j] = static_cast<T>(blk.qs[j]);
        }
    }

    // (transpose=false) : 행 단위로 블록 복사 (가장 빠름)
    template<typename T>
    inline void q80_to_T_rowwise(const ggml_tensor *src,
                                 T *dst_base, size_t dst_stride_elems,
                                 int64_t rows, int64_t cols) {
        GGML_ASSERT(src->type == GGML_TYPE_Q8_0);
        GGML_ASSERT(cols % QK8_0 == 0);
        const int64_t nblk = cols / QK8_0;

        // (w,z) 전개
        const int64_t ny = src->ne[1] ? src->ne[1] : 1;
        const int64_t nz = src->ne[2] ? src->ne[2] : 1;
        const int64_t nw = src->ne[3] ? src->ne[3] : 1;

        int64_t row_idx = 0;
        for (int64_t iw = 0; iw < nw; ++iw) {
            for (int64_t iz = 0; iz < nz; ++iz) {
                for (int64_t iy = 0; iy < ny; ++iy) {
                    const block_q8_0 *row_blocks = get_q80_row_ptr(src, iy, iz, iw);
                    T *dst_row = dst_base + row_idx * dst_stride_elems;

                    for (int64_t b = 0; b < nblk; ++b) {
                        copy_qs_block_to_T<T>(row_blocks[b], dst_row + b * QK8_0);
                    }
                    ++row_idx;
                }
            }
        }
        GGML_ASSERT(row_idx == rows);
    }
    template<typename T>
    inline void q80_to_T_transposed(const ggml_tensor *src,
                                    T *dst_base, size_t dst_stride_elems,
                                    int64_t rows, int64_t cols) {
        GGML_ASSERT(src->type == GGML_TYPE_Q8_0);
        GGML_ASSERT(rows % QK8_0 == 0 || rows % QK8_0 == 0 || rows > 0); // rows(=src->ne[0]) 제약은 호출부에서 보장
        const int64_t nx_src = src->ne[0]; // 원 src의 x 길이
        GGML_ASSERT(nx_src % QK8_0 == 0);
        const int64_t nblk_x = nx_src / QK8_0;

        const int64_t ny = src->ne[1] ? src->ne[1] : 1;
        const int64_t nz = src->ne[2] ? src->ne[2] : 1;
        const int64_t nw = src->ne[3] ? src->ne[3] : 1;

        // 논리적으로 dst_rows = src_cols(ny*nz*nw), dst_cols = src_rows(nx_src)
        // 여기서 rows = dst_rows, cols = dst_cols 로 들어온다.
        // (w,z) 묶음으로 각 y를 'dst의 행'으로 봄
        int64_t dst_row_idx = 0;
        for (int64_t iw = 0; iw < nw; ++iw) {
            for (int64_t iz = 0; iz < nz; ++iz) {
                for (int64_t iy = 0; iy < ny; ++iy) {
                    const block_q8_0 *src_row_blocks = get_q80_row_ptr(src, iy, iz, iw);
                    T *dst_row = dst_base + dst_row_idx * dst_stride_elems;

                    // dst_row[c] = src(x=c, y=iy, z=iz, w=iw)
                    // src(x=c) → block = c / 32, off = c % 32
                    for (int64_t c = 0; c < cols; ++c) {
                        const int64_t blk = c / QK8_0;
                        const int     off = static_cast<int>(c % QK8_0);
                        if constexpr (std::is_same_v<T, int8_t>) {
                            dst_row[c] = src_row_blocks[blk].qs[off];
                        } else {
                            dst_row[c] = static_cast<T>(src_row_blocks[blk].qs[off]);
                        }
                    }
                    ++dst_row_idx;
                }
            }
        }
        GGML_ASSERT(dst_row_idx == rows);
    }

/////////////////////////////////////////////////////////////////////////////////////////////////

    template <typename T>
    static inline ggml_type ggml_type_of()
    {
        return std::is_same<T, int8_t>::value ? GGML_TYPE_I8
                                              : GGML_TYPE_I32;
    }

    template <typename T>
    inline int64_t numel4(const ggml_tensor *t) 
    {
        const int64_t n0 = t->ne[0] ? t->ne[0] : 1;
        const int64_t n1 = t->ne[1] ? t->ne[1] : 1;
        const int64_t n2 = t->ne[2] ? t->ne[2] : 1;
        const int64_t n3 = t->ne[3] ? t->ne[3] : 1;
        return n0 * n1 * n2 * n3;
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
            if (src->type == GGML_TYPE_Q8_0){
                const int64_t rows = src_rows;
                const int64_t cols = src_cols;

                // 목적지: 패딩 반영된 행 스트라이드(요소 단위)
                T *dst_base = reinterpret_cast<T *>(this->data_);
                const size_t dst_stride_elems = this->stride_;

                if (!transpose) {
                    // src의 x축 방향(Q8_0 블록)이 그대로 열이며, 행 단위 직복사
                    q80_to_T_rowwise<T>(src, dst_base, dst_stride_elems, rows, cols);
                } else {
                    // 전치: dst(r,c) = src(x=r, y=c). 1원소 gather
                    q80_to_T_transposed<T>(src, dst_base, dst_stride_elems, rows, cols);
                }

                // 패딩 영역은 0으로 채우기(열 패딩분)
                if (padded_cols > src_cols) {
                    for (int64_t r = 0; r < rows; ++r) {
                        T *rowp = dst_base + r * dst_stride_elems;
                        std::memset(rowp + src_cols, 0, (padded_cols - src_cols) * sizeof(T));
                    }
                }
            }else 
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
    void extract_q80_qs_to_linear(
                                const ggml_tensor *t_q80,
                                T *out_linear      // 길이 = numel4(t_q80)
                                ) 
    {
        static_assert(std::is_integral<T>::value || std::is_floating_point<T>::value,
                      "T must be arithmetic type");
        assert(t_q80);
        assert(t_q80->type == GGML_TYPE_Q8_0);
        assert(t_q80->ne[0] % QK8_0 == 0);

        const int64_t nx = t_q80->ne[0];
        const int64_t ny = t_q80->ne[1] ? t_q80->ne[1] : 1;
        const int64_t nz = t_q80->ne[2] ? t_q80->ne[2] : 1;
        const int64_t nw = t_q80->ne[3] ? t_q80->ne[3] : 1;

        // view 대응
        const char  *base = (const char *)(t_q80->view_src ? t_q80->view_src->data : t_q80->data);
        const size_t offs  = t_q80->view_src ? t_q80->view_offs : 0;

        const size_t nb1 = t_q80->nb[1];
        const size_t nb2 = t_q80->nb[2];
        const size_t nb3 = t_q80->nb[3];

        const int64_t nblk = nx / QK8_0;

        int64_t row_idx = 0;
        for (int64_t iw = 0; iw < nw; ++iw) {
            for (int64_t iz = 0; iz < nz; ++iz) {
                for (int64_t iy = 0; iy < ny; ++iy) {
                    const char *row_ptr = base + offs + iw * nb3 + iz * nb2 + iy * nb1;
                    const block_q8_0 *row_blocks = reinterpret_cast<const block_q8_0 *>(row_ptr);

                    T *dst_row = out_linear + row_idx * nx;

                    // 블록 단위로 qs[32]를 복사/캐스팅
                    for (int64_t b = 0; b < nblk; ++b) {
                        // qs는 int8_t[32]. T가 int8_t면 memcpy, 아니면 캐스팅 복사
                        if constexpr (std::is_same<T,int8_t>::value) {
                            std::memcpy(dst_row + b * QK8_0, row_blocks[b].qs, QK8_0 * sizeof(int8_t));
                        } else {
                            for (int j = 0; j < QK8_0; ++j) {
                                dst_row[b * QK8_0 + j] = static_cast<T>(row_blocks[b].qs[j]);
                            }
                        }
                    }

                    ++row_idx;
                }
            }
        }
        // sanity
        assert(row_idx == (ny * nz * nw));
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
            // DBG("----------------------Q8_0 type----------------------\n")
            // DBG("\nchecking q8_0 tensor: type=%s, cols=%d, rows=%d, buf_bytes=%zu\n", ggml_type_name(src->type), src->ne[0], src->ne[1], buf_bytes_);
            // assert(ctx && src);
            // assert(src->type == GGML_TYPE_Q8_0);
            // assert(t_q80->ne[0] % QK8_0 == 0);

            // int64_t ne[4] = { t_q80->ne[0], t_q80->ne[1], t_q80->ne[2], t_q80->ne[3] };

            // // ggml 타입 매핑
            // constexpr ggml_type out_type = ggml_type_of<T>::value;
            // ggml_tensor *t_out = ggml_new_tensor(ctx, out_type, 4, ne);

            // const int64_t n = numel4(t_q80);
            // // 임시 연속 버퍼에 추출 후 텐서 메모리에 복사
            // // (ggml_new_tensor()는 보통 contiguous)
            // T *tmp = reinterpret_cast<T *>(malloc(sizeof(T) * (size_t)n));
            // assert(tmp);

            // extract_q80_qs_to_linear<T>(t_q80, tmp);
            // std::memset(dst_row, tmp, sizeof(T) * (size_t)n)
            // free(tmp);

            ////////////////////////////////////////////////////////////////////////////////////////////
            

            // const int64_t K = src->ne[0];                 // 열 길이 (k)
            // const int64_t I = src->ne[1] * (src->ne[2] ? src->ne[2] : 1) * (src->ne[3] ? src->ne[3] : 1); // 총 행 수
            // const size_t  row_stride_bytes_q = src->nb[1]; // Q8_0 텐서의 행 간 byte stride
            // const size_t  row_stride_blocks  = row_stride_bytes_q / sizeof(block_q8_0); // 한 행에서 block 개수 * sizeof(block_q8_0) 와 일치해야 함

            // // 임시 float 버퍼 (연속 메모리): A_f[I*K], B_f[I*K]
            // float *A_f = (float *)malloc(sizeof(float)*I*K);
            // float *B_f = (float *)malloc(sizeof(float)*I*K);

            // if (src->type == GGML_TYPE_Q8_0) {
            //     for (int64_t i = 0; i < I; ++i) {
            //         const block_q8_0 *row_blocks = (const block_q8_0 *)((const char*)src->data + i * row_stride_bytes_q);
            //         dequantize_row_q8_0(row_blocks, A_f + i*K, K); // 이 함수는 한 행(k개)을 복원
            //     }
            // }
            // /* TODO: copy */

            // std::memset(dst_row, 0, buf_bytes_); // 임시 패딩
            // break;
            // DBG("----------------------Q8_0 type end----------------------\n")
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