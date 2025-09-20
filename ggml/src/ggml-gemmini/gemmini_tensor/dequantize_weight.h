//gemmini_tensor/dequantize_weight.h
#pragma once
#include <cstring>
#include <cstdint>
#include <type_traits>

#include "include/gemmini.h"
#include "ggml.h"

#ifndef QK8_0
#define QK8_0 32
#endif

namespace aisa
{
    struct block_q8_0
    {
        int8_t qs[QK8_0]; // 우리가 추출할 대상
        ggml_fp16_t d;    // 스케일(이번 작업에서는 폐기)
    };

    inline const block_q8_0 *get_q80_row_ptr(const ggml_tensor *src,
                                             int64_t iy, int64_t iz, int64_t iw)
    {
        const char *base = (const char *)(src->view_src ? src->view_src->data : src->data);
        const size_t offs = src->view_src ? src->view_offs : 0;
        return reinterpret_cast<const block_q8_0 *>(
            base + offs + iw * src->nb[3] + iz * src->nb[2] + iy * src->nb[1]);
    }
    template <typename T>
    inline void copy_qs_block_to_T(const block_q8_0 &blk, T *dst)
    {
        if constexpr (std::is_same_v<T, int8_t>)
        {
            std::memcpy(dst, blk.qs, QK8_0 * sizeof(int8_t));
        }
        else
        {
            for (int j = 0; j < QK8_0; ++j)
                dst[j] = static_cast<T>(blk.qs[j]);
        }
    }

    // (transpose=false) : 행 단위로 블록 복사 (가장 빠름)
    template <typename T>
    inline void q80_to_T_rowwise(const ggml_tensor *src,
                                 T *dst_base, size_t dst_stride_elems,
                                 int64_t rows, int64_t cols)
    {
        DBG("checking bp2\n");
        GGML_ASSERT(src->type == GGML_TYPE_Q8_0);
        GGML_ASSERT(cols % QK8_0 == 0);
        const int64_t nblk = cols / QK8_0;

        // (w,z) 전개
        const int64_t ny = src->ne[0] ? src->ne[0] : 1;
        const int64_t nz = src->ne[2] ? src->ne[2] : 1;
        const int64_t nw = src->ne[3] ? src->ne[3] : 1;

        int64_t row_idx = 0;
        for (int64_t iw = 0; iw < nw; ++iw)
        {
            for (int64_t iz = 0; iz < nz; ++iz)
            {
                for (int64_t iy = 0; iy < ny; ++iy)
                {
                    const block_q8_0 *row_blocks = get_q80_row_ptr(src, iy, iz, iw);
                    T *dst_row = dst_base + row_idx * dst_stride_elems;

                    for (int64_t b = 0; b < nblk; ++b)
                    {
                        copy_qs_block_to_T<T>(row_blocks[b], dst_row + b * QK8_0);
                    }
                    ++row_idx;
                }
            }
        }
        GGML_ASSERT(row_idx == rows);
    }
    template <typename T>
    inline void q80_to_T_transposed(const ggml_tensor *src,
                                    T *dst_base, size_t dst_stride_elems,
                                    int64_t rows, int64_t cols)
    {
        DBG("checking bp3\n");
        GGML_ASSERT(src->type == GGML_TYPE_Q8_0);
        GGML_ASSERT(rows % QK8_0 == 0 || rows % QK8_0 == 0 || rows > 0); // rows(=src->ne[0]) 제약은 호출부에서 보장
        const int64_t nx_src = src->ne[0];                               // 원 src의 x 길이
        GGML_ASSERT(nx_src % QK8_0 == 0);
        const int64_t nblk_x = nx_src / QK8_0;
        const int64_t ny = src->ne[0] ? src->ne[0] : 1;
        const int64_t nz = src->ne[2] ? src->ne[2] : 1;
        const int64_t nw = src->ne[3] ? src->ne[3] : 1;
        // DBG0("rows : %d ny : %d, nz: %d, nw %d\n",rows, ny, nz, nw);

        // 논리적으로 dst_rows = src_cols(ny*nz*nw), dst_cols = src_rows(nx_src)
        // 여기서 rows = dst_rows, cols = dst_cols 로 들어온다.
        // (w,z) 묶음으로 각 y를 'dst의 행'으로 봄
        int64_t dst_row_idx = 0;
        for (int64_t iw = 0; iw < nw; ++iw)
        {

            // DBG("checing bp w: %d \n", iw);
            for (int64_t iz = 0; iz < nz; ++iz)
            {

                // DBG("checking bp z: ");
                for (int64_t iy = 0; iy < ny; ++iy)
                {
                    // DBG0("Checking bp y : %d\n", iy);
                    const block_q8_0 *src_row_blocks = get_q80_row_ptr(src, iy, iz, iw);
                    // DBG("getting ptr sucess...\n");
                    T *dst_row = dst_base + dst_row_idx * dst_stride_elems;
                    // DBG0("dst_base = %d, dst_row = %d, dst_stride = %d", dst_base, dst_row_idx, dst_stride_elems);
                    // DBG0("cols : %d", cols);
                    //  dst_row[c] = src(x=c, y=iy, z=iz, w=iw)
                    //  src(x=c) → block = c / 32, off = c % 32
                    for (int64_t c = 0; c < cols; ++c)
                    {
                        // if (iy == 824)
                        // DBG0("C=%d", c);
                        const int64_t blk = c / QK8_0;
                        const int off = static_cast<int>(c % QK8_0);
                        if constexpr (std::is_same_v<T, int8_t>)
                        {
                            if (iy == 824)
                            {
                                // DBG0("checking value : %d\n", src_row_blocks[blk].qs[off]);
                                // DBG0("addr : %d", &dst_row[c]);
                                // DBG0("blk : %d\n", blk);
                                // DBG0("off : %d\n", off);
                            }
                            dst_row[c] = src_row_blocks[blk].qs[off];
                            // DBG("checking Is copy sucess");
                        }
                        else
                        {
                            // DBG("no constexpr\n");
                            dst_row[c] = static_cast<T>(src_row_blocks[blk].qs[off]);
                        }
                    }
                    ++dst_row_idx;
                }
            }
        }
        GGML_ASSERT(dst_row_idx == rows);
    }
}
