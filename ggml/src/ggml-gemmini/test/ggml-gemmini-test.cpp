// ggml-gemmini-test.cpp
#include "ggml-gemmini-test.h"
#include <algorithm>
#include <memory>
#include <cstdio>  
#include <type_traits>
#include <limits>
#include "include/gemmini.h"

namespace aisa
{
    // 1. 최상위 테스트 엔트리
    void ggml_gemmini_test(ggml_backend_gemmini_context *ctx, struct ggml_tensor *dst)
    {
        TestConfig config;

        // Testbench 객체를 생성하고 실행
        GemminiTestbench testbench(ctx, dst, config);
        testbench.run();
    }

    // 2. GemminiTestbench 클래스 구현
    GemminiTestbench::GemminiTestbench(ggml_backend_gemmini_context *ctx, ggml_tensor *dst, const TestConfig &config)
        : ctx_(ctx), dst_(dst), src0_(dst->src[0]), src1_(dst->src[1]), config_(config)
    {
        GGML_ASSERT(ctx_ && dst_ && src0_ && src1_);
    }

    void GemminiTestbench::run()
    {
        setUpDimensions();
        if (config_.test_shape)
            debugShapes();

        createTensors();

        if (config_.test_slice)
            dumpTensorSlices();

        prepareBias();

        if (config_.run_cpu_ref)
            runCpuReference();
        if (config_.run_gemmini)
            runGemminiComputation();
        if (config_.compare_results)
            compareAndReport();
        if (config_.dequantize_output)
            dequantizeAndFinalize();
    }

    void GemminiTestbench::setUpDimensions()
    {
        DBG0("[setUpDimensions]...\n");
        J_ = (int)dst_->ne[0];
        I_ = (int)dst_->ne[1];
        K_ = (int)src1_->ne[0];

        GGML_ASSERT((int)src0_->ne[1] == J_);
        GGML_ASSERT((int)src1_->ne[1] == I_);
        GGML_ASSERT((int)src0_->ne[0] == K_);
    }

    void GemminiTestbench::debugShapes()
    {
        DBG0("[debugShapes] Logical Dims: I=%d, J=%d, K=%d\n", I_, J_, K_);
        DBG0(" dst  (logical I x J) stored ne=[%llu,%llu], nb=[%llu,%llu]\n",
             (unsigned long long)dst_->ne[0], (unsigned long long)dst_->ne[1],
             (unsigned long long)dst_->nb[0], (unsigned long long)dst_->nb[1]);
        DBG0(" src0 (W^T stored K x J, logical W J x K) ne=[%llu,%llu], nb=[%llu,%llu]\n",
             (unsigned long long)src0_->ne[0], (unsigned long long)src0_->ne[1],
             (unsigned long long)src0_->nb[0], (unsigned long long)src0_->nb[1]);
        DBG0(" src1 (A stored K x I,   logical A I x K) ne=[%llu,%llu], nb=[%llu,%llu]\n",
             (unsigned long long)src1_->ne[0], (unsigned long long)src1_->ne[1],
             (unsigned long long)src1_->nb[0], (unsigned long long)src1_->nb[1]);
    }

    void GemminiTestbench::createTensors()
    {
        DBG0("[createTensors] Creating GemminiTensors...\n");
        // Activation: 버퍼만 재사용, 값은 매번 갱신 (BenchTensor)
        tA_ = aisa::GemminiTensor<int8_t>::getOrCreateTransient(ctx_, src1_, ".i8_A", false);
        // Weight: 완전 캐싱 (0-fill) 고정 (BenchTensor)
        tB_ = aisa::GemminiTensor<int8_t>::getOrCreate(ctx_, src0_, ".i8_B", false, TRANSPOSE_B); // 항상 KxJ로 간주
        // Output: 버퍼만 재사용. Gemmini 결과 저장
        tC_ = aisa::GemminiTensor<int8_t>::getOrCreateTransient(ctx_, dst_, ".i8_C", false);

        DBG0("[createTensors] Validating tensor dimensions...\n");

        // tA 검증 (I x K)
        if (tA_->getRows() != (size_t)I_)
        {
            DBG0("[ERROR] tA dimension mismatch!\n");
            DBG0("  Source tensor (src1): ne=[%lld,%lld] (K x I in storage)\n", 
                 (long long)src1_->ne[0], (long long)src1_->ne[1]);
            DBG0("  Expected logical dims: I=%d, K=%d => tA should be %dx%d\n", I_, K_, I_, K_);
            DBG0("  Actual tA dims: %zux%zu\n", tA_->getRows(), tA_->getCols());
            DBG0("  transpose_A=%s\n", "false");
            GGML_ASSERT(false);
        }
        GGML_ASSERT(tA_->getCols() == (size_t)K_);
        
        // tB 검증 (K x J)
        if (tB_->getRows() != (size_t)K_)
        {
            DBG0("[ERROR] tB dimension mismatch!\n");
            DBG0("  Source tensor (src0): ne=[%lld,%lld] (K x J in storage)\n", 
                 (long long)src0_->ne[0], (long long)src0_->ne[1]);
            DBG0("  Expected logical dims: J=%d, K=%d => tB should be %dx%d\n", J_, K_, K_, J_);
            DBG0("  Actual tB dims: %zux%zu\n", tB_->getRows(), tB_->getCols());
            DBG0("  transpose_B=%s\n", TRANSPOSE_B ? "true" : "false");
            GGML_ASSERT(false);
        }
        GGML_ASSERT(tB_->getCols() == (size_t)J_);
        
        // tC 검증 (I x J)
        if (tC_->getRows() != (size_t)I_)
        {
            DBG0("[ERROR] tC dimension mismatch!\n");
            DBG0("  Source tensor (dst): ne=[%lld,%lld] (J x I in storage)\n", 
                 (long long)dst_->ne[0], (long long)dst_->ne[1]);
            DBG0("  Expected logical dims: I=%d, J=%d => tC should be %dx%d\n", I_, J_, I_, J_);
            DBG0("  Actual tC dims: %zux%zu\n", tC_->getRows(), tC_->getCols());
            GGML_ASSERT(false);
        }
        GGML_ASSERT(tC_->getCols() == (size_t)J_);
        
        DBG0("[createTensors] All tensor dimensions validated successfully.\n");
        DBG0("  tA: %zux%zu (I=%d x K=%d) ✓\n", tA_->getRows(), tA_->getCols(), I_, K_);
        DBG0("  tB: %zux%zu (K=%d x J=%d) ✓\n", tB_->getRows(), tB_->getCols(), K_, J_);
        DBG0("  tC: %zux%zu (I=%d x J=%d) ✓\n", tC_->getRows(), tC_->getCols(), I_, J_);
    }

    void GemminiTestbench::prepareBias()
    {
        static const std::vector<int32_t> zero_bias(J_, 0);
        bias_data_ = zero_bias.data();
    }

    void GemminiTestbench::runCpuReference()
    {
        DBG0("[runCpuReference]...\n");
        const int8_t *A = static_cast<const int8_t *>(tA_->get());
        const int8_t *B = static_cast<const int8_t *>(tB_->get());
        const size_t sA = tA_->getStride();
        const size_t sB = tB_->getStride();
        const size_t sC = tC_->getStride();

        cpu_ref_c_.assign(I_ * sC, 0);

        for (int i = 0; i < I_; ++i)
        {
            for (int j = 0; j < J_; ++j)
            {
                int32_t acc = 0;
                for (int k = 0; k < K_; ++k)
                    acc += (int32_t)A[i * sA + k] * (int32_t)B[k * sB + j];
                acc += bias_data_[j];
                cpu_ref_c_[i * sC + j] = saturationToInt8(acc);
            }
        }
    }

    void GemminiTestbench::runGemminiComputation()
    {
        DBG0("[runGemminiComputation]...\n");
        tiled_matmul_auto(
            I_, J_, K_,
            (elem_t *)tA_->get(), (elem_t *)tB_->get(),
            (void *)bias_data_, (elem_t *)tC_->get(),
            tA_->getStride(), tB_->getStride(), 0, tC_->getStride(),
            1.f, 1.f, 1.f, NO_ACTIVATION,
            1, 1, true, false, !TRANSPOSE_B, false, false, 0, OPTION);
    }

    void GemminiTestbench::compareAndReport()
    {
        DBG0("[compareAndReport]...\n");

        bool ok = true;
        const int8_t *C_gemmini = static_cast<const int8_t *>(tC_->get());
        const size_t sC = tC_->getStride();

        for (int i = 0; i < I_; ++i)
        {
            for (int j = 0; j < J_; ++j)
            {
                elem_t got = C_gemmini[i * sC + j];
                elem_t exp = cpu_ref_c_[i * sC + j];
                if (got != exp)
                {
                    DBG0("[NG] mismatch (%d,%d): got=%d exp=%d\n", i, j, (int)got, (int)exp);
                    ok = false;
                }
            }
        }
        DBG0(ok ? "[OK] Gemmini matmul matches CPU reference\n"
                : "[FAIL] Mismatch detected\n");
    }

    void GemminiTestbench::dequantizeAndFinalize()
    {
        DBG0("[dequantizeAndFinalize]...\n");
        if (dst_->type != GGML_TYPE_F32)
            return;

        const int8_t *C_i8 = static_cast<const int8_t *>(tC_->get());
        const size_t sC = tC_->getStride();
        const size_t nb1 = dst_->nb[1];
        uint8_t *out_base = static_cast<uint8_t *>(dst_->data);
        for (int r = 0; r < I_; ++r)
        {
            const elem_t *row_c = C_i8 + (size_t)r * sC;
            float *row_out = reinterpret_cast<float *>(out_base + (size_t)r * nb1);
            for (int j = 0; j < J_; ++j)
                row_out[j] = (float)row_c[j];
        }
    }

    void GemminiTestbench::dumpTensorSlices()
    {
#if DUMP
        DBG0("[dumpTensorSlices]...\n");

        // 1. 클래스 멤버 변수(I_, K_, J_)를 사용하여 슬라이스 크기 계산
        const int vI = std::min(I_, SLICE_I);
        const int vK = std::min(K_, SLICE_K);
        const int vJ = std::min(J_, SLICE_J);

        // 1) 로컬 뷰 컨텍스트 (메타데이터만, 데이터는 부모 공유)
        ggml_init_params ip = {};
        ip.mem_size  = 256 * 1024;   // 텐서 노드 3~6개면 충분 (필요시 512KB로)
        ip.mem_buffer= nullptr;
        ip.no_alloc  = true;

        ggml_context* vctx = ggml_init(ip);
        GGML_ASSERT(vctx);
        
        // 2. 클래스 멤버 변수(ctx_, src1_, src0_, dst_)를 사용하여 ggml 뷰 생성
        ggml_tensor *A_slice = ggml_view_2d(vctx, const_cast<ggml_tensor *>(src1_), vK, vI, src1_->nb[1], 0);
        ggml_tensor *B_slice = ggml_view_2d(vctx, const_cast<ggml_tensor *>(src0_), vJ, vK, src0_->nb[1], 0);
        ggml_tensor *C_slice = ggml_view_2d(vctx, dst_, vJ, vI, dst_->nb[1], 0);

        DBG0("[SLICE] View Dims: vI=%d, vK=%d, vJ=%d | Logical Dims: I=%d, K=%d, J=%d\n", vI, vK, vJ, I_, K_, J_);

        // 3. 타입에 따라 행렬을 덤프하는 람다 함수
        auto dump_any = [](const char *tag, const ggml_tensor *t, int r, int c, int s)
        {
            switch (t->type)
            {
            case GGML_TYPE_I8:
                GGML_ASSERT(t->nb[0] == sizeof(int8_t));
                aisa::dumpMatrix<int8_t>(tag, (const int8_t *)t->data, r, c, s);
                break;
            case GGML_TYPE_F32:
                GGML_ASSERT(t->nb[0] == sizeof(float));
                aisa::dumpMatrix<float>(tag, (const float *)t->data, r, c, s);
                break;
            case GGML_TYPE_I32:
                GGML_ASSERT(t->nb[0] == sizeof(acc_t));
                aisa::dumpMatrix<acc_t>(tag, (const acc_t *)t->data, r, c, s);
                break;
            default:
                DBG0("%s: unsupported ggml type=%d — skipped\n", tag, (int)t->type);
                break;
            }
        };

        auto elems_stride = [](const ggml_tensor *t) -> int
        {
            GGML_ASSERT(t && t->nb[0] > 0);
            return (int)(t->nb[1] / t->nb[0]);
        };

        const int sA_view = elems_stride(A_slice);
        const int sB_view = elems_stride(B_slice);
        const int sC_view = elems_stride(C_slice);

        dump_any("Original A Slice (from src1)", A_slice, vI, vK, sA_view);
        dump_any("Original B Slice (from src0)", B_slice, vK, vJ, sB_view);
        dump_any("Original C Slice (from dst)", C_slice, vI, vJ, sC_view);

        // 5. 변환된 GemminiTensor의 내부 버퍼를 덤프
        aisa::dumpMatrix<int8_t>("Converted tA (I x K)",
                                 static_cast<const int8_t *>(tA_->get()), vI, vK, (int)tA_->getStride());
        aisa::dumpMatrix<int8_t>("Converted tB (K x J)",
                                 static_cast<const int8_t *>(tB_->get()), vK, vJ, (int)tB_->getStride());
        aisa::dumpMatrix<int8_t>("Converted tC (I x J)",
                                 static_cast<const int8_t *>(tC_->get()), vI, vJ, (int)tC_->getStride());
#endif // DUMP
    }

    // utils
    static inline int8_t saturationToInt8(int x)
    {
        return x > 127 ? 127 : (x < -128 ? -128 : (int8_t)x);
    }

    template <typename T>
    static inline void dumpMatrix(const char *name, const T *m, int r, int c, int s)
    {
#if DUMP
        if (!m)
        {
            DBG0("%s: <null>\n", name);
            return;
        }
        DBG0("%s (r=%d, c=%d, ld=%d) =\n", name, r, c, s);

        for (int i = 0; i < r; ++i)
        {
            DBG0("[ ");
            for (int j = 0; j < c; ++j)
            {
                const T v = m[(size_t)i * (size_t)s + (size_t)j];

                if constexpr (std::is_same_v<T, float>)
                {
                    // fp32
                    DBG0("%.6g ", (double)v);
                }
                else if constexpr (std::is_same_v<T, int8_t>)
                {
                    // int8 -> 가독성을 위해 int로 승격 출력
                    DBG0("%d ", (int)v);
                }
                else if constexpr (std::is_same_v<T, acc_t>)
                {
                    // acc_t (일반적으로 int32_t)
                    if constexpr (std::numeric_limits<acc_t>::is_signed)
                        DBG0("%lld ", (long long)v);
                    else
                        DBG0("%llu ", (unsigned long long)v);
                }
                else
                {
                    // 컴파일 타임 가드: 지원 타입 외 사용 방지
                    static_assert(std::is_same_v<T, void>,
                                  "dump_matrix: supported types are float, int8_t, acc_t only.");
                }
            }
            DBG0("]\n");
        }
#else
        (void)name;
        (void)m;
        (void)r;
        (void)c;
        (void)s;
#endif
    }
}