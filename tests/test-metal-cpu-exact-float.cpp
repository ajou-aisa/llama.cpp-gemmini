#include "ggml-metal-cpu-exact.h"
#include "ggml-gemmini-matmul.hpp"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>

static uint32_t random_bits(uint32_t & state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

int main() {
    const size_t shapes[][3] = {{1,7,32}, {4,4,32}, {5,7,64}, {4,5,96}, {8,8,768}, {17,19,2048}, {8,12,128}};
    uint32_t state = 0x913721u;
    size_t compared = 0;
    const uint64_t initial_calls = ggml_metal_cpu_exact_gpu_calls();
    const uint64_t initial_software_calls = ggml_metal_cpu_exact_software_calls();
    for (unsigned pattern=0; pattern<7; ++pattern) {
      for (const auto & s : shapes) {
        const size_t m=s[0], n=s[1], k=s[2];
        std::vector<float> a(m*k), b(n*k), cpu(m*n), gpu(m*n);
        for (float & x : a) {
            x = std::ldexp(static_cast<float>(int(random_bits(state)%65537)-32768)/32768.0f,
                           int(random_bits(state)%17)-8);
        }
        for (float & x : b) {
            x = std::ldexp(static_cast<float>(int(random_bits(state)%65537)-32768)/32768.0f,
                           int(random_bits(state)%17)-8);
        }
        if (pattern == 1) {
            for (float & x : a) { x = std::ldexp(x,60); }
            for (float & x : b) { x = std::ldexp(x,-60); }
        } else if (pattern == 2) {
            for (size_t i=0; i<a.size(); ++i) {
                a[i] = i%4 == 0 ? 1.0e20f : i%4 == 2 ? -1.0e20f : a[i];
            }
            for (float & x : b) { x = 1.00000011920928955078125f; }
        } else if (pattern == 3) {
            for (size_t i=0; i<a.size(); ++i) { a[i] = std::ldexp(float(i%17+1),-145); }
            for (float & x : b) { x = 1.0f; }
        } else if (pattern == 4) {
            for (size_t i=0; i<a.size(); ++i) { a[i] = std::ldexp(float(i%17+1),-72); }
            for (float & x : b) { x = std::ldexp(1.0f,-72); }
        } else if (pattern == 5) {
            for (float & x : a) {
                const uint32_t bits=(random_bits(state)&0x807fffffu) | ((random_bits(state)%255)<<23);
                std::memcpy(&x,&bits,sizeof(x));
            }
            for (float & x : b) {
                const uint32_t bits=(random_bits(state)&0x807fffffu) | ((random_bits(state)%101)<<23);
                std::memcpy(&x,&bits,sizeof(x));
            }
        } else if (pattern == 6) {
            for (size_t i=0; i<a.size(); ++i) { a[i] = i%2 ? -0.0f : 0.0f; }
            for (float & x : b) { x = -1.0f; }
        }
        ggml_gemmini_args_t args{};
        args.I=m; args.J=n; args.K=k;
        args.A_fp32=a.data(); args.B_fp32=b.data();
        args.sA=k; args.sB=k; args.f_out=cpu.data(); args.stride_f_out=n;
        args.transpose_B=true;
        args.tiled_matmul_type=static_cast<tiled_matmul_type_t>(2);
        ggml::gemmini::MatMul operation(args);
        if (operation.run_dense().status != ggml::gemmini::MatMulStatus::success) {
            std::fprintf(stderr,"CPU FLOAT MatMul failed\n");
            return 1;
        }
        if (!ggml_metal_cpu_exact_float(m,n,k,a.data(),b.data(),gpu.data())) {
            std::fprintf(stderr,"GPU failed: %s\n",ggml_metal_cpu_exact_last_error());
            return 1;
        }
        for (size_t i=0; i<cpu.size(); ++i) {
            uint32_t c, g;
            std::memcpy(&c,&cpu[i],sizeof(c));
            std::memcpy(&g,&gpu[i],sizeof(g));
            if (c != g) {
                std::fprintf(stderr,"pattern=%u M=%zu N=%zu K=%zu index=%zu CPU=%08x Metal=%08x\n",pattern,m,n,k,i,c,g);
                return 1;
            }
        }
        compared += cpu.size();
      }
    }
    float sentinel = 13.0f, input = 1.0f;
    if (ggml_metal_cpu_exact_float(1,1,31,&input,&input,&sentinel) || sentinel != 13.0f ||
        ggml_metal_cpu_exact_gpu_calls() != initial_calls+7*(sizeof(shapes)/sizeof(shapes[0])) ||
        ggml_metal_cpu_exact_software_calls() != initial_software_calls+3*(sizeof(shapes)/sizeof(shapes[0]))) {
        std::fprintf(stderr,"Invalid K rejection or GPU call accounting failed\n");
        return 1;
    }
    std::vector<float> invalid(32,1.0f);
    const uint32_t infinity=0x7f800000u;
    std::memcpy(invalid.data(),&infinity,sizeof(infinity));
    if (ggml_metal_cpu_exact_float(1,1,32,invalid.data(),invalid.data(),&sentinel) || sentinel != 13.0f) {
        std::fprintf(stderr,"Nonfinite input transactional rejection failed\n");
        return 1;
    }
    for (float & x : invalid) { x = std::ldexp(1.0f,127); }
    if (ggml_metal_cpu_exact_float(1,1,32,invalid.data(),invalid.data(),&sentinel) || sentinel != 13.0f) {
        std::fprintf(stderr,"Nonfinite output transactional rejection failed\n");
        return 1;
    }
    std::vector<float> fp16_input(32), fp16_reference(32), expected(7), actual(7);
    std::vector<float> matrix_weights(7 * 32);
    for (size_t i = 0; i < 32; ++i) {
        fp16_input[i] = (i % 2 ? -1.0f : 1.0f) * (1.0f + float(i % 7) / 4096.0f);
        fp16_reference[i] = static_cast<float>(static_cast<_Float16>(fp16_input[i]));
        for (size_t j = 0; j < 7; ++j) matrix_weights[j * 32 + i] = float(i + j + 1) / 32.0f;
    }
    ggml_gemmini_args_t rounded_args{};
    rounded_args.I = 1; rounded_args.J = 7; rounded_args.K = 32;
    rounded_args.A_fp32 = fp16_reference.data(); rounded_args.B_fp32 = matrix_weights.data();
    rounded_args.sA = 32; rounded_args.sB = 32;
    rounded_args.f_out = expected.data(); rounded_args.stride_f_out = 7;
    rounded_args.transpose_B = true;
    rounded_args.tiled_matmul_type = static_cast<tiled_matmul_type_t>(2);
    ggml::gemmini::MatMul rounded_operation(rounded_args);
    if (rounded_operation.run_dense().status != ggml::gemmini::MatMulStatus::success ||
        setenv("GGML_GEMMINI_METAL_ACTIVATION_FP16", "1", 1) != 0 ||
        !ggml_metal_cpu_exact_float(1, 7, 32, fp16_input.data(), matrix_weights.data(), actual.data()) ||
        std::memcmp(expected.data(), actual.data(), 7 * sizeof(float)) != 0) {
        std::fprintf(stderr, "FP16 activation rounding vs independent _Float16 oracle failed\n");
        return 1;
    }
    unsetenv("GGML_GEMMINI_METAL_ACTIVATION_FP16");
    for (ggml_type type : {GGML_TYPE_Q4_0, GGML_TYPE_Q8_0, GGML_TYPE_Q6_K}) {
        ggml_context * context = ggml_init({1<<20,nullptr,false});
        if (!context) return 1;
        constexpr size_t m=8, n=12, k=256;
        ggml_tensor * weight = ggml_new_tensor_2d(context,type,k,n);
        std::vector<float> original(n*k), decoded(n*k), input(m*k), cpu(m*n), gpu(m*n);
        for (size_t i=0; i<input.size(); ++i) input[i] = float(int(i%37)-18)/32.0f;
        for (size_t i=0; i<original.size(); ++i) original[i] = float(int(i%53)-26)/16.0f;
        for (int repeat=0; repeat<3; ++repeat) {
            if (repeat == 2) for (float & x : original) x += 0.375f;
            if (ggml_quantize_chunk(type,original.data(),weight->data,0,n,k,nullptr) != ggml_nbytes(weight)) return 1;
            ggml_get_type_traits(type)->to_float(weight->data,decoded.data(),n*k);
            ggml_gemmini_args_t args{};
            args.I=m; args.J=n; args.K=k; args.sA=k; args.sB=k;
            args.A_fp32=input.data(); args.B_fp32=decoded.data();
            args.f_out=cpu.data(); args.stride_f_out=n; args.transpose_B=true;
            args.tiled_matmul_type=static_cast<tiled_matmul_type_t>(2);
            ggml::gemmini::MatMul operation(args);
            if (operation.run_dense().status != ggml::gemmini::MatMulStatus::success ||
                !ggml_metal_cpu_exact_float_quantized(m,n,k,input.data(),weight,gpu.data()) ||
                std::memcmp(cpu.data(),gpu.data(),cpu.size()*sizeof(float)) != 0) {
                std::fprintf(stderr,"Native weight reuse/invalidation failed: type=%d repeat=%d\n",int(type),repeat);
                ggml_free(context);
                return 1;
            }
        }
        ggml_free(context);
    }
    std::printf("PASS CPU FLOAT vs Metal and FP16 activation rounding; %zu original bitwise outputs\n", compared);
    return 0;
}
