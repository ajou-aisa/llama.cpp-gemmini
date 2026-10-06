#include "ggml-metal-cpu-exact.h"
#include "ggml-gemmini-matmul.hpp"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static uint32_t random_bits(uint32_t & state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

int main() {
    const size_t shapes[][3] = {{1,7,32}, {4,4,32}, {5,7,64}, {4,5,96}, {8,8,768}, {17,19,2048}};
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
        ggml_metal_cpu_exact_gpu_calls() != initial_calls+42 ||
        ggml_metal_cpu_exact_software_calls() != initial_software_calls+18) {
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
    std::printf("PASS actual CPU FLOAT helper vs Metal: %zu bitwise outputs, 42 GPU calls (24 hardware, 18 software); invalid K/nonfinite transactional rejection\n",compared);
    return 0;
}
