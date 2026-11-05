#pragma once
#include <cmath>
#include <cstdint>

#ifdef __CUDACC__
#define NVFP4_HD __host__ __device__
#else
#define NVFP4_HD
#endif

// NVIDIA Model-Optimizer's E2M1 codebook; ties choose the even encoded ordinal.
NVFP4_HD inline uint8_t nvfp4_code(float x) {
    const float bounds[7] = {0.25f, 0.75f, 1.25f, 1.75f, 2.5f, 3.5f, 5.0f};
    float a = fabsf(x);
    unsigned q = 0;
    while (q < 7 && (a > bounds[q] || (a == bounds[q] && (q & 1)))) ++q;
    return uint8_t(q | (x < 0 ? 8 : 0));
}
NVFP4_HD inline float nvfp4_value(uint8_t code) {
    const float values[8] = {0, 0.5f, 1, 1.5f, 2, 3, 4, 6};
    return (code & 8) ? -values[code & 7] : values[code & 7];
}
NVFP4_HD inline float nvfp4_e4m3_value(unsigned code) {
    unsigned e = code >> 3, m = code & 7;
    return e ? ldexpf(1.0f + float(m) / 8.0f, int(e) - 7)
             : ldexpf(float(m), -9);
}
// Positive E4M3FN block scales, including subnormals. ModelOpt clamps before cast.
NVFP4_HD inline uint8_t nvfp4_scale_code(float x) {
    x = fmaxf(0x1p-9f, fminf(448.0f, x));
    unsigned lo = 0, hi = 126;
    while (lo < hi) {
        unsigned mid = (lo + hi) / 2;
        if (nvfp4_e4m3_value(mid) < x) lo = mid + 1; else hi = mid;
    }
    if (!lo) return 0;
    float below = x - nvfp4_e4m3_value(lo - 1);
    float above = nvfp4_e4m3_value(lo) - x;
    return uint8_t(below < above || (below == above && (lo & 1)) ? lo - 1 : lo);
}
// Each arithmetic step rounds to FP32, matching the official tensor operations.
NVFP4_HD inline float nvfp4_div(float a, float b) {
#ifdef __CUDA_ARCH__
    return __fdiv_rn(a, b);
#else
    return a / b;
#endif
}
NVFP4_HD inline float nvfp4_block_scale(float block_amax, float tensor_scale) {
    float raw = nvfp4_div(block_amax, 6.0f * tensor_scale);
    if (raw == 0) raw = 1.0f;
    return nvfp4_e4m3_value(nvfp4_scale_code(raw));
}
