#include "nvfp4-numeric.h"
#include <algorithm>
#include <limits>

extern "C" unsigned nvfp4_ref_code(float x) { return nvfp4_code(x); }
extern "C" unsigned nvfp4_ref_scale_code(float x) { return nvfp4_scale_code(x); }
extern "C" float nvfp4_ref_scale_value(unsigned x) { return nvfp4_e4m3_value(x); }
extern "C" int nvfp4_ref_quantize(const float * x, unsigned n, uint8_t * q,
                                  uint8_t * scales, float * global, float * dequant) {
    if (!n || n % 16) return 1;
    float amax = 0;
    for (unsigned i = 0; i < n; ++i) {
        if (!std::isfinite(x[i])) return 2;
        amax = std::max(amax, std::fabs(x[i]));
    }
    // Defined zero-tensor extension: avoid the official dynamic helper's 0/0.
    *global = amax ? amax / 2688.0f : 1.0f;
    if (!std::isfinite(*global) || *global == 0) return 3;
    for (unsigned b = 0; b < n / 16; ++b) {
        float local = 0;
        for (unsigned j = 0; j < 16; ++j) local = std::max(local, std::fabs(x[b * 16 + j]));
        float s = nvfp4_block_scale(local, *global);
        scales[b] = nvfp4_scale_code(s);
        float d = s * *global;
        for (unsigned j = 0; j < 16; ++j) {
            unsigned i = b * 16 + j;
            q[i] = nvfp4_code(x[i] / d);
            dequant[i] = nvfp4_value(q[i]) * d;
        }
    }
    return 0;
}
