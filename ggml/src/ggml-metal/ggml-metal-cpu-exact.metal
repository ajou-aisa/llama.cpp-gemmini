#include <metal_stdlib>
using namespace metal;

struct cpu_exact_shape { ulong m, n, k, software; };

ulong fp32_shift_jam(ulong value, uint shift) {
    if (shift == 0) { return value; }
    if (shift >= 64) { return value != 0; }
    return (value >> shift) | ((value << (64-shift)) != 0);
}

uint fp32_round(bool negative, ulong magnitude, int exponent) {
    const uint sign = negative ? 0x80000000u : 0;
    if (!magnitude) { return sign; }
    const int top = 63-int(clz(magnitude));
    int real_exponent = exponent-62+top;
    if (real_exponent > 127) { return sign | 0x7f800000u; }
    const int shift = real_exponent < -126 ? -exponent-87 : top-23;
    ulong rounded;
    if (shift <= 0) {
        rounded = magnitude << uint(-shift);
    } else if (shift > 64) {
        rounded = 0;
    } else {
        const ulong halfway = 1ul << uint(shift-1);
        const ulong remainder = magnitude & (halfway | (halfway-1));
        rounded = shift == 64 ? 0 : magnitude >> uint(shift);
        rounded += remainder > halfway || (remainder == halfway && (rounded & 1));
    }
    if (real_exponent < -126) { return sign | uint(rounded); }
    if (rounded >= 0x1000000ul) { rounded >>= 1; ++real_exponent; }
    if (real_exponent > 127) { return sign | 0x7f800000u; }
    return sign | (uint(real_exponent+127) << 23) | (uint(rounded) & 0x7fffffu);
}

uint fp32_fma(uint a, uint b, uint c) {
    const uint ae=(a>>23)&255, be=(b>>23)&255, ce=(c>>23)&255;
    const bool ps = ((a^b)>>31) != 0, cs = (c>>31) != 0;
    if ((ae == 255 && (a&0x7fffff)) || (be == 255 && (b&0x7fffff)) ||
        (ce == 255 && (c&0x7fffff))) { return 0x7fc00000u; }
    if (ae == 255 || be == 255) {
        if (!(a&0x7fffffff) || !(b&0x7fffffff) || (ce == 255 && ps != cs)) { return 0x7fc00000u; }
        return (ps ? 0x80000000u : 0) | 0x7f800000u;
    }
    if (ce == 255) { return c; }
    const ulong am=(a&0x7fffff) | (ae ? 0x800000u : 0u);
    const ulong bm=(b&0x7fffff) | (be ? 0x800000u : 0u);
    const ulong cm=(c&0x7fffff) | (ce ? 0x800000u : 0u);
    ulong product=am*bm, addend=cm;
    if (!product && !addend) { return ps && cs ? 0x80000000u : 0u; }
    int pe=(ae ? int(ae)-150 : -149)+(be ? int(be)-150 : -149);
    int cexp=ce ? int(ce)-150 : -149;
    if (product) {
        const int top=63-int(clz(product));
        pe += top;
        product <<= uint(62-top);
    }
    if (addend) {
        const int top=63-int(clz(addend));
        cexp += top;
        addend <<= uint(62-top);
    }
    if (!product) { return c; }
    if (!addend) { return fp32_round(ps,product,pe); }
    const int exponent=max(pe,cexp);
    product=fp32_shift_jam(product,uint(exponent-pe));
    addend=fp32_shift_jam(addend,uint(exponent-cexp));
    if (ps == cs) { return fp32_round(ps,product+addend,exponent); }
    if (product == addend) { return 0; }
    return product > addend ? fp32_round(ps,product-addend,exponent)
                            : fp32_round(cs,addend-product,exponent);
}

uint fp32_add(uint a, uint b) { return fp32_fma(a,0x3f800000u,b); }

kernel void cpu_exact_float(device const float * a [[buffer(0)]],
                            device const float * b [[buffer(1)]],
                            device float * output [[buffer(2)]],
                            constant cpu_exact_shape & s [[buffer(3)]],
                            uint2 pos [[thread_position_in_grid]]) {
    const ulong row = pos.y, col = pos.x;
    if (row >= s.m || col >= s.n) { return; }
    const ulong ai = row*s.k, bi = col*s.k;
    if (s.software) {
        uint lanes[16] = {};
        const uint count=row < (s.m & ~3ul) && col < (s.n & ~3ul) ? 4 : 16;
        for (ulong k=0; k<s.k; k+=count) {
            for (uint lane=0; lane<count; ++lane) {
                lanes[lane]=fp32_fma(as_type<uint>(a[ai+k+lane]),as_type<uint>(b[bi+k+lane]),lanes[lane]);
            }
        }
        if (count == 16) {
            for (uint lane=0; lane<4; ++lane) {
                lanes[lane]=fp32_add(fp32_add(lanes[lane+12],lanes[lane+8]),
                                    fp32_add(lanes[lane+4],lanes[lane]));
            }
        }
        output[row*s.n+col]=as_type<float>(fp32_add(fp32_add(lanes[0],lanes[1]),fp32_add(lanes[2],lanes[3])));
        return;
    }
    float4 v0 = 0.0f, v1 = 0.0f, v2 = 0.0f, v3 = 0.0f;
    if (row < (s.m & ~3ul) && col < (s.n & ~3ul)) {
        for (ulong k = 0; k < s.k; k += 4) {
            v0 = fma(float4(a[ai+k], a[ai+k+1], a[ai+k+2], a[ai+k+3]),
                     float4(b[bi+k], b[bi+k+1], b[bi+k+2], b[bi+k+3]), v0);
        }
    } else {
        for (ulong k = 0; k < s.k; k += 16) {
            v0 = fma(float4(a[ai+k], a[ai+k+1], a[ai+k+2], a[ai+k+3]),
                     float4(b[bi+k], b[bi+k+1], b[bi+k+2], b[bi+k+3]), v0);
            v1 = fma(float4(a[ai+k+4], a[ai+k+5], a[ai+k+6], a[ai+k+7]),
                     float4(b[bi+k+4], b[bi+k+5], b[bi+k+6], b[bi+k+7]), v1);
            v2 = fma(float4(a[ai+k+8], a[ai+k+9], a[ai+k+10], a[ai+k+11]),
                     float4(b[bi+k+8], b[bi+k+9], b[bi+k+10], b[bi+k+11]), v2);
            v3 = fma(float4(a[ai+k+12], a[ai+k+13], a[ai+k+14], a[ai+k+15]),
                     float4(b[bi+k+12], b[bi+k+13], b[bi+k+14], b[bi+k+15]), v3);
        }
        v0 = (v3 + v2) + (v1 + v0);
    }
    output[row*s.n+col] = (v0.x+v0.y)+(v0.z+v0.w);
}
