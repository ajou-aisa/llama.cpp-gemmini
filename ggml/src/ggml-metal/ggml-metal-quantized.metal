#include <metal_stdlib>
using namespace metal;

struct mq_parameters {
    ulong m, n, k, blocks, fragments;
    uint bits, dim, mode, trace;
    uint has_residual, reserved;
};
struct mq_request_parameters {
    ulong m, n, k, runs, source_begin, source_count;
    uint bits, dim;
};
struct mq_run { ulong begin, count; };
struct mq_row { uint lane, source; };

inline int mq_sat32(long value) {
    return value > 2147483647L ? 2147483647 :
           value < (-2147483647L - 1) ? (-2147483647 - 1) : int(value);
}
inline int mq_scu(int value, uint carrier) {
    if (!value || carrier == 0x80000000u) { return 0; }
    if (carrier >= 32) { return value < 0 ? (-2147483647 - 1) : 2147483647; }
    return mq_sat32(long(value) * (1L << carrier));
}
inline bool mq_finite32(uint value) { return (value & 0x7f800000u) != 0x7f800000u; }
inline bool mq_finite64(mq_f64 value) { return (value & 0x7ff0000000000000ul) != 0x7ff0000000000000ul; }
inline mq_f64 mq_scaled(long value, uint weight_scale, uint activation_scale) {
    return mq_f64_mul(mq_f64_mul(mq_f64_from_i64(value), mq_f64_from_f32_bits(weight_scale)),
                      mq_f64_from_f32_bits(activation_scale));
}
inline int mq_dot4_codes(device const char * a, device const char * w) {
    const int4 av = int4(*reinterpret_cast<device const packed_char4 *>(a));
    const int4 wv = int4(*reinterpret_cast<device const packed_char4 *>(w));
    const int4 products = av * wv;
    return products.x + products.y + products.z + products.w;
}

kernel void mq_dense(
    constant mq_parameters & p [[buffer(0)]],
    device const char * a [[buffer(1)]], device const char * w [[buffer(2)]],
    device const uint * ascale [[buffer(3)]], device const uint * wscale [[buffer(4)]],
    device const uint * carriers [[buffer(5)]], device const uint * columns [[buffer(6)]],
    device const int * residual [[buffer(7)]], device uint * output [[buffer(8)]],
    device int * raw_trace [[buffer(9)]], device int * scu_trace [[buffer(10)]],
    device int * integer_trace [[buffer(11)]], device long * correction_trace [[buffer(12)]],
    device atomic_uint * status [[buffer(13)]], uint index [[thread_position_in_grid]]) {
    if (ulong(index) >= p.m * p.n) { return; }
    ulong row = index / p.n, col = index % p.n;
    uint final_bits;
    if (p.mode == 0) {
        mq_f64 dense = mq_f64_from_i64(0), correction = dense;
        long raw_correction = 0;
        for (ulong block = 0; block < p.blocks; ++block) {
            int dot = 0;
            long correction_dot = 0;
            ulong end = min((block + 1) * 32, p.k);
            for (ulong k = block * 32; k < end; ++k) {
                dot += int(a[row * p.k + k]) * int(w[col * p.k + k]);
                if (p.has_residual) { correction_dot += long(residual[row * p.k + k]) * long(w[col * p.k + k]); }
            }
            uint ws = wscale[col * p.blocks + block], as = ascale[row * p.blocks + block];
            dense = mq_f64_add(dense, mq_scaled(dot, ws, as));
            if (p.has_residual) { correction = mq_f64_add(correction, mq_scaled(correction_dot, ws, as)); }
            raw_correction += correction_dot;
            if (p.trace & 1) { raw_trace[ulong(index) * p.fragments + block] = dot; }
            if (p.trace & 2) { scu_trace[ulong(index) * p.fragments + block] = dot; }
        }
        final_bits = mq_f64_to_f32_bits(dense);
        if (p.has_residual) { final_bits = mq_f32_add_bits(final_bits, mq_f64_to_f32_bits(correction)); }
        if (!mq_finite64(dense) || !mq_finite64(correction)) { atomic_store_explicit(status, 1u, memory_order_relaxed); }
        if (p.trace & 4) { integer_trace[index] = 0; }
        if (p.trace & 8) { correction_trace[index] = raw_correction; }
    } else {
        int acc = 0;
        ulong fragment_width = min(p.dim, 32u);
        for (ulong f = 0; f < p.fragments; ++f) {
            ulong begin = f * fragment_width, end = min(begin + fragment_width, p.k);
            int dot = 0;
            ulong k = begin;
            for (; k + 4 <= end; k += 4) { dot += mq_dot4_codes(a + row * p.k + k, w + col * p.k + k); }
            for (; k < end; ++k) { dot += int(a[row * p.k + k]) * int(w[col * p.k + k]); }
            int value = mq_scu(dot, carriers[(begin / 32) * p.n + col]);
            acc = mq_sat32(long(acc) + long(value));
            if (p.trace & 1) { raw_trace[ulong(index) * p.fragments + f] = dot; }
            if (p.trace & 2) { scu_trace[ulong(index) * p.fragments + f] = value; }
        }
        mq_f64 scaled = mq_scaled(acc, columns[col], ascale[row]);
        final_bits = mq_f64_to_f32_bits(scaled);
        if (!mq_finite64(scaled)) { atomic_store_explicit(status, 1u, memory_order_relaxed); }
        if (p.trace & 4) { integer_trace[index] = acc; }
        if (p.trace & 8) { correction_trace[index] = 0; }
    }
    if (!mq_finite32(final_bits)) { atomic_store_explicit(status, 1u, memory_order_relaxed); }
    output[index] = final_bits;
}

kernel void mq_residual(
    constant mq_request_parameters & p [[buffer(0)]],
    device const char * a [[buffer(1)]], device const int * w [[buffer(2)]],
    device const uint * carriers [[buffer(3)]], device const mq_run * runs [[buffer(4)]],
    device int * output [[buffer(5)]], uint index [[thread_position_in_grid]]) {
    if (ulong(index) >= p.m * p.n) { return; }
    ulong row = index / p.n, col = index % p.n;
    int acc = 0;
    for (ulong run = 0; run < p.runs; ++run) {
        ulong end = runs[run].begin + runs[run].count;
        for (ulong begin = runs[run].begin; begin < end; begin += p.dim) {
            int dot = 0;
            const ulong fragment_end = min(begin + p.dim, end);
            ulong k = begin;
            if (p.dim == 16) {
                for (; k + 4 <= fragment_end; k += 4) {
                    const int4 av = int4(*reinterpret_cast<device const packed_char4 *>(a + row * p.k + k));
                    const int4 wv = int4(w[k * p.n + col], w[(k + 1) * p.n + col],
                                        w[(k + 2) * p.n + col], w[(k + 3) * p.n + col]);
                    const int4 products = av * wv;
                    dot += products.x + products.y + products.z + products.w;
                }
            }
            for (; k < fragment_end; ++k) { dot += int(a[row * p.k + k]) * w[k * p.n + col]; }
            acc = mq_sat32(long(acc) + long(mq_scu(dot, carriers[run * p.n + col])));
        }
    }
    output[index] = acc;
}

kernel void mq_merge(
    constant mq_request_parameters & p [[buffer(0)]], device const int * lanes [[buffer(1)]],
    device const mq_row * rows [[buffer(2)]], device const uint * columns [[buffer(3)]],
    device const uint * ascale [[buffer(4)]], device uint * output [[buffer(5)]],
    device long * correction [[buffer(6)]], device atomic_uint * status [[buffer(7)]],
    constant uint & trace [[buffer(8)]], uint index [[thread_position_in_grid]]) {
    if (ulong(index) >= p.source_count * p.n) { return; }
    ulong source = index / p.n, col = index % p.n;
    // Signed 128-bit accumulation matches the checked CPU oracle, including
    // cancellation where an intermediate radix sum exceeds signed 64 bits.
    ulong lo = 0, hi = 0;
    for (ulong row = 0; row < p.m; ++row) {
        if (rows[row].source != source) { continue; }
        long lane = lanes[row * p.n + col];
        uint shift = rows[row].lane * p.bits;
        ulong term_lo = as_type<ulong>(lane) << shift;
        ulong term_hi = shift ? (as_type<ulong>(lane) >> (64 - shift)) : 0;
        if (lane < 0) { term_hi |= (~0ul) << shift; }
        ulong next = lo + term_lo;
        hi += term_hi + ulong(next < lo);
        lo = next;
    }
    bool negative = (lo >> 63) != 0;
    if (hi != (negative ? ~0ul : 0ul)) { atomic_store_explicit(status, 2u, memory_order_relaxed); return; }
    long value = as_type<long>(lo);
    ulong global_index = (p.source_begin + source) * p.n + col;
    mq_f64 scaled = mq_scaled(value, columns[col], ascale[p.source_begin + source]);
    uint delta = mq_f64_to_f32_bits(scaled);
    uint merged = mq_f32_add_bits(output[global_index], delta);
    if (!mq_finite64(scaled) || !mq_finite32(delta) || !mq_finite32(merged)) {
        atomic_store_explicit(status, 1u, memory_order_relaxed); return;
    }
    output[global_index] = merged;
    if (trace & 8) { correction[global_index] = value; }
}
