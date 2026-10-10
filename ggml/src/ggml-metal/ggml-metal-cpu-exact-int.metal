#include <metal_stdlib>
using namespace metal;

struct residual_event { uint k; int value; };
struct residual_result { long value; uint event; uint error; };

kernel void cpu_exact_hp1_residual(device const uchar * weights [[buffer(0)]],
    device const uint * rows [[buffer(1)]], device const residual_event * events [[buffer(2)]],
    device residual_result * output [[buffer(3)]], constant uint4 & shape [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= shape.x * shape.y) return;
    const uint row = index / shape.y, column = index % shape.y;
    const uint bits = shape.w, block_bytes = bits == 4 ? 24 : 40, carrier_offset = bits == 4 ? 16 : 32;
    const long hi = 0x7fffffffffffffffl, lo = as_type<long>(0x8000000000000000ul);
    long accumulator = 0;
    for (uint begin = rows[row]; begin < rows[row + 1];) {
        const uint block = events[begin].k / 32;
        device const uchar * w = weights + (ulong(column) * (shape.z / 32) + block) * block_bytes;
        long raw = 0;
        uint end = begin;
        while (end < rows[row + 1] && events[end].k / 32 == block) {
            const uint local = events[end].k % 32;
            const int code = bits == 4 ? int((w[local % 16] >> (local < 16 ? 0 : 4)) & 15) - 8
                                       : int(as_type<char>(w[local]));
            raw += long(events[end].value) * code;
            ++end;
        }
        const short exponent = as_type<short>(ushort(uint(w[carrier_offset]) | (uint(w[carrier_offset + 1]) << 8)));
        const uint scale_bits = uint(w[carrier_offset + 4]) | (uint(w[carrier_offset + 5]) << 8) |
                                (uint(w[carrier_offset + 6]) << 16) | (uint(w[carrier_offset + 7]) << 24);
        if (w[carrier_offset + 2] || w[carrier_offset + 3] || (scale_bits & 0x7f800000u) == 0x7f800000u ||
            (exponent < 0 && exponent != -32768)) {
            output[index] = {0, begin, 1}; return;
        }
        if (exponent >= 63) { output[index] = {0, begin, 2}; return; }
        long scaled = 0;
        if (exponent != -32768) {
            const bool negative = raw < 0;
            const ulong magnitude = negative ? 0ul - ulong(raw) : ulong(raw);
            const ulong limit = negative ? 0x8000000000000000ul : 0x7ffffffffffffffful;
            if (magnitude > (limit >> uint(exponent))) { output[index] = {0, begin, 2}; return; }
            const ulong shifted = magnitude << uint(exponent);
            scaled = as_type<long>(negative ? 0ul - shifted : shifted);
        }
        if ((scaled > 0 && accumulator > hi - scaled) || (scaled < 0 && accumulator < lo - scaled)) {
            output[index] = {0, begin, 2}; return;
        }
        accumulator += scaled;
        begin = end;
    }
    output[index] = {accumulator, 0, 0};
}

kernel void cpu_exact_int_blocks(device const int * a [[buffer(0)]],
    device const int * w [[buffer(1)]], device int * dots [[buffer(2)]],
    constant uint4 & shape [[buffer(3)]], uint index [[thread_position_in_grid]]) {
    const uint rows = shape.x, columns = shape.y, k = shape.z, block_size = shape.w;
    const uint blocks = (k + block_size - 1) / block_size;
    if (index >= rows * columns * blocks) return;
    const uint column = index % columns;
    const uint row = (index / columns) % rows;
    const uint block = index / (rows * columns);
    const uint begin = block * block_size;
    int dot = 0;
    for (uint offset = begin; offset < min(begin + block_size, k); ++offset) {
        dot += a[row * k + offset] * w[column * k + offset];
    }
    dots[index] = dot;
}

kernel void cpu_exact_hp1_blocks(device const int * a [[buffer(0)]],
    device const uchar * w [[buffer(1)]], device int * dots [[buffer(2)]],
    constant uint4 & shape [[buffer(3)]], constant uint & bits [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
    const uint blocks=shape.z/shape.w;
    if (index >= shape.x*shape.y*blocks) return;
    const uint column=index%shape.y, row=(index/shape.y)%shape.x, begin=(index/(shape.x*shape.y))*shape.w;
    const uint block_bytes=bits==4 ? 24 : 40;
    device const uchar * weight=w+(ulong(column)*(shape.z/32)+begin/32)*block_bytes;
    int dot=0;
    for (uint k=begin;k<begin+shape.w;++k) {
        const uint local=k%32;
        const int code=bits==4 ? int((weight[local%16]>>(local<16 ? 0 : 4))&15)-8 : int(as_type<char>(weight[local]));
        dot+=a[ulong(row)*shape.z+k]*code;
    }
    dots[index]=dot;
}
