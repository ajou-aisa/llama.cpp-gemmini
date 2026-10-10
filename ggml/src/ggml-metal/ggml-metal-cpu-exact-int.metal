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
    constant uint & stored [[buffer(5)]],
    uint index [[thread_position_in_grid]]) {
    const uint blocks=shape.z/shape.w, row_groups=(shape.x+3)/4;
    if (index >= row_groups*shape.y*blocks) return;
    const uint column=index%shape.y, row=((index/shape.y)%row_groups)*4, block=index/(row_groups*shape.y), begin=block*shape.w;
    const uint block_bytes=stored ? (bits==4 ? 18 : 34) : (bits==4 ? 24 : 40);
    device const uchar * weight=w+(ulong(column)*(shape.z/32)+begin/32)*block_bytes+(stored ? 2 : 0);
    int4 dot=0;
    for (uint k=begin;k<begin+shape.w;++k) {
        const uint local=k%32;
        const int code=bits==4 ? int((weight[local%16]>>(local<16 ? 0 : 4))&15)-8 : int(as_type<char>(weight[local]));
        for (uint r=0;r<4 && row+r<shape.x;++r) dot[r]+=a[ulong(row+r)*shape.z+k]*code;
    }
    for (uint r=0;r<4 && row+r<shape.x;++r) dots[(block*shape.x+row+r)*shape.y+column]=dot[r];
}

kernel void cpu_exact_q6_blocks(device const int * a [[buffer(0)]],
    device const uchar * weights [[buffer(1)]], device int * dots [[buffer(2)]],
    constant uint4 & shape [[buffer(3)]], uint index [[thread_position_in_grid]]) {
    const uint blocks=shape.z/16, row_groups=(shape.x+3)/4;
    if (index >= row_groups*shape.y*blocks) return;
    const uint column=index%shape.y, row=((index/shape.y)%row_groups)*4, block=index/(row_groups*shape.y), begin=block*16;
    device const uchar * w=weights+(ulong(column)*(shape.z/256)+begin/256)*210;
    int4 dot=0;
    for (uint i=0;i<16;++i) {
        const uint k=begin+i, local=k%256, half_block=local/128, quarter=(local%128)/32, l=local%32;
        const uint low=w[half_block*64+(quarter%2)*32+l];
        const uint high=w[128+half_block*32+l];
        const int code=int(((low>>(quarter>=2 ? 4 : 0))&15)|(((high>>(2*quarter))&3)<<4))-32;
        for (uint r=0;r<4 && row+r<shape.x;++r) dot[r]+=a[ulong(row+r)*shape.z+k]*code;
    }
    for (uint r=0;r<4 && row+r<shape.x;++r) dots[(block*shape.x+row+r)*shape.y+column]=dot[r];
}
