#include <metal_stdlib>
using namespace metal;

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
