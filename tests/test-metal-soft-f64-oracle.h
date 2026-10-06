#ifndef TEST_METAL_SOFT_F64_ORACLE_H
#define TEST_METAL_SOFT_F64_ORACLE_H

#include <cstddef>
#include <cstdint>

struct F64Input {
    int64_t i;
    uint64_t a, b;
    uint32_t s, t, ma, mb;
};

struct F64Output {
    uint64_t integer, scale, product, sum;
    uint32_t result, merge;
};

static_assert(sizeof(F64Input) == 40, "input must match the Metal structure");
static_assert(sizeof(F64Output) == 40, "output must match the Metal structure");

void oracle_f64(const F64Input *, F64Output *, size_t);

#endif
