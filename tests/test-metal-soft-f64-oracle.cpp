#include "test-metal-soft-f64-oracle.h"

#include <cstring>

template<class T, class U>
static T bit_copy(U value) {
    T result;
    static_assert(sizeof(T) == sizeof(U), "bit copy requires equal widths");
    memcpy(&result, &value, sizeof(T));
    return result;
}

// This translation unit deliberately excludes the software arithmetic helper.
// Volatile intermediates preserve the reference's separate rounding points.
void oracle_f64(const F64Input *input, F64Output *output, size_t count) {
    for (size_t n = 0; n < count; n++) {
        const auto &x = input[n];
        auto &y = output[n];
        volatile double integer = static_cast<double>(x.i);
        volatile float sf = bit_copy<float>(x.s), tf = bit_copy<float>(x.t);
        volatile double scale = static_cast<double>(sf);
        volatile double a = bit_copy<double>(x.a), b = bit_copy<double>(x.b);
        volatile double product = a * b, sum = a + b;
        volatile double first = integer * scale;
        volatile double second = first * static_cast<double>(tf);
        volatile float result = static_cast<float>(second);
        volatile float ma = bit_copy<float>(x.ma), mb = bit_copy<float>(x.mb);
        volatile float merge = ma + mb;
        y.integer = bit_copy<uint64_t>(double(integer));
        y.scale = bit_copy<uint64_t>(double(scale));
        y.product = bit_copy<uint64_t>(double(product));
        y.sum = bit_copy<uint64_t>(double(sum));
        y.result = bit_copy<uint32_t>(float(result));
        y.merge = bit_copy<uint32_t>(float(merge));
    }
}
