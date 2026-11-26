#pragma once
#include <cstdint>

namespace ggml::gemmini::fp32 {
// Little-endian IEEE binary32 files. For each head: A[M,K], optional B[K,N], C[M,N].
// Linear B is deliberately omitted: replay must read the named tensor from GGUF.
struct DumpHeader {
    uint64_t magic = 0x3144383233504647ull; // GFP328D1
    uint64_t graph = 0, order = 0, role = 0, m = 0, n = 0, k = 0, heads = 0;
    char weight[128]{};
};
static_assert(sizeof(DumpHeader) == 192);
}
