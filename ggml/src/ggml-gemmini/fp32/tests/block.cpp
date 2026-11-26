#include "test-util.hpp"
#include <array>

int main(int argc, char ** argv) {
    start_test_runtime(argc, argv);
    uint32_t seed = 0x3591c4d7;
    uint64_t checks = 0;
    // Each requested Qwen weight shape, plus array and SDK tile tails/crossings.
    const std::array<size_t, 3> cases[] = {
        {9,896,896}, {9,128,896}, {9,4864,896}, {9,896,4864}, {1,151936,896},
        {7,13,15}, {9,17,9}, {97,103,257},
    };
    for (const auto & shape : cases) {
        const size_t m = shape[0], n = shape[1], k = shape[2];
        std::vector<float> a(m*k), b(k*n), c(m*n);
        for (auto & x : a) x = golden_fp32::sample(seed);
        for (auto & x : b) x = golden_fp32::sample(seed);
        fp32::matmul(a.data(), b.data(), c.data(), m, n, k);
        const auto s = golden_fp32::compare(a.data(), b.data(), c.data(), m, n, k, unsigned(fp32::get_order()));
        const std::string name = std::to_string(m)+"x"+std::to_string(n)+"x"+std::to_string(k);
        golden_fp32::report(name.c_str(), s); checks += s.values;
    }
    // Real buffer write/free hooks must invalidate the first-use F16 weight cache.
    constexpr size_t k = 896, n = 128;
    std::vector<uint16_t> half(k*n, 0x3800);
    std::vector<float> a(k, 0.25f), c(n), b(k*n, 0.5f);
    auto * buffer = ggml_backend_buft_alloc_buffer(ggml::gemmini::gemmini_buffer_type(), half.size()*2);
    golden_fp32::check(buffer != nullptr, "weight buffer allocation");
    ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    auto w = tensor(GGML_TYPE_F16, ggml_backend_buffer_get_base(buffer), k, n);
    w.buffer = buffer;
    std::strcpy(w.name, "blk.0.attn_k.weight");
    auto activation = tensor(GGML_TYPE_F32, a.data(), k, 1);
    auto output = tensor(GGML_TYPE_F32, c.data(), n, 1);
    output.op = GGML_OP_MUL_MAT; output.src[0] = &w; output.src[1] = &activation;
    ggml_backend_tensor_set(&w, half.data(), 0, half.size()*2);
    fp32::execute(&output);
    const auto first = fp32::counters().conversions;
    fp32::execute(&output);
    golden_fp32::check(fp32::counters().conversions == first, "weight conversion cached");
    const uint16_t changed = 0x3c00;
    ggml_backend_tensor_set(&w, &changed, 0, 2);
    b[0] = 1.0f;
    fp32::execute(&output);
    golden_fp32::check(fp32::counters().conversions == first + 1, "set invalidates cached weight");
    golden_fp32::report("cache-set", golden_fp32::compare(a.data(), b.data(), c.data(), 1,n,k,unsigned(fp32::get_order())));
    ggml_backend_tensor_memset(&w, 0, 0, half.size()*2);
    fp32::execute(&output);
    golden_fp32::check(c[0] == 0 && fp32::counters().conversions == first + 2, "memset invalidates cache");
    ggml_backend_buffer_clear(buffer, 0);
    fp32::execute(&output);
    golden_fp32::check(fp32::counters().conversions == first + 3, "clear invalidates cache");
    ggml_backend_buffer_free(buffer);
    golden_fp32::check(fp32::counters().cached_bytes == 0, "free invalidates cache");
    finish_test_runtime();
    std::printf("gemmini-fp32-block-test: PASS checks=%llu order=%s\n", (unsigned long long)checks, fp32::order_name());
}
