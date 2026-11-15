#include "test-util.hpp"

int main(int argc, char ** argv) {
    start_test_runtime(argc, argv);
    uint32_t seed = 0x38519;
    uint64_t checks = 0;
    for (const auto role : {GGML_GEMMINI_ATTENTION_QK, GGML_GEMMINI_ATTENTION_PV}) {
        for (const size_t sequence : {size_t(13), size_t(64), size_t(513)}) {
            const size_t k = role == GGML_GEMMINI_ATTENTION_QK ? 64 : sequence;
            const size_t n = role == GGML_GEMMINI_ATTENTION_QK ? sequence : 64, m = 9;
            std::vector<uint16_t> half(k*n*2);
            std::vector<float> activation(m*k*14), output(m*n*14);
            for (auto & h : half) {
                const float x = golden_fp32::sample(seed);
                h = ggml_fp32_to_fp16(x);
            }
            for (auto & x : activation) x = golden_fp32::sample(seed);
            auto a = tensor(GGML_TYPE_F16, half.data(), k, n, 2);
            auto b = tensor(GGML_TYPE_F32, activation.data(), k, m, 14);
            auto c = tensor(GGML_TYPE_F32, output.data(), n, m, 14);
            c.op = GGML_OP_MUL_MAT; c.src[0] = &a; c.src[1] = &b;
            ggml_gemmini_set_attention_role(&c, role);
            fp32::execute(&c);
            for (size_t head = 0; head < 14; ++head) {
                std::vector<float> weights(k*n);
                for (size_t col = 0; col < n; ++col)
                    for (size_t z = 0; z < k; ++z)
                        weights[z*n+col] = golden_fp32::widen(half[(head/7)*k*n+col*k+z]);
                const auto stats = golden_fp32::compare(activation.data()+head*m*k, weights.data(),
                    output.data()+head*m*n, m,n,k,unsigned(fp32::get_order()));
                golden_fp32::report(role == GGML_GEMMINI_ATTENTION_QK ? "qk" : "pv", stats);
                checks += stats.values;
            }
        }
    }
    finish_test_runtime();
    std::printf("gemmini-fp32-attention-test: PASS checks=%llu order=%s\n", (unsigned long long)checks, fp32::order_name());
}
