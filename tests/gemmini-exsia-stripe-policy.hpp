// Shared environment scope and malformed-policy boundary for the ExSIA test owner.
struct RestorePolicy {
    const char * key = "GGML_GEMMINI_EXSIA_DIRECT_STRIPE";
    bool present = std::getenv(key) != nullptr;
    std::string value = present ? std::getenv(key) : "";
    bool set(const char * policy) {
#if defined(_WIN32)
        return _putenv_s(key, policy ? policy : "") == 0;
#else
        return policy ? setenv(key, policy, 1) == 0 : unsetenv(key) == 0;
#endif
    }
    ~RestorePolicy() { set(present ? value.c_str() : nullptr); }
};

bool test_direct_stripe_policy_boundary() {
    using namespace quants::act::exsia;
    RestorePolicy policy;
    // Reject malformed policies at the same actual run boundary, then prove
    // a reused producer recovers when the policy is corrected.
    std::vector<float> invalid_source(32, 1);
    ggml_tensor invalid_tensor{};
    invalid_tensor.type = GGML_TYPE_F32;
    invalid_tensor.data = invalid_source.data();
    ggml_gemmini_args_t invalid_args{};
    invalid_args.I = 1; invalid_args.J = 1; invalid_args.K = 32; invalid_args.sA = 32;
    invalid_args.tile_I = invalid_args.tile_J = invalid_args.tile_K = 1;
    invalid_args.activation_rows_per_stripe = DIM;
    if (!invalid_args.A.allocate(1, 32, GGML_GEMMINI_ACTIVATION_BITS)) return false;
    ExSIA invalid_exsia;
    Meta invalid_meta;
    for (int anchor : {2, 3}) {
        invalid_args.folding_anchor = anchor;
        for (const char * bad : {"", "2", "true", "01", " 1", "-1"}) {
            if (!policy.set(bad) ||
                !check(!invalid_exsia.run(invalid_meta, &invalid_tensor, invalid_args) &&
                       invalid_exsia.state().failure_code == ExSIAState::FailureCode::InvalidInput &&
                       invalid_meta.theta.empty(), "invalid policy fails at ExSIA boundary"))
                return false;
            if (!check(!quants::quantize_activation(&invalid_tensor, invalid_args),
                       "public dispatch rejects invalid direct policy")) return false;
        }
        if (!policy.set("1") ||
            !check(invalid_exsia.run(invalid_meta, &invalid_tensor, invalid_args),
                   "producer recovers after invalid policy")) return false;
    }
    return true;
}
