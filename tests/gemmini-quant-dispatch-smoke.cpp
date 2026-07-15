#include "../ggml/src/ggml-gemmini/ggml-gemmini-args.h"
#include "../ggml/src/ggml-gemmini/quants/act/dispatch.hpp"
#include "../ggml/src/ggml-gemmini/quants/act/exsia/types.hpp"
#include "../ggml/src/ggml-gemmini/quants/act/tensor/types.hpp"
#include "../ggml/src/ggml-gemmini/quants/act/token/types.hpp"

#include <cassert>
#include <string_view>
#include <vector>

int main(int argc, char **argv) {
    assert(argc == 2);

    const std::string_view mode = argv[1];
    ggml_gemmini_args_t args;

    if (mode == "fp") {
        const auto scales = ggml::gemmini::quants::act::activation_scales(args, 2);
        assert(scales == std::vector<float>({1.0f, 1.0f}));
        assert(ggml::gemmini::quants::act::outliers(args).empty());
    } else if (mode == "exsia") {
        args.I = 1;
        auto &meta = args.act_quant.storage().emplace<ggml::gemmini::quants::act::exsia::Meta>();
        meta.theta = {1};
        meta.outliers.push_back({2, 3, 4});

        const auto scales = ggml::gemmini::quants::act::activation_scales(args, 2);
        assert(scales.size() == 2 && scales[0] == 2.0f && scales[1] == 1.0f);
        assert(ggml::gemmini::quants::act::outliers(args).size() == 1);
    } else if (mode == "tensor") {
        auto &meta = args.act_quant.storage().emplace<ggml::gemmini::quants::act::tensor::Meta>();
        meta.scale = 0.25f;
        meta.outliers.push_back({1, 2, 3});

        const auto scales = ggml::gemmini::quants::act::activation_scales(args, 2);
        assert(scales.size() == 2 && scales[0] == 0.25f && scales[1] == 0.25f);
        assert(ggml::gemmini::quants::act::outliers(args).size() == 1);
    } else if (mode == "unsupported") {
        args.act_quant.storage().emplace<ggml::gemmini::quants::act::token::Meta>();
        const auto scales = ggml::gemmini::quants::act::activation_scales(args, 2);
        assert(scales == std::vector<float>({1.0f, 1.0f}));
        assert(ggml::gemmini::quants::act::outliers(args).empty());
    } else {
        assert(false && "unknown smoke mode");
    }

    return 0;
}
