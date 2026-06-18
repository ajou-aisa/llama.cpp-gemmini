#pragma once

#include <map>
#include <optional>
#include <string>

#include <gemmini/layer.hpp>

struct ggml_gemmini_exsia_patch
{
    std::optional<int> q;
};

struct ggml_gemmini_exsia_arch_config
{
    ggml_gemmini_exsia_patch defaults;
    std::map<ggml::gemmini::types::LayerType, ggml_gemmini_exsia_patch> layers;
};

struct ggml_gemmini_exsia_config_registry
{
    bool available = false;
    ggml_gemmini_exsia_patch defaults;
    std::map<std::string, ggml_gemmini_exsia_arch_config> architectures;
};

struct ggml_gemmini_resolved_exsia_override
{
    int q = 0;
};

ggml_gemmini_exsia_config_registry ggml_gemmini_load_exsia_config_registry();

std::optional<ggml_gemmini_resolved_exsia_override> ggml_gemmini_resolve_exsia_override(
    const ggml_gemmini_exsia_config_registry &registry,
    const std::string &model_arch,
    ggml::gemmini::types::LayerType layer_type);
