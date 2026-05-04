#include "ggml-gemmini-ethos-config.hpp"

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <utility>

#include "../../../common/json.hpp"

namespace
{
    using json = nlohmann::ordered_json;

    void merge_ethos_patch(
        ggml_gemmini_ethos_patch &dst,
        const ggml_gemmini_ethos_patch &src)
    {
        if (src.q.has_value())
            dst.q = src.q;
        if (src.delta.has_value())
            dst.delta = src.delta;
        if (src.l2_enabled.has_value())
            dst.l2_enabled = src.l2_enabled;
        if (src.l2_c.has_value())
            dst.l2_c = src.l2_c;
        if (src.l2_d.has_value())
            dst.l2_d = src.l2_d;
    }

    [[noreturn]] void throw_json_error(const std::string &context, const std::string &message)
    {
        throw std::runtime_error("ethos_config.json " + context + ": " + message);
    }

    void require_object(const json &value, const std::string &context)
    {
        if (!value.is_object())
            throw_json_error(context, "must be an object");
    }

    int require_int(const json &value, const std::string &context)
    {
        if (!value.is_number_integer())
            throw_json_error(context, "must be an integer");

        return value.get<int>();
    }

    bool require_bool(const json &value, const std::string &context)
    {
        if (!value.is_boolean())
            throw_json_error(context, "must be a boolean");

        return value.get<bool>();
    }

    void validate_q(int q, const std::string &context)
    {
        if (q < 2 || q > 8)
            throw_json_error(context, "q must be in [2, 8]");
    }

    ggml_gemmini_ethos_patch parse_ethos_patch(const json &value, const std::string &context)
    {
        require_object(value, context);

        ggml_gemmini_ethos_patch patch{};
        for (auto it = value.begin(); it != value.end(); ++it)
        {
            if (it.key() == "q")
            {
                const int q = require_int(it.value(), context + ".q");
                validate_q(q, context + ".q");
                patch.q = q;
                continue;
            }

            if (it.key() == "delta")
            {
                patch.delta = require_int(it.value(), context + ".delta");
                continue;
            }

            if (it.key() == "l2")
            {
                require_object(it.value(), context + ".l2");
                for (auto l2_it = it.value().begin(); l2_it != it.value().end(); ++l2_it)
                {
                    if (l2_it.key() == "enabled")
                    {
                        patch.l2_enabled = require_bool(l2_it.value(), context + ".l2.enabled");
                        continue;
                    }

                    if (l2_it.key() == "c")
                    {
                        patch.l2_c = require_int(l2_it.value(), context + ".l2.c");
                        continue;
                    }

                    if (l2_it.key() == "d")
                    {
                        patch.l2_d = require_int(l2_it.value(), context + ".l2.d");
                        continue;
                    }

                    throw_json_error(context + ".l2", "unknown key '" + l2_it.key() + "'");
                }
                continue;
            }

            throw_json_error(context, "unknown key '" + it.key() + "'");
        }

        return patch;
    }

    std::filesystem::path find_ethos_config_path()
    {
        namespace fs = std::filesystem;

        const fs::path local = fs::path(__FILE__).parent_path() / "ethos_config.json";
        if (fs::exists(local) && fs::is_regular_file(local))
            return local;

        return {};
    }
}

ggml_gemmini_ethos_config_registry ggml_gemmini_load_ethos_config_registry()
{
    namespace fs = std::filesystem;

    ggml_gemmini_ethos_config_registry registry{};
    const fs::path config_path = find_ethos_config_path();
    if (config_path.empty())
        return registry;

    std::ifstream input(config_path);
    if (!input)
        throw std::runtime_error("failed to open ethos_config.json: " + config_path.string());

    json root;
    try
    {
        input >> root;
    }
    catch (const json::exception &e)
    {
        throw std::runtime_error("failed to parse ethos_config.json: " + std::string(e.what()));
    }

    require_object(root, "root");
    for (auto it = root.begin(); it != root.end(); ++it)
    {
        if (it.key() == "version")
        {
            const int version = require_int(it.value(), "version");
            if (version != 1)
                throw_json_error("version", "must be 1");
            continue;
        }

        if (it.key() == "defaults")
        {
            registry.defaults = parse_ethos_patch(it.value(), "defaults");
            continue;
        }

        if (it.key() == "architectures")
        {
            require_object(it.value(), "architectures");
            for (auto arch_it = it.value().begin(); arch_it != it.value().end(); ++arch_it)
            {
                require_object(arch_it.value(), "architectures." + arch_it.key());
                ggml_gemmini_ethos_arch_config arch_cfg{};

                for (auto arch_value_it = arch_it.value().begin(); arch_value_it != arch_it.value().end(); ++arch_value_it)
                {
                    if (arch_value_it.key() == "default")
                    {
                        arch_cfg.defaults = parse_ethos_patch(
                            arch_value_it.value(),
                            "architectures." + arch_it.key() + ".default");
                        continue;
                    }

                    if (arch_value_it.key() == "layers")
                    {
                        require_object(arch_value_it.value(), "architectures." + arch_it.key() + ".layers");
                        for (auto layer_it = arch_value_it.value().begin(); layer_it != arch_value_it.value().end(); ++layer_it)
                        {
                            const auto layer_type = orca::types::parse_layer(layer_it.key());
                            if (layer_type == orca::types::LayerType::unknown)
                            {
                                throw_json_error(
                                    "architectures." + arch_it.key() + ".layers",
                                    "unknown layer '" + layer_it.key() + "'");
                            }

                            arch_cfg.layers[layer_type] = parse_ethos_patch(
                                layer_it.value(),
                                "architectures." + arch_it.key() + ".layers." + layer_it.key());
                        }
                        continue;
                    }

                    throw_json_error(
                        "architectures." + arch_it.key(),
                        "unknown key '" + arch_value_it.key() + "'");
                }

                registry.architectures.emplace(arch_it.key(), std::move(arch_cfg));
            }
            continue;
        }

        throw_json_error("root", "unknown key '" + it.key() + "'");
    }

    if (!registry.defaults.q.has_value() ||
        !registry.defaults.delta.has_value() ||
        !registry.defaults.l2_enabled.has_value() ||
        !registry.defaults.l2_c.has_value() ||
        !registry.defaults.l2_d.has_value())
        throw_json_error("defaults", "must define q, delta, l2.enabled, l2.c, and l2.d");

    registry.available = true;
    return registry;
}

std::optional<ggml_gemmini_resolved_ethos_override> ggml_gemmini_resolve_ethos_override(
    const ggml_gemmini_ethos_config_registry &registry,
    const std::string &model_arch,
    orca::types::LayerType layer_type)
{
    if (!registry.available)
        return std::nullopt;

    const auto arch_it = registry.architectures.find(model_arch);
    if (arch_it == registry.architectures.end())
        return std::nullopt;

    ggml_gemmini_ethos_patch merged = registry.defaults;
    merge_ethos_patch(merged, arch_it->second.defaults);

    const auto layer_it = arch_it->second.layers.find(layer_type);
    if (layer_it != arch_it->second.layers.end())
        merge_ethos_patch(merged, layer_it->second);

    if (!merged.q.has_value() || !merged.delta.has_value() ||
        !merged.l2_enabled.has_value() || !merged.l2_c.has_value() ||
        !merged.l2_d.has_value())
        throw std::runtime_error("ethos_config.json resolved an incomplete Ethos override");

    return ggml_gemmini_resolved_ethos_override{
        *merged.q,
        *merged.delta,
        *merged.l2_enabled,
        *merged.l2_c,
        *merged.l2_d,
    };
}
