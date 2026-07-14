#include "ggml-gemmini-q8-h1-artifact.hpp"
#include "ggml-gemmini-q8-h1-artifact-reader.hpp"
#include <gemmini/log.hpp>

#include <cstdlib>
#include <exception>
#include <new>

namespace ggml::gemmini {
namespace {

bool fail(q8_h1_artifact_store & store, std::string * error, const char * message) noexcept {
    store.clear();
    if (error != nullptr) {
        try {
            *error = message;
        } catch (...) {
        }
    }
    return false;
}

}

bool load_q8_h1_artifact(
        const std::string & path,
        q8_h1_artifact_store & store,
        std::string * error) {
    store.clear();
    if (error != nullptr) {
        error->clear();
    }

    try {
        return detail::load_q8_h1_artifact_impl(path, store, error);
    } catch (const std::bad_alloc &) {
        return fail(store, error, "artifact allocation failed");
    } catch (const std::exception &) {
        return fail(store, error, "artifact load failed");
    }
}

void load_q8_h1_artifact_from_env(q8_h1_artifact_runtime_state & state) {
    state = {};

    const char * path = std::getenv("LLAMA_GEMMINI_Q8_H1_ARTIFACT");
    if (path == nullptr || path[0] == '\0') {
        ggml::gemmini::log::debug("Q8_H1 artifact", "LLAMA_GEMMINI_Q8_H1_ARTIFACT is not set");
        return;
    }

    ggml::gemmini::log::debug("Q8_H1 artifact", "loading q8_h1 artifact from '%s'", path);
    state.path = path;
    if (load_q8_h1_artifact(state.path, state.store, &state.error)) {
        state.status = q8_h1_artifact_runtime_status::ready;
        ggml::gemmini::log::debug("Q8_H1 artifact", "loaded artifact '%s' tensors=%zu", state.path.c_str(), state.store.size());
        return;
    }

    ggml::gemmini::log::debug("Q8_H1 artifact", "artifact '%s' load failed: %s", state.path.c_str(), state.error.c_str());
    state.status = q8_h1_artifact_runtime_status::disabled_load_error;
    state.store.clear();
}

q8_h1_artifact_lookup_status lookup_q8_h1_artifact_tensor(
        const q8_h1_artifact_runtime_state & state,
        const char * tensor_name,
        const std::array<int64_t, GGML_MAX_DIMS> & dims,
        const q8_h1_artifact_tensor ** tensor) {
    if (tensor != nullptr) {
        *tensor = nullptr;
    }

    if (state.status != q8_h1_artifact_runtime_status::ready) {
        return q8_h1_artifact_lookup_status::disabled;
    }

    if (tensor_name == nullptr || tensor_name[0] == '\0') {
        return q8_h1_artifact_lookup_status::missing_tensor;
    }

    const auto it = state.store.find(tensor_name);
    if (it == state.store.end()) {
        return q8_h1_artifact_lookup_status::missing_tensor;
    }

    size_t logical_rows = 0;
    size_t k = 0;
    size_t blocks_per_row = 0;
    if (!detail::dims_to_geometry(dims, logical_rows, k, blocks_per_row)) {
        return q8_h1_artifact_lookup_status::shape_mismatch;
    }

    const q8_h1_artifact_tensor & candidate = it->second;
    if (candidate.dims != dims ||
        candidate.logical_rows != logical_rows ||
        candidate.k != k ||
        candidate.blocks_per_row != blocks_per_row) {
        return q8_h1_artifact_lookup_status::shape_mismatch;
    }

    if (tensor != nullptr) {
        *tensor = &candidate;
    }
    return q8_h1_artifact_lookup_status::ready;
}

}
