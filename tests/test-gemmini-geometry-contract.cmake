if(NOT DEFINED TEST_SOURCE_DIR)
    message(FATAL_ERROR "TEST_SOURCE_DIR is required")
endif()

set(forbidden_symbols
    "GGML_GEMMINI_DEFAULT_" "STRIPE_ROWS"
    "GEMMINI_" "STRIPE_ROWS")
list(GET forbidden_symbols 0 build_prefix)
list(GET forbidden_symbols 1 stripe_suffix)
list(GET forbidden_symbols 2 runtime_prefix)
list(GET forbidden_symbols 3 runtime_suffix)
set(legacy_build_symbol "${build_prefix}${stripe_suffix}")
set(legacy_runtime_symbol "${runtime_prefix}${runtime_suffix}")

get_filename_component(source_root "${TEST_SOURCE_DIR}" REALPATH)
file(GLOB_RECURSE repository_files LIST_DIRECTORIES false "${source_root}/*")

set(matches "")
foreach(path IN LISTS repository_files)
    file(RELATIVE_PATH relative_path "${source_root}" "${path}")
    if(relative_path MATCHES
       "(^|/)(\\.git|\\.omo|\\.omx|\\.cache|\\.ruff_cache|\\.venv|\\.senpi|Testing|CMakeFiles|generated|target|models?|logs?|output)(/|$)" OR
       relative_path MATCHES "(^|/)build[^/]*/" OR
       relative_path STREQUAL "tests/parse-stripe-ws-profile.py")
        continue()
    endif()
    file(READ "${path}" content)
    foreach(symbol IN ITEMS "${legacy_build_symbol}" "${legacy_runtime_symbol}")
        string(FIND "${content}" "${symbol}" found_at)
        if(NOT found_at EQUAL -1)
            list(APPEND matches "${relative_path}: ${symbol}")
        endif()
    endforeach()
endforeach()

if(matches)
    list(JOIN matches "\n  " formatted_matches)
    message(FATAL_ERROR "Obsolete Gemmini stripe-row symbols remain:\n  ${formatted_matches}")
endif()
