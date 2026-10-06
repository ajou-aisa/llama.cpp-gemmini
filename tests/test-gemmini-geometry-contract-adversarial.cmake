foreach(required IN ITEMS TEST_CMAKE_COMMAND TEST_SCANNER TEST_BINARY_ROOT)
    if(NOT DEFINED ${required})
        message(FATAL_ERROR "${required} is required")
    endif()
endforeach()

set(symbol_parts "GEMMINI_" "STRIPE_ROWS")
list(GET symbol_parts 0 symbol_prefix)
list(GET symbol_parts 1 symbol_suffix)
set(forbidden_symbol "${symbol_prefix}${symbol_suffix}")

set(offenders
    "src/offender.cpp"
    "common/offender.hpp"
    "tools/offender-config.toml"
    "release-check.sh")

file(REMOVE_RECURSE "${TEST_BINARY_ROOT}")
foreach(offender IN LISTS offenders)
    string(REPLACE "/" "-" case_name "${offender}")
    set(case_root "${TEST_BINARY_ROOT}/${case_name}")
    get_filename_component(parent "${case_root}/${offender}" DIRECTORY)
    file(MAKE_DIRECTORY "${parent}")
    file(WRITE "${case_root}/${offender}" "${forbidden_symbol}\n")

    execute_process(
        COMMAND "${TEST_CMAKE_COMMAND}"
            -DTEST_SOURCE_DIR=${case_root}
            -P "${TEST_SCANNER}"
        RESULT_VARIABLE rc
        OUTPUT_VARIABLE stdout
        ERROR_VARIABLE stderr)
    if(rc EQUAL 0)
        message(FATAL_ERROR "Scanner allowed injected token in ${offender}")
    endif()
    string(CONCAT output "${stdout}" "\n" "${stderr}")
    string(FIND "${output}" "${offender}" offender_at)
    if(offender_at EQUAL -1)
        message(FATAL_ERROR
            "Scanner failure did not name ${offender}:\n${output}")
    endif()
    message(STATUS "Owned source mutation rejected: ${offender}")
endforeach()

set(case_root "${TEST_BINARY_ROOT}/artifact-poison")
file(MAKE_DIRECTORY "${case_root}/src")
file(WRITE "${case_root}/src/clean.cpp" "int geometry_fixture;\n")
foreach(artifact IN ITEMS
        runs/poison.cpp models/poison.cpp .cache/poison.cpp .omo/poison.cpp
        graphify-out/poison.cpp build-local/poison.cpp unrelated-experiment/poison.cpp
        src/generated/poison.cpp ggml/src/cache/poison.cpp
        ggml/src/graphify-out/poison.cpp tools/node_modules/poison.cpp
        ggml/src/quants/archive.zip)
    get_filename_component(parent "${case_root}/${artifact}" DIRECTORY)
    file(MAKE_DIRECTORY "${parent}")
    file(WRITE "${case_root}/${artifact}" "${forbidden_symbol}\n")
endforeach()
execute_process(COMMAND "${TEST_CMAKE_COMMAND}"
    -DTEST_SOURCE_DIR=${case_root} -DTEST_INVENTORY=${case_root}/inventory.txt
    -P "${TEST_SCANNER}"
    RESULT_VARIABLE rc OUTPUT_VARIABLE stdout ERROR_VARIABLE stderr)
if(NOT rc EQUAL 0)
    message(FATAL_ERROR "Artifact poison affected the source verdict:\n${stdout}\n${stderr}")
endif()
file(READ "${case_root}/inventory.txt" inventory)
if(NOT inventory STREQUAL "src/clean.cpp\n")
    message(FATAL_ERROR "Unexpected source inventory with artifact poison:\n${inventory}")
endif()
message(STATUS "Artifact poison ignored; inventory contains only owned source")

file(WRITE "${case_root}/src/new-owned.cpp" "${forbidden_symbol}\n")
execute_process(COMMAND "${TEST_CMAKE_COMMAND}" -DTEST_SOURCE_DIR=${case_root}
    -P "${TEST_SCANNER}"
    RESULT_VARIABLE rc OUTPUT_VARIABLE stdout ERROR_VARIABLE stderr)
if(rc EQUAL 0 OR NOT "${stdout}\n${stderr}" MATCHES "src/new-owned\\.cpp")
    message(FATAL_ERROR "New owned source escaped the scanner:\n${stdout}\n${stderr}")
endif()
message(STATUS "New untracked owned source mutation rejected")

file(MAKE_DIRECTORY "${TEST_BINARY_ROOT}/empty")
foreach(invalid_root IN ITEMS missing artifact-poison/src/clean.cpp empty)
    execute_process(COMMAND "${TEST_CMAKE_COMMAND}"
        -DTEST_SOURCE_DIR=${TEST_BINARY_ROOT}/${invalid_root} -P "${TEST_SCANNER}"
        RESULT_VARIABLE rc OUTPUT_VARIABLE stdout ERROR_VARIABLE stderr)
    if(rc EQUAL 0 OR NOT "${stdout}\n${stderr}" MATCHES "TEST_SOURCE_DIR")
        message(FATAL_ERROR "Invalid source root was not rejected: ${invalid_root}")
    endif()
    message(STATUS "Invalid source root rejected: ${invalid_root}")
endforeach()
execute_process(COMMAND "${TEST_CMAKE_COMMAND}" -P "${TEST_SCANNER}"
    RESULT_VARIABLE rc OUTPUT_VARIABLE stdout ERROR_VARIABLE stderr)
if(rc EQUAL 0 OR NOT "${stdout}\n${stderr}" MATCHES "TEST_SOURCE_DIR")
    message(FATAL_ERROR "Missing TEST_SOURCE_DIR was not rejected")
endif()
message(STATUS "Missing source root argument rejected")
file(REMOVE_RECURSE "${TEST_BINARY_ROOT}")
