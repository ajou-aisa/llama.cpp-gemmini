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
endforeach()
file(REMOVE_RECURSE "${TEST_BINARY_ROOT}")
