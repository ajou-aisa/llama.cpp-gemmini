set(_production_files
    "${CMAKE_CURRENT_LIST_DIR}/../ggml/src/ggml-gemmini/residual/direct/direct-executor.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/../ggml/src/ggml-gemmini/residual/rmd/rmd-executor.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/../ggml/src/ggml-gemmini/residual/rmd/rmd-reference.cpp")

set(_caller_files 0)
foreach(_file IN LISTS _production_files)
    file(READ "${_file}" _source)

    string(FIND "${_source}" "weight_reader.hpp" _include_position)
    string(FIND "${_source}" "wreader::read_code" _caller_position)
    if(_include_position EQUAL -1 OR _caller_position EQUAL -1)
        message(FATAL_ERROR "${_file} must consume the shared production weight reader")
    endif()
    math(EXPR _caller_files "${_caller_files} + 1")

    foreach(_duplicate_pattern IN ITEMS
            "read_weight_code"
            "bool[ \t\r\n]+weight_code[ \t\r\n]*\\("
            "block[ \t\r\n]*->[ \t\r\n]*qs"
            "q8_h1_block[ \t\r\n]*\\("
            "q8_hp1_block[ \t\r\n]*\\(")
        string(REGEX MATCH "${_duplicate_pattern}" _duplicate "${_source}")
        if(NOT _duplicate STREQUAL "")
            message(FATAL_ERROR "${_file} retains duplicate Q8-only decoding: ${_duplicate}")
        endif()
    endforeach()
endforeach()

if(NOT _caller_files EQUAL 3)
    message(FATAL_ERROR "expected three production weight-reader callers, found ${_caller_files}")
endif()

message(STATUS "PASS: production weight-reader caller files=3 duplicate-decoders=0")
