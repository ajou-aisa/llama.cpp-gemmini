if(NOT DEFINED TEST_SCRIPT)
    message(FATAL_ERROR "TEST_SCRIPT is required")
endif()

file(READ "${TEST_SCRIPT}" script)
foreach(variable IN ITEMS LOG_DEBUG LOG_CYCLE)
    string(FIND "${script}"
        "${variable}_DEFAULT=\${${variable}:-0}"
        default_offset)
    if(default_offset EQUAL -1)
        message(FATAL_ERROR
            "${variable} must default off in build-arm64.sh")
    endif()
endforeach()
