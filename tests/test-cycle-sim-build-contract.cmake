if(NOT DEFINED TEST_SOURCE_DIR OR NOT DEFINED TEST_BUILD_DIR)
    message(FATAL_ERROR "TEST_SOURCE_DIR and TEST_BUILD_DIR are required")
endif()

if(TEST_CYCLE_SIM)
    include("${TEST_BUILD_DIR}/cycle-sim-generated/cycle-sim-target.cmake")
    file(READ "${TEST_BUILD_DIR}/generated/gemmini_params.h" selected_header)
    foreach(pair IN ITEMS "BANK_NUM=${IM2P_CYCLE_SIM_BANK_COUNT}"
                          "BANK_ROWS=${IM2P_CYCLE_SIM_BANK_ROWS}"
                          "ACC_ROWS=${IM2P_CYCLE_SIM_ACCUMULATOR_ROWS}")
        string(REPLACE "=" ";" parts "${pair}")
        list(GET parts 0 macro)
        list(GET parts 1 expected)
        string(REGEX MATCH "#define[ \t]+${macro}[ \t]+([0-9]+)" definition "${selected_header}")
        if(NOT definition OR NOT CMAKE_MATCH_1 STREQUAL expected)
            message(FATAL_ERROR "selected production header disagrees with resolved ${pair}")
        endif()
    endforeach()
endif()

foreach(script IN ITEMS build-arm64.sh build-arm64-cpu.sh build-x86.sh build-riscv.sh)
    foreach(enabled IN ITEMS 0 1)
        execute_process(COMMAND "${CMAKE_COMMAND}" -E env
            "CYCLE_SIM=${enabled}" "LOG_CYCLE=0" "GGML_CPU_CYCLE_LOG=0"
            "CYCLE_DETAIL=0" "GGML_GEMMINI_EXECUTION_BACKEND=HARDWARE"
            "BUILD_JOBS=1"
            "BUILD_DIR=${TEST_BUILD_DIR}/dry-${script}-${enabled}"
            bash "${TEST_SOURCE_DIR}/${script}" --dry-run
            RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
        if(NOT result EQUAL 0)
            message(FATAL_ERROR "${script}: ${error}")
        endif()
        foreach(pair IN ITEMS "CYCLE_SIM=${enabled}" "LOG_CYCLE=0" "GGML_CPU_CYCLE_LOG=0" "CYCLE_DETAIL=0")
            string(REPLACE "=" "\": \"" json_pair "${pair}")
            string(FIND "${error}" "\"${json_pair}\"" found)
            if(found EQUAL -1)
                message(FATAL_ERROR "${script} did not preserve ${pair}: ${error}")
            endif()
        endforeach()
    endforeach()
endforeach()

execute_process(COMMAND "${CMAKE_COMMAND}" -E env
    "CYCLE_SIM=0" "IM2P_ARTIFACT_SET=ALL_MATCHED"
    "GGML_GEMMINI_EXECUTION_BACKEND=IM2P_SIM" "GGML_GEMMINI_OPTION=WS"
    bash "${TEST_SOURCE_DIR}/build-riscv.sh" static --dry-run -DCYCLE_SIM=1
    RESULT_VARIABLE cross_result OUTPUT_VARIABLE cross_output ERROR_VARIABLE cross_error)
if(NOT cross_result EQUAL 0 OR NOT cross_error MATCHES "\"CYCLE_SIM\": \"1\"")
    message(FATAL_ERROR "RISC-V CPU-functional option was dropped/rejected: ${cross_error}")
endif()

string(RANDOM LENGTH 16 suffix)
set(script_root "${TEST_BUILD_DIR}/cycle-sim-script-contract-${suffix}")
file(MAKE_DIRECTORY "${script_root}/bin")
file(WRITE "${script_root}/bin/make" "#!/bin/sh\nexit 87\n")
file(WRITE "${script_root}/bin/cmake" "#!/bin/sh\nprintf '%s\\n' \"$*\" >> \"$CONTRACT_LOG\"\n")
execute_process(COMMAND chmod +x "${script_root}/bin/make" "${script_root}/bin/cmake"
    RESULT_VARIABLE executable_result)
if(NOT executable_result EQUAL 0)
    message(FATAL_ERROR "Cannot prepare build-script command observers")
endif()
foreach(script IN ITEMS build-arm64.sh build-arm64-cpu.sh build-x86.sh)
    execute_process(COMMAND "${CMAKE_COMMAND}" -E env
        "PATH=${script_root}/bin:$ENV{PATH}" "CONTRACT_LOG=${script_root}/${script}.log"
        "BUILD_DIR=${script_root}/${script}" "BUILD_JOBS=1"
        "GGML_GEMMINI_EXECUTION_BACKEND=IM2P_SIM" "GGML_GEMMINI_OPTION=WS"
        "CYCLE_SIM=1" "LOG_CYCLE=0" "GGML_CPU_CYCLE_LOG=0" "CYCLE_DETAIL=0"
        bash "${TEST_SOURCE_DIR}/${script}"
        RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
    if(NOT result EQUAL 0)
        message(FATAL_ERROR "${script} CPU-functional build tried provisioning: ${output}\n${error}")
    endif()
    file(READ "${script_root}/${script}.log" calls)
    if(NOT calls MATCHES "-DCYCLE_SIM=1")
        message(FATAL_ERROR "${script} dropped resolved CYCLE_SIM before configure")
    endif()
endforeach()

file(READ "${TEST_BUILD_DIR}/compile_commands.json" compile_database)
string(REGEX MATCHALL "\"command\": [^\n]+" commands "${compile_database}")
foreach(target IN ITEMS ggml-gemmini ggml-gemmini-utils ggml-cpu ggml-base common llama llama-cli)
    set(found FALSE)
    foreach(command IN LISTS commands)
        if(command MATCHES "CMakeFiles/${target}\\.dir/")
            set(found TRUE)
            string(REGEX MATCHALL "-DCYCLE_SIM=[^ ]+" definitions "${command}")
            list(LENGTH definitions count)
            if(NOT count EQUAL 1 OR NOT definitions STREQUAL "-DCYCLE_SIM=${TEST_CYCLE_SIM}")
                message(FATAL_ERROR "${target} CYCLE_SIM must occur exactly once: ${definitions}")
            endif()
            if(target MATCHES "^(ggml-gemmini|ggml-gemmini-utils|ggml-cpu|ggml-base|llama|llama-cli)$")
                string(REGEX MATCHALL "-DLOG_CYCLE=[^ ]+" definitions "${command}")
                if(NOT definitions STREQUAL "-DLOG_CYCLE=${TEST_LOG_CYCLE}")
                    message(FATAL_ERROR "${target} ordinary telemetry flag changed: ${definitions}")
                endif()
            endif()
            if(target STREQUAL "ggml-cpu")
                string(REGEX MATCHALL "-DCYCLE_LOG=[^ ]+" definitions "${command}")
                if(NOT definitions STREQUAL "-DCYCLE_LOG=${TEST_CPU_CYCLE_LOG}")
                    message(FATAL_ERROR "ggml-cpu telemetry flag changed: ${definitions}")
                endif()
            endif()
        endif()
    endforeach()
    if(NOT found)
        message(FATAL_ERROR "No compilation command for ${target}")
    endif()
endforeach()
