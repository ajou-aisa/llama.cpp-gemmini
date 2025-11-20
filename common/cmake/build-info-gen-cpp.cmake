cmake_minimum_required(VERSION 3.14)

# Paths
get_filename_component(PROJECT_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
set(COMMON_DIR "${PROJECT_ROOT}/common")

# Reuse the shared build-info logic (sets BUILD_NUMBER, BUILD_COMMIT, BUILD_COMPILER, BUILD_TARGET)
include("${PROJECT_ROOT}/cmake/build-info.cmake")

# Render the cpp from template
configure_file(
  "${COMMON_DIR}/build-info.cpp.in"
  "${COMMON_DIR}/build-info.cpp"
  @ONLY
)
