cmake_minimum_required(VERSION 3.20)
project(mase-cuda VERSION 0.1 LANGUAGES CUDA CXX)

# print CUDA_ARCHITECTURES
message(STATUS "CUDA_ARCHITECTURES: ${CUDA_ARCHITECTURES}")

# add CUTLASS
set(CUTLASS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/submodules/cutlass" CACHE PATH "CUTLASS Repository Directory")

# cutlass/include
add_library(cutlass INTERFACE)
target_include_directories(cutlass INTERFACE "${CUTLASS_DIR}/include")
add_library(cutlass_util INTERFACE)
target_include_directories(cutlass_util INTERFACE "${CUTLASS_DIR}/tools/util/include")
add_library(cutlass_exp INTERFACE)
target_include_directories(cutlass_exp INTERFACE "${CUTLASS_DIR}/examples/common")

# json
add_library(nlohmann_json INTERFACE)
target_include_directories(nlohmann_json INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/submodules/nlohmann_json/single_include")

# add mase_cuda
add_library(mase_cuda INTERFACE)
target_include_directories(mase_cuda INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/src/csrc")

# a function to add a cuda test
function(add_executable_test name)
  set(multiValueArgs SRC)
  cmake_parse_arguments(PARSE_ARGV 1 arg "" "" ${multiValueArgs})

  add_executable(${name} ${arg_SRC})
  target_link_libraries(${name} cutlass cutlass_util cutlass_exp nlohmann_json mase_cuda)

  # if NVCCGDB is set, add -g -G flags
  if(NVCCGDB)
    target_compile_options(${name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-g -G>)
  endif()
endfunction()

# add tests
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND BUILD_TESTING)
  add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/test/cu")
endif()
