cmake_minimum_required(VERSION 3.26)
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

# python
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
include_directories("${Python3_INCLUDE_DIRS}")

# torch
# check if libtorch exists
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/submodules/libtorch")
  message(FATAL_ERROR "libtorch not found. Please run `just download-torchlib-if-not-exists`.")
endif()

list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/submodules/libtorch")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

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

  # add dependencies: cutlass, cutlass_util, cutlass_exp, python, libtorch, nlohmann_json, mase_cuda
  target_link_libraries(${name} cutlass cutlass_util cutlass_exp ${Python3_LIBRARIES} ${TORCH_LIBRARIES} nlohmann_json mase_cuda)
  set_property(TARGET ${name} PROPERTY CXX_STANDARD 17)

  # if NVCCGDB is set, add -g -G flags
  if(NVCCGDB)
    target_compile_options(${name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-g -G>)
  endif()
endfunction()

# a function to add a cuda executable for nsight profiling
function(add_executable_nsight name)
  set(multiValueArgs SRC)
  cmake_parse_arguments(PARSE_ARGV 1 arg "" "" ${multiValueArgs})

  add_executable(${name} ${arg_SRC})

  # add dependencies: cutlass, cutlass_util, cutlass_exp, python, libtorch, nlohmann_json, mase_cuda
  target_link_libraries(${name} cutlass cutlass_util cutlass_exp ${Python3_LIBRARIES} ${TORCH_LIBRARIES} nlohmann_json mase_cuda)
  set_property(TARGET ${name} PROPERTY CXX_STANDARD 17)

  target_compile_options(${name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
endfunction()

# add tests
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND BUILD_TESTING)
  add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/test/cu")
endif()

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND BUILD_PROFILING)
  add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/profile/cu")
endif()
