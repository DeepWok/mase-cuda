# https://cmake.org/cmake/help/latest/variable/CMAKE_LANG_FLAGS.html#variable:CMAKE_%3CLANG%3E_FLAGS
export CUDAFLAGS := "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 --expt-relaxed-constexpr"
export CUDA_ARCHITECTURES := "native"
project_dir := justfile_directory()

clean:
    if [ -d {{project_dir}}/build ]; then rm -r {{project_dir}}/build; fi


build-test:
    echo $(which cmake)
    cmake -D BUILD_TESTING=ON -D CMAKE_CUDA_ARCHITECTURES={{CUDA_ARCHITECTURES}} -B build -S .

build-test-gdb:
    cmake -D BUILD_TESTING=ON -D CMAKE_CUDA_ARCHITECTURES={{CUDA_ARCHITECTURES}} -D NVCCGDB=ON -B build -S .