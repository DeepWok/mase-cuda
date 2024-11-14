# Beginner Guide

This repo assumes that you have basic knowledge of the following:

- Git
- Bash
- Python
- PyTorch
- C++
- CMakes

## IDE Setup

- C++ Language Server: [Clangd](https://clangd.llvm.org/) is recommended. VSCode has a good [extension](https://marketplace.visualstudio.com/items?itemName=llvm-vs-code-extensions.vscode-clangd) for it. Clangd is faster than VSCode's official C++ extension.

  - Create Clangd configuration file for both global and project-specific settings. The global configuration file is located at `~/.config/clangd/config.yaml`. The project-specific configuration file is the `.clangd` file in the project root directory.

  - Here is an example of my global config
    ```yaml
    CompileFlags:
    Compiler: /usr/local/cuda/bin/nvcc
    Add:
      - --cuda-path=/usr/local/cuda
      - --cuda-gpu-arch=sm_86
      - -I/usr/local/cuda/include
      - "-xcuda"
      # report all errors
      - "-ferror-limit=0"
      - --std=c++17
      - "-D__INTELLISENSE__"
      - "-D__CLANGD__"
      # - "-DCUDA_12_0_SM90_FEATURES_SUPPORTED"
      # - "-DCUTLASS_ARCH_MMA_SM90_SUPPORTED=1"
      - "-D_LIBCUDACXX_STD_VER=12"
      - "-D__CUDACC_VER_MAJOR__=12"
      - "-D__CUDACC_VER_MINOR__=6"
      - "-D__CUDA_ARCH__=860"
      # - "-D__CUDA_ARCH_FEAT_SM90_ALL"
      - "-Wno-invalid-constexpr"
    Remove:
      # strip CUDA fatbin args
      - "-Xfatbin*"
      # strip CUDA arch flags
      - "-gencode*"
      - "--generate-code*"
      # strip CUDA flags unknown to clang
      - "-ccbin*"
      - "--compiler-options*"
      - "--expt-extended-lambda"
      - "--expt-relaxed-constexpr"
      - "-forward-unknown-to-host-compiler"
      - "-Werror=cross-execution-space-call"
    Hover:
    ShowAKA: No
    InlayHints:
    Enabled: No
    Diagnostics:
    Suppress:
        - "variadic_device_fn"
        - "attributes_not_allowed"
    ```

  - Here is an example of my project-specific config
    ```yaml
    CompileFlags:
      Compiler: /usr/local/cuda/bin/nvcc
      Add:
          - -I/home/zz7522/Projects/mase-cuda/submodules/cutlass/include/
          - -I/home/zz7522/Projects/mase-cuda/submodules/cutlass/tools/util/include/
          - -I/home/zz7522/Projects/mase-cuda/submodules/cutlass/examples/common/
          - -I/home/zz7522/Projects/mase-cuda/src/csrc/
          - -I/home/zz7522/Projects/mase-cuda/submodules/nlohmann_json/single_include/
          - -I/home/zz7522/Projects/mase-cuda/submodules/libtorch/include/
          - -I/home/zz7522/Projects/mase-cuda/submodules/libtorch/include/torch/csrc/api/include/
          - -I/mnt/data/zz7522/miniconda/envs/mase-cuda/include/python3.11/
    ```