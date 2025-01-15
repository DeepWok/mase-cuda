# MASE CUDA

This is the cuda extension for [DeepWok/MASE](https://github.com/DeepWok/mase).

> [!NOTE]
> This project is still under development.

> [!NOTE]
> [Beginner Guide](/docs/beginner.md) notes down my learning process and setup.

## Conda Env Setup

- C++17 (GCC < 14)
- CUDA 12.4/12.5/12.6
- CMake >= 3.20
- Python >= 3.11
  - [Tox](https://tox.wiki/en/latest/index.html) for Python package building and testing.
  - Torch >= 2.3.0 (`pip install torch` in conda env) which includes LibTorch for wrapping CUDA kernels.
- [Justfile](https://github.com/casey/just)


### C++/CUDA

#### Build Tests


- Build Tests

  ```bash
  just build-cu-test
  ```

- Build Profiling for NSight Compute

  ```bash
  just build-cu-profile
  ```

#### Build Specific Target

- Build `test_mxint8_dequantize1d` for debug and launch cuda-gdb for debugging

  ```bash
  just --set CU_BUILD_TARGETS test_mxint8_dequantize1d build-cu-test-debug

  cuda-gdb --args ./build/test/cu/mxint/dequantize/test_mxint8_dequantize1d 25600 256
  ```


### Python

`tox` sets up the python environment automatically.

- Build `mase_cuda` package and run python tests

  ```bash
  tox # this is slow since cpu & gpu performance profiling is enabled
  ```

  - Run quick test
    ```bash
    just test-py-fast
    ```

  - The package is built in `dist/` directory.

- Create env for dev

  ```bash
  tox -e dev
  ```

- Build `mase_cuda` package

  ```bash
  just build-py
  ```


