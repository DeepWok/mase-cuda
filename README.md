# MASE CUDA

This is the cuda extension for [DeepWok/MASE](https://github.com/DeepWok/mase).

> [!NOTE]
> This project is still under development.

> [!NOTE]
> [Beginner Guide](/docs/beginner.md) notes down my learning process and setup.

## Env Setup

- C++17 (GCC < 14)
- CUDA 12.4/12.5/12.6
- CMake >= 3.20
- Python >= 3.11 [Tox](https://tox.wiki/en/latest/index.html)
- [Justfile](https://github.com/casey/just)

### C++/CUDA

#### Build

1. Submodules and LibTorch

```bash
git submodule update --init
just download-libtorch-if-not-exists
```

2. Build Tests

```bash
just build-cu-test
```

3. Build Profiling for NSight Compute

```bash
just build-cu-profile
```

#### Build Specific Target

- Build `test_mxint8_dequantize1d_fast` for debug and launch cuda-gdb for debugging
  ```bash
  just --set CU_BUILD_TARGETS test_mxint8_dequantize1d_fast build-cu-test-debug
  cuda-gdb --args ./build/test/cu/mxint/dequantize/test_mxint8_dequantize1d_fast  30000 30
  ```


### Python

`tox` sets up the python environment automatically.

- Build `mase_cuda` and test

  ```bash
  tox # this is slow since cpu & gpu profiling is enabled
  ```

  - Run quick test
    ```bash
    just test-py-fast
    ```

- Create env for dev

  ```bash
  tox -e dev
  ```

## Functionalities

See [functionality list](/docs/functionality.md) for more details.

