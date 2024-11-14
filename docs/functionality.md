# Functionalities

## MXINT8

- Dequantization
    - [`mase_cuda::mxint8::dequantize::dequantize1d_device`](/src/csrc/mxint/dequantize.cuh): CUDA kernel for dequantizing a 1D mxint8 array
    - [`mase_cuda::mxint8::dequantize::dequantize1d_host`](/src/csrc/mxint/dequantize.cuh): Host function for dequantizing a 1D mxint8 array
    - [`mase_cuda::mxint8::dequantize::dequantize1d`](/src/csrc/mxint/dequantize.cuh): LibTorch wrapper for dequantizing a 1D mxint8 array, hanlding both host and device tensors