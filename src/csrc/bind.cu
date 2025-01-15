#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "./mxint/dequantize.cuh"
#include "./mxint/dequantize_fast.cuh"

// refer to https://github.com/pybind/python_example/blob/master/src/main.cpp
// https://pytorch.org/tutorials/advanced/cpp_extension.html

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "This is a CUDA-accelerated PyTorch extension";
    auto m_mxint8 = m.def_submodule("mxint8", "OCP-MXINT8 module");
    m_mxint8.def("dequantize1d", &mase_cuda::mxint8::dequantize::dequantize1d, py::arg("x"), py::arg("scales"),
                 py::arg("group_size"));

    m_mxint8.def("dequantize1d_fast", &mase_cuda::mxint8::dequantize_fast::dequantize1d, py::arg("x"),
                 py::arg("scales"), py::arg("group_size"));
}