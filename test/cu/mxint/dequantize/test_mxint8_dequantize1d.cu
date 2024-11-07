#include "mxint/dequantize.cuh"
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cute/layout.hpp>
#include <cute/numeric/integral_constant.hpp>
#include <cute/numeric/numeric_types.hpp>
#include <cutlass/bfloat16.h>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

int main(int argc, char **argv) {
    using namespace cute;
    using namespace mase_cuda;
    using json = nlohmann::json;

    const int m = 64 * 32;
    const int group_size = 32;
    const int num_groups = m / group_size;

    // initialize data
    thrust::host_vector<uint8_t> x_h(m);
    thrust::host_vector<uint8_t> scales_h(num_groups);
    thrust::host_vector<cutlass::bfloat16_t> y_h(m);
    thrust::host_vector<cutlass::bfloat16_t> y_ref_h(m);

    for (int i = 0; i < m; ++i) {
        x_h[i] = (129 + i) % 256;
        y_h[i] = cutlass::bfloat16_t(0.0);
        y_ref_h[i] = cutlass::bfloat16_t(0.0);
        if (i % group_size == 0) {
            scales_h[i / group_size] = (128 + i) % 256;
        }
    }

    // CPU implementation
    mase_cuda::mxint8::dequantize::dequantize1d_host(x_h.data(), m, scales_h.data(), group_size, y_ref_h.data());

    // GPU implementation
    auto BLK_M = Int<8>{};
    auto BLK_K = Int<32>{};
    auto thd_m = Int<4>{};
    auto thd_k = BLK_K;

    thrust::device_vector<uint8_t> x_d = x_h;
    thrust::device_vector<uint8_t> scales_d = scales_h;
    thrust::device_vector<cutlass::bfloat16_t> y_d = y_h;

    auto shape_x = make_shape(m);
    auto stride_x = make_stride(Int<1>{});

    auto shape_scale = make_shape(num_groups);
    auto stride_scale = make_stride(Int<1>{});
    auto group_tiler = make_shape(group_size);

    auto cta_tiler = make_shape(BLK_M, BLK_K);
    auto layout_sX = make_layout(make_shape(BLK_M, BLK_K));
    auto layout_sScale = make_layout(make_shape(BLK_K));

    auto layout_tX = make_layout(make_shape(thd_m, thd_k));

    dim3 dimBlock(size(layout_tX));
    dim3 dimGrid(size(ceil_div(group_size, BLK_M)));
    mase_cuda::mxint8::dequantize::dequantize1d_device<<<dimGrid, dimBlock, 0, 0>>>(
        x_d.data().get(), shape_x, stride_x, scales_d.data().get(), shape_scale, stride_scale, group_tiler, cta_tiler,
        layout_sX, layout_sScale, layout_tX, y_d.data().get());

    // copy y_d back to y_h
    y_h = y_d;

    // compare results
    bool passed = true;
    for (int i = 0; i < m; ++i) {
        if (y_h[i] != y_ref_h[i]) {
            std::cout << "Mismatch at index " << i << "  " << y_h[i] << "  " << y_ref_h[i] << std::endl;
            passed = false;
        }
    }
    if (passed) {
        std::cout << "PASSED" << std::endl;
    } else {
        std::cout << "FAILED" << std::endl;
    }

    // dump results to json
    json j;
    std::vector<uint8_t> x_f(m);
    std::vector<uint8_t> scales_f(m);
    std::vector<float> y_cuda_f(m);
    std::vector<float> y_cpu_f(m);
    for (int i = 0; i < m; ++i) {
        x_f[i] = x_h[i];
        scales_f[i] = scales_h[i / group_size];
        y_cuda_f[i] = static_cast<float>(y_h[i]);
        y_cpu_f[i] = static_cast<float>(y_ref_h[i]);
    }
    j["x"] = x_f;
    j["scales"] = scales_f;
    j["y_cuda"] = y_cuda_f;
    j["y_cpu"] = y_cpu_f;
    std::ofstream file("dequantize1d.json");
    file << j.dump(4);
    file.close();
    std::cout << "Results dumped to dequantize1d.json" << std::endl;
    return 0;
}
