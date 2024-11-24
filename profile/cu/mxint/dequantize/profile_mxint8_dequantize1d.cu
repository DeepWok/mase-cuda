#include "mxint/dequantize.cuh"
#include <cstdint>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cute/layout.hpp>
#include <cute/numeric/integral_constant.hpp>
#include <cute/numeric/numeric_types.hpp>
#include <cutlass/bfloat16.h>
#include <iostream>
#include <nlohmann/json.hpp>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

int main(int argc, char **argv) {
    using namespace cute;
    using namespace mase_cuda::mxint8::dequantize;
    int m = 5120;
    int group_size = 32;
    if (argc >= 2)
        sscanf(argv[1], "%d", &m);

    if (argc >= 3)
        sscanf(argv[2], "%d", &group_size);

    std::cout << "m: " << m << std::endl;
    std::cout << "group_size: " << group_size << std::endl;

    const int num_groups = m / group_size;

    thrust::host_vector<uint8_t> x_h(m);
    thrust::host_vector<uint8_t> scales_h(num_groups);
    thrust::host_vector<cutlass::bfloat16_t> y_h(m);

    for (int i = 0; i < num_groups; i++) {
        x_h[i] = rand() % 256;
        y_h[i] = cutlass::bfloat16_t(0.0);
        if (i % group_size == 0) {
            scales_h[i] = rand() % 255;
        }
    }

    auto shape_x = make_shape(m);
    auto stride_x = make_stride(Int<1>{});
    auto shape_scale = make_shape(num_groups);
    auto stride_scale = make_stride(Int<1>{});
    auto group_tiler = make_shape(group_size);

    thrust::device_vector<uint8_t> x_d = x_h;
    thrust::device_vector<uint8_t> scales_d = scales_h;
    thrust::device_vector<cutlass::bfloat16_t> y_d = y_h;

    const uint8_t *x_ptr = x_d.data().get();
    const uint8_t *scales_ptr = scales_d.data().get();
    auto y_ptr = y_d.data().get();

    if (group_size == 8) {
        auto BLK_M = Int<8>{};
        auto BLK_K = Int<128>{};
        auto THD_M = Int<8>{};
        auto cta_tiler = make_shape(BLK_M, BLK_K);
        auto layout_sX = make_layout(make_shape(BLK_M, BLK_K));
        auto layout_sScale = make_layout(make_shape(BLK_K));
        auto layout_tX = make_layout(make_shape(THD_M, BLK_K));
        dim3 dimBlock(size(layout_tX));
        dim3 dimGrid(size(ceil_div(group_size, BLK_M)));
        dequantize1d_device<<<dimGrid, dimBlock, 0, 0>>>(x_ptr, shape_x, stride_x, scales_ptr, shape_scale,
                                                         stride_scale, group_tiler, cta_tiler, layout_sX, layout_sScale,
                                                         layout_tX, y_ptr);
    } else if (group_size == 16) {
        auto BLK_M = Int<16>{};
        auto BLK_K = Int<64>{};
        auto THD_M = Int<16>{};
        auto cta_tiler = make_shape(BLK_M, BLK_K);
        auto layout_sX = make_layout(make_shape(BLK_M, BLK_K));
        auto layout_sScale = make_layout(make_shape(BLK_K));
        auto layout_tX = make_layout(make_shape(THD_M, BLK_K));
        dim3 dimBlock(size(layout_tX));
        dim3 dimGrid(size(ceil_div(group_size, BLK_M)));
        dequantize1d_device<<<dimGrid, dimBlock, 0, 0>>>(x_ptr, shape_x, stride_x, scales_ptr, shape_scale,
                                                         stride_scale, group_tiler, cta_tiler, layout_sX, layout_sScale,
                                                         layout_tX, y_ptr);
    } else if (group_size == 32) {
        auto BLK_M = Int<32>{};
        auto BLK_K = Int<32>{};
        auto THD_M = Int<32>{};
        auto cta_tiler = make_shape(BLK_M, BLK_K);
        auto layout_sX = make_layout(make_shape(BLK_M, BLK_K));
        auto layout_sScale = make_layout(make_shape(BLK_K));
        auto layout_tX = make_layout(make_shape(THD_M, BLK_K));
        dim3 dimBlock(size(layout_tX));
        dim3 dimGrid(size(ceil_div(group_size, BLK_M)));
        dequantize1d_device<<<dimGrid, dimBlock, 0, 0>>>(x_ptr, shape_x, stride_x, scales_ptr, shape_scale,
                                                         stride_scale, group_tiler, cta_tiler, layout_sX, layout_sScale,
                                                         layout_tX, y_ptr);
    } else if (group_size == 64) {
        auto BLK_M = Int<64>{};
        auto BLK_K = Int<16>{};
        auto THD_M = Int<64>{};
        auto cta_tiler = make_shape(BLK_M, BLK_K);
        auto layout_sX = make_layout(make_shape(BLK_M, BLK_K));
        auto layout_sScale = make_layout(make_shape(BLK_K));
        auto layout_tX = make_layout(make_shape(THD_M, BLK_K));
        dim3 dimBlock(size(layout_tX));
        dim3 dimGrid(size(ceil_div(group_size, BLK_M)));
        dequantize1d_device<<<dimGrid, dimBlock, 0, 0>>>(x_ptr, shape_x, stride_x, scales_ptr, shape_scale,
                                                         stride_scale, group_tiler, cta_tiler, layout_sX, layout_sScale,
                                                         layout_tX, y_ptr);
    } else if (group_size == 128) {
        auto BLK_M = Int<128>{};
        auto BLK_K = Int<8>{};
        auto THD_M = Int<64>{};
        auto cta_tiler = make_shape(BLK_M, BLK_K);
        auto layout_sX = make_layout(make_shape(BLK_M, BLK_K));
        auto layout_sScale = make_layout(make_shape(BLK_K));
        auto layout_tX = make_layout(make_shape(THD_M, BLK_K));
        dim3 dimBlock(size(layout_tX));
        dim3 dimGrid(size(ceil_div(group_size, BLK_M)));
        dequantize1d_device<<<dimGrid, dimBlock, 0, 0>>>(x_ptr, shape_x, stride_x, scales_ptr, shape_scale,
                                                         stride_scale, group_tiler, cta_tiler, layout_sX, layout_sScale,
                                                         layout_tX, y_ptr);

    } else if (group_size == 256) {
        // group_size == 256
        auto BLK_M = Int<256>{};
        auto BLK_K = Int<4>{};
        auto THD_M = Int<128>{};
        auto cta_tiler = make_shape(BLK_M, BLK_K);
        auto layout_sX = make_layout(make_shape(BLK_M, BLK_K));
        auto layout_sScale = make_layout(make_shape(BLK_K));
        auto layout_tX = make_layout(make_shape(THD_M, BLK_K));
        dim3 dimBlock(size(layout_tX));
        dim3 dimGrid(size(ceil_div(group_size, BLK_M)));
        dequantize1d_device<<<dimGrid, dimBlock, 0, 0>>>(x_ptr, shape_x, stride_x, scales_ptr, shape_scale,
                                                         stride_scale, group_tiler, cta_tiler, layout_sX, layout_sScale,
                                                         layout_tX, y_ptr);
    } else {
        // group_size == 512
        auto BLK_M = Int<256>{};
        auto BLK_K = Int<4>{};
        auto THD_M = Int<128>{};
        auto cta_tiler = make_shape(BLK_M, BLK_K);
        auto layout_sX = make_layout(make_shape(BLK_M, BLK_K));
        auto layout_sScale = make_layout(make_shape(BLK_K));
        auto layout_tX = make_layout(make_shape(THD_M, BLK_K));
        dim3 dimBlock(size(layout_tX));
        dim3 dimGrid(size(ceil_div(group_size, BLK_M)));
        dequantize1d_device<<<dimGrid, dimBlock, 0, 0>>>(x_ptr, shape_x, stride_x, scales_ptr, shape_scale,
                                                         stride_scale, group_tiler, cta_tiler, layout_sX, layout_sScale,
                                                         layout_tX, y_ptr);
    }

    std::cout << "Done" << std::endl;
}