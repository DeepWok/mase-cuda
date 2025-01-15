import time
import logging
import json
import tabulate
import numpy as np
import pytest
import ml_dtypes
import torch
from mase_cuda.constants import MASE_CUDA_ROOT_PATH

from mase_cuda.mxint8.dequantize import dequantize1d_fast

logger = logging.getLogger(__name__)


def compose_mxint8_exp_mantissa_np(scale: np.ndarray, mantissa: np.ndarray):
    assert scale.dtype == np.uint8
    assert mantissa.dtype == np.uint8
    assert scale.shape == mantissa.shape
    mxint8 = np.zeros(scale.shape, dtype=np.uint16)
    sign = (mantissa & 0x80).astype(np.uint16) << 8
    mantissa = ((mantissa << 1) & 0x7E).astype(np.uint16)
    exp = scale.astype(np.uint16) << 7
    mxint8 = sign | exp | mantissa
    mxint8 = mxint8.view(ml_dtypes.bfloat16)

    return mxint8


def dequantize1d_fake_pt(x: torch.Tensor, scales: torch.Tensor, group_size: int) -> torch.Tensor:
    assert x.dtype == torch.uint8
    assert scales.dtype == torch.uint8
    assert x.shape[0] // group_size == scales.shape[0]

    m = x.shape[0]
    num_groups = m // group_size

    x = x.reshape(num_groups, group_size)
    scales = scales.view(num_groups, 1)

    mxint8 = torch.zeros_like(x, dtype=torch.uint16)

    sign = (x & 0x80).to(torch.int16) << 8
    exp = scales.to(torch.int16) << 7
    mantissa = ((x << 1) & 0x7E).to(torch.int16)

    mxint8 = sign | exp | mantissa
    mxint8 = mxint8.view(torch.bfloat16)

    mxint8 = mxint8.reshape(m)
    return mxint8


def test_ext_dequantize1d():
    group_sizes = [8, 16, 32, 64, 128, 256, 512]
    num_elements = [1024, 2048, 4096, 8192, 16384]
    num_random_tests = 100

    for group_size in group_sizes:
        for _ in range(num_random_tests):
            for m in num_elements:
                # m = 4096
                # m = 8192
                num_groups = m // group_size
                x = torch.empty(m, dtype=torch.uint8).random_(0, 256)
                scales = torch.empty(num_groups, dtype=torch.uint8).random_(0, 255)  # max=254 to avoid NaN
                scales_dup = scales.repeat_interleave(group_size)

                # view as uint16 to avoid NaN comparison
                out_cpu = dequantize1d_fast(x, scales, group_size)
                out_ref = dequantize1d_fake_pt(x, scales, group_size)

                # find mismatch idx
                if not torch.equal(out_cpu, out_ref):
                    mismatch_idx = torch.where(torch.logical_not(torch.eq(out_cpu, out_ref)))
                    logger.error(f"m: {m}, group_size: {group_size}")
                    logger.error(f"mismatch_idx: {mismatch_idx}")
                    logger.error(f"x[mismatch_idx]: {x[mismatch_idx]}")
                    logger.error(f"scales[mismatch_idx]: {scales_dup[mismatch_idx]}")
                    logger.error(f"out_cpu[mismatch_idx]: {out_cpu[mismatch_idx]}")
                    logger.error(f"out_ref[mismatch_idx]: {out_ref[mismatch_idx]}")

                out_gpu = dequantize1d_fast(x.cuda(), scales.cuda(), group_size).cpu()

                if not torch.equal(out_gpu, out_ref):
                    mismatch_idx = torch.where(torch.logical_not(torch.eq(out_gpu, out_ref)))
                    logger.error(f"m: {m}, group_size: {group_size}")
                    logger.error(f"mismatch_idx: {mismatch_idx}")
                    logger.error(f"x[mismatch_idx]: {x[mismatch_idx]}")
                    logger.error(f"scales[mismatch_idx]: {scales_dup[mismatch_idx]}")
                    logger.error(f"out_gpu[mismatch_idx]: {out_gpu[mismatch_idx]}")
                    logger.error(f"out_ref[mismatch_idx]: {out_ref[mismatch_idx]}")

                assert torch.equal(out_cpu, out_ref)
                assert torch.equal(out_gpu, out_ref)
    logger.info("test_ext_dequantize1d: PASS")


@pytest.mark.slow
def test_ext_dequantize1d_latency():
    # measure latecy of dequantize1d on cpu and gpu

    group_sizes = [8, 16, 32, 64, 128, 256, 512]
    num_elements = [1024, 512 * 4096, 4096 * 4096, 4096 * 14336, 8192 * 28672]
    n_repeats = 20

    results = []
    for m in num_elements:
        for group_size in group_sizes:
            num_groups = m // group_size
            x = torch.empty(m, dtype=torch.uint8).random_(0, 256)
            scales = torch.empty(num_groups, dtype=torch.uint8).random_(0, 256)

            # cpu
            start = time.time()
            for _ in range(n_repeats):
                out_cpu = dequantize1d_fast(x, scales, group_size)
            end = time.time()
            latency_cpu = (end - start) / n_repeats  # s

            # gpu
            x = x.cuda()
            scales = scales.cuda()
            start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_repeats)]
            end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_repeats)]
            for i in range(n_repeats):
                start_events[i].record()
                out_gpu = dequantize1d_fast(x, scales, group_size)
                end_events[i].record()
            torch.cuda.synchronize()
            latencies_gpu = [start_events[i].elapsed_time(end_events[i]) for i in range(n_repeats)]
            latency_gpu = sum(latencies_gpu) / n_repeats / 1000  # ms -> s
            results.append([m, group_size, latency_cpu, latency_gpu, latency_cpu / latency_gpu])

    headers = ["m", "group_size", "latency_cpu", "latency_gpu", "GPU speedup"]
    logger.info("\n{}".format(tabulate.tabulate(results, headers=headers, tablefmt="pretty", floatfmt=".3E")))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_ext_dequantize1d()
    test_ext_dequantize1d_latency()
