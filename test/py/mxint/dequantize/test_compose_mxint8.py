import logging
import json
import numpy as np
import pandas as pd
import ml_dtypes
from mase_cuda.tools.constants import MASE_CUDA_ROOT_PATH

logger = logging.getLogger(__name__)


def compose_mxint8_exp_mantissa(scale: np.ndarray, mantissa: np.ndarray):
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


def test_compose_mxint8_exp_mantissa():
    json_path = MASE_CUDA_ROOT_PATH.joinpath("build/test/cu/mxint/dequantize/dequantize1d.json").resolve()
    if not json_path.exists():
        logger.warning(f"test_compose_mxint8_exp_mantissa: {json_path} not found. Skip test.")
        return

    with open(json_path, "r") as f:
        de_j = json.load(f)

    mantissa = np.array(de_j["x"], dtype=np.uint8)
    scale = np.array(de_j["scales"], dtype=np.uint8)
    out_cuda = np.array(de_j["y_cuda"], dtype=np.float32)
    out_cpu = np.array(de_j["y_cpu"], dtype=np.float32)

    out_ref = compose_mxint8_exp_mantissa(scale, mantissa)
    out_ref = out_ref.astype(np.float32)

    # find mismatch
    idx_cuda_ref_ne = np.where(out_cuda != out_ref)
    idx_cpu_ref_ne = np.where(out_cpu != out_ref)

    assert len(idx_cuda_ref_ne[0]) == 0
    assert len(idx_cpu_ref_ne[0]) == 0

    logger.info("test_compose_mxint8_exp_mantissa: PASS")


if __name__ == "__main__":
    test_compose_mxint8_exp_mantissa()
