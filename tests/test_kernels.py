"""Gaussian blur against SciPy's zero-padded Gaussian filter as an independent reference."""

from typing import cast

import numpy as np
from medmlx_core.runtime import import_mlx
from scipy.ndimage import gaussian_filter

from fdg_pet_ct_mlx._typing import FdgRuntime
from fdg_pet_ct_mlx.acquisition import gaussian_blur


def test_blur_matches_scipy_zero_padding() -> None:
    mx = cast(FdgRuntime, import_mlx())
    values = np.random.default_rng(21).normal(size=(7, 8, 9)).astype(np.float32)
    expected = gaussian_filter(values, sigma=1.0, radius=3, mode="constant", cval=0.0)
    result = gaussian_blur(
        mx.array(values), (2.354820045, 2.354820045, 2.354820045), (1.0, 1.0, 1.0), mx
    )
    np.testing.assert_allclose(np.asarray(result, dtype=np.float32), expected, atol=2e-7, rtol=2e-6)
