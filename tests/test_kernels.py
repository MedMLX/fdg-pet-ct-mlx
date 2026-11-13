"""Gaussian blur against SciPy's zero-padded Gaussian filter as an independent reference."""

import math
import os
from dataclasses import replace
from pathlib import Path
from typing import cast

import nibabel as nib
import numpy as np
import numpy.typing as npt
import pytest
from medmlx_core.runtime import import_mlx
from scipy.ndimage import gaussian_filter

from fdg_pet_ct_mlx._typing import FdgRuntime
from fdg_pet_ct_mlx.acquisition import apply_acquisition_model, gaussian_blur
from fdg_pet_ct_mlx.activity import synthesize_activity_map
from fdg_pet_ct_mlx.medmlx import Organs, Scanner, Tissue


@pytest.fixture(scope="module")
def real_ct_and_liver() -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Load aligned HCC-TACE CT and liver crops for acquisition checks."""
    root = Path.home() / ".cache/medmlx/test-data"
    ct = nib.load(os.environ.get("MEDMLX_TEST_CT", str(root / "hcc_063_ct.nii.gz")))
    liver = nib.load(os.environ.get("MEDMLX_TEST_MASK", str(root / "hcc_063_liver.nii.gz")))
    assert isinstance(ct, nib.Nifti1Image) and isinstance(liver, nib.Nifti1Image)
    assert ct.affine is not None and liver.affine is not None
    assert ct.shape == liver.shape and np.allclose(ct.affine, liver.affine)
    region = (slice(120, 127), slice(220, 228), slice(60, 69))
    return (
        np.asarray(ct.dataobj, dtype=np.float32)[region].copy(),
        np.asarray(liver.dataobj, dtype=np.float32)[region].copy(),
    )


@pytest.mark.parametrize(
    "fwhm,spacing",
    [
        ((2.354820045,) * 3, (1.0,) * 3),
        ((12.0, 0.0, 0.0), (1.0,) * 3),
        ((0.0, 12.0, 0.0), (1.0,) * 3),
        ((0.0, 0.0, 12.0), (1.0,) * 3),
        ((2.0, 3.0, 4.0), (0.7, 1.3, 2.1)),
        ((0.0,) * 3, (1.0,) * 3),
    ],
)
def test_blur_matches_scipy_zero_padding(
    fwhm: tuple[float, float, float],
    spacing: tuple[float, float, float],
    real_ct_and_liver: tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]],
) -> None:
    mx = cast(FdgRuntime, import_mlx())
    values = real_ct_and_liver[0] / np.float32(1024)
    sigma = tuple(
        f / (2 * math.sqrt(2 * math.log(2))) / s for f, s in zip(fwhm, spacing, strict=True)
    )
    radius = tuple(max(1, math.ceil(3 * s)) if s else 0 for s in sigma)
    expected = gaussian_filter(values, sigma=sigma, radius=radius, mode="constant", cval=0.0)
    result = gaussian_blur(mx.array(values), fwhm, spacing, mx)
    np.testing.assert_allclose(np.asarray(result, dtype=np.float32), expected, atol=2e-7, rtol=2e-6)


@pytest.mark.parametrize(
    "problem", ["boolean", "complex", "nonfinite", "negative", "range", "fraction", "empty", "ndim"]
)
def test_public_activity_rejects_invalid_labels(
    problem: str, real_ct_and_liver: tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]
) -> None:
    """The independently callable activity API keeps validating native inputs."""
    mx = cast(FdgRuntime, import_mlx())
    values = real_ct_and_liver[1][:2, :3, :4].copy()
    if problem == "boolean":
        labels = mx.array(values, dtype=mx.bool_)
    elif problem == "complex":
        labels = mx.array(values.astype(np.complex64))
    elif problem == "empty":
        labels = mx.array(values[:0])
    elif problem == "ndim":
        labels = mx.array(values[0])
    else:
        values[0, 0, 0] = {
            "nonfinite": float("nan"),
            "negative": -1.0,
            "range": 2.0**31,
            "fraction": 1.5,
        }[problem]
        labels = mx.array(values)
    profile = Organs({"unclassified_soft_tissue": Tissue(2.0, 0.3, ())})
    with pytest.raises(ValueError):
        synthesize_activity_map(labels, profile, seed=1, mx=mx)


@pytest.mark.parametrize(
    "problem", ["activity", "negative", "ct", "grid", "scatter", "counts", "spacing", "fwhm"]
)
def test_public_acquisition_rejects_invalid_inputs(
    problem: str, real_ct_and_liver: tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]
) -> None:
    """The public acquisition API validates both grids and scanner domains."""
    mx = cast(FdgRuntime, import_mlx())
    ct = real_ct_and_liver[0][:2, :3, :4].copy()
    activity = np.maximum((ct + np.float32(1000)) / np.float32(1000), np.float32(0))
    scanner = Scanner(0.0, 0.0, (0.0, 0.0, 0.0), 100.0)
    spacing = (1.0, 1.0, 1.0)
    if problem in ("activity", "negative"):
        activity[0, 0, 0] = float("nan") if problem == "activity" else -1.0
    elif problem == "ct":
        ct[0, 0, 0] = float("inf")
    elif problem == "grid":
        ct = ct[:1]
    elif problem == "scatter":
        scanner = replace(scanner, scatter_fraction=2.0)
    elif problem == "counts":
        scanner = replace(scanner, counts_per_suv=0.0)
    elif problem == "spacing":
        spacing = (1.0, 0.0, 1.0)
    else:
        scanner = replace(scanner, fwhm_mm=(-1.0, 0.0, 0.0))
    with pytest.raises(ValueError):
        apply_acquisition_model(
            mx.array(activity),
            mx.array(ct),
            scanner=scanner,
            count_level=1.0,
            spacing_mm=spacing,
            seed=2,
            mx=mx,
        )
