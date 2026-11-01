"""Image-space approximation of a reconstructed PET volume.

This applies simple multiplicative, additive, blur and noise steps directly to the
SUV image. It does not simulate projection physics, attenuation correction or
reconstruction.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from fdg_pet_ct_mlx._typing import FdgRuntime
from fdg_pet_ct_mlx.profiles import FdgScannerProfile

if TYPE_CHECKING:
    from mlx.core import array

# FWHM = 2*sqrt(2*ln 2) * sigma for a Gaussian (about 2.3548).
_FWHM_SIGMA = 2.0 * math.sqrt(2.0 * math.log(2.0))
# Metal can flush subnormal float32 values to zero. Count scales must remain
# positive and finite in the precision used for the noise and division.
_MIN_NORMAL_FLOAT32 = 2.0**-126
_MAX_FLOAT32 = float.fromhex("0x1.fffffep127")


def validate_count_scale(counts_per_suv: float, count_level: float) -> float:
    """Return the positive normal float32 count divisor, or reject invalid inputs."""
    if (
        not math.isfinite(counts_per_suv)
        or counts_per_suv <= 0
        or not math.isfinite(count_level)
        or count_level <= 0
    ):
        raise ValueError("counts_per_suv and count_level must be finite and positive")
    result = counts_per_suv * count_level
    if not _MIN_NORMAL_FLOAT32 <= result <= _MAX_FLOAT32:
        raise ValueError("count_level * counts_per_suv must be a positive normal float32 value")
    return result


def apply_acquisition_model(
    activity: array,
    ct_hu: array,
    *,
    scanner: FdgScannerProfile,
    count_level: float,
    spacing_mm: tuple[float, float, float],
    seed: int,
    mx: FdgRuntime,
) -> array:
    """Turn a synthetic activity volume into a PET-like SUV volume.

    Steps, in order, all in float32:

    1. Multiply by a CT-dependent factor ``exp(-0.02 * clip((HU + 1000) / 1000, 0, 3))``.
       This is a fixed heuristic, not a physical attenuation model: air gives 1.0 and
       HU of 2000 or more gives about 0.94.
    2. If ``scatter_fraction + randoms_fraction > 0``, add a uniform offset equal to
       that sum times the mean of the whole volume (background voxels included, so the
       offset depends on the grid extent).
    3. Blur with ``gaussian_blur`` using ``scanner.fwhm_mm`` and ``spacing_mm``.
    4. Convert to expected counts ``lam = SUV * counts_per_suv * count_level``, draw
       ``max(lam + sqrt(max(lam, 0)) * N(0, 1), 0)`` (a Gaussian approximation to
       Poisson noise, independent per voxel), and divide back by the same factor.

    Args:
        activity: Finite nonnegative SUV volume, shape ``(X, Y, Z)``.
        ct_hu: Finite CT in Hounsfield units on the same grid and in the same axis order as
            ``activity``. Registration and resampling are the caller's job.
        scanner: Scatter/randoms fractions in [0, 1], nonnegative finite blur FWHM,
            and positive finite counts per SUV.
        count_level: Positive finite multiplier on ``scanner.counts_per_suv``. Their
            product must be representable as a positive normal float32 value.
        spacing_mm: Finite positive voxel spacing in mm for each array axis.
        seed: Seed for the noise draw.
        mx: The ``mlx.core`` module.

    Returns:
        Evaluated finite float32 SUV array with the shape of ``activity``. Counts are
        floored at zero before division. Float32 overflow raises an error.

    Raises:
        ValueError: If the grids, input voxels, scanner parameters, count scale or
            resulting SUV values are invalid.
    """

    if activity.ndim != 3 or activity.size == 0 or activity.shape != ct_hu.shape:
        raise ValueError("activity and CT must share the same 3D grid and be nonempty")
    fractions = (scanner.scatter_fraction, scanner.randoms_fraction)
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in fractions):
        raise ValueError("scatter and randoms fractions must be finite and in [0, 1]")
    counts_per = validate_count_scale(scanner.counts_per_suv, count_level)
    valid_activity = mx.all(mx.isfinite(activity) & (activity >= 0))
    valid_ct = mx.all(mx.isfinite(ct_hu))
    mx.eval(valid_activity, valid_ct)
    if not bool(valid_activity.item()):
        raise ValueError("activity voxels must be finite and nonnegative")
    if not bool(valid_ct.item()):
        raise ValueError("CT voxels must be finite")
    hu = ct_hu.astype(mx.float32)
    shifted = (hu + mx.array(1000.0, dtype=mx.float32)) / mx.array(1000.0, dtype=mx.float32)
    mu = mx.clip(shifted, 0.0, 3.0)
    residual = mx.exp(mx.array(-0.02, dtype=mx.float32) * mu)
    volume = activity.astype(mx.float32) * residual
    floor_fraction = scanner.scatter_fraction + scanner.randoms_fraction
    if floor_fraction > 0.0:
        volume = volume + mx.mean(volume) * mx.array(floor_fraction, dtype=mx.float32)
    volume = gaussian_blur(volume, scanner.fwhm_mm, spacing_mm, mx)
    lam = volume * mx.array(counts_per, dtype=mx.float32)
    noise = mx.random.normal(volume.shape, dtype=mx.float32, key=mx.random.key(seed))
    zero = mx.array(0.0, dtype=mx.float32)
    counts = mx.maximum(lam + mx.sqrt(mx.maximum(lam, 0.0)) * noise, zero)
    suv = counts / mx.array(counts_per, dtype=mx.float32)
    finite = mx.all(mx.isfinite(suv))
    mx.eval(suv, finite)
    if not bool(finite.item()):
        raise ValueError("acquisition produced non-finite SUV values")
    return suv


def gaussian_blur(
    volume: array,
    fwhm_mm: tuple[float, float, float],
    spacing_mm: tuple[float, float, float],
    mx: FdgRuntime,
) -> array:
    """Blur a volume with a separable Gaussian, one 1-D pass per array axis.

    For each axis, ``sigma = fwhm / (2 * sqrt(2 * ln 2)) / spacing`` in voxels. The
    kernel is truncated at ``ceil(3 * sigma)`` voxels (at least 1) and normalized to
    sum to 1. The volume is zero-padded, so values near the borders are pulled toward
    zero. An axis with zero FWHM is skipped.

    Args:
        volume: Nonempty 3-D array.
        fwhm_mm: Finite nonnegative blur FWHM in mm for each array axis.
        spacing_mm: Finite positive voxel spacing in mm for each array axis.
        mx: The ``mlx.core`` module.

    Returns:
        Float32 array with the shape of ``volume``, already evaluated.

    Raises:
        ValueError: If the volume shape, FWHM or spacing is invalid.
    """
    if volume.ndim != 3 or volume.size == 0:
        raise ValueError("blur volume must be a nonempty 3D array")
    if len(fwhm_mm) != 3 or any(not math.isfinite(v) or v < 0 for v in fwhm_mm):
        raise ValueError("blur FWHM must contain three finite nonnegative values")
    if len(spacing_mm) != 3 or any(not math.isfinite(v) or v <= 0 for v in spacing_mm):
        raise ValueError("blur spacing must contain three finite positive values")
    blurred = volume.astype(mx.float32)
    for axis, (fwhm, spacing) in enumerate(zip(fwhm_mm, spacing_mm, strict=True)):
        sigma = (fwhm / _FWHM_SIGMA) / spacing
        if not math.isfinite(sigma):
            raise ValueError("blur FWHM / spacing exceeds the finite range")
        if sigma <= 0.0:
            continue
        blurred = _smooth_axis(blurred, _gaussian_kernel(sigma, mx), axis, mx)
    mx.eval(blurred)
    return blurred


def _gaussian_kernel(sigma: float, mx: FdgRuntime) -> array:
    # Truncate at 3 sigma; the kernel is symmetric, so conv1d's correlation equals
    # convolution.
    radius = max(1, math.ceil(3.0 * sigma))
    axis = mx.arange(-radius, radius + 1, dtype=mx.float32)
    scaled = axis / mx.array(sigma, dtype=mx.float32)
    kernel = mx.exp(mx.array(-0.5, dtype=mx.float32) * scaled**2)
    return kernel / mx.sum(kernel)


def _smooth_axis(volume: array, kernel: array, axis: int, mx: FdgRuntime) -> array:
    # MLX convolutions are channels-last: (batch, length, channels). Fold every
    # other axis into the batch and use a single channel.
    moved = mx.moveaxis(volume, axis, -1)
    length = moved.shape[-1]
    flat = mx.reshape(moved, (-1, length, 1))
    weight = mx.reshape(kernel, (1, -1, 1))
    pad = kernel.shape[0] // 2
    padded = mx.pad(flat, [(0, 0), (pad, pad), (0, 0)])
    out = mx.conv1d(padded, weight)
    return mx.moveaxis(mx.reshape(out, moved.shape), -1, axis)
