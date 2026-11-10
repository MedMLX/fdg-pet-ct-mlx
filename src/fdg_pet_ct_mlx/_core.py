"""Shared native synthesis paths behind the validated public APIs and file runner."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from fdg_pet_ct_mlx._typing import FdgRuntime
from fdg_pet_ct_mlx.profiles import FdgOrganProfile, FdgScannerProfile

if TYPE_CHECKING:
    from mlx.core import array

# MAISI label IDs run 0..200 (200 is the body envelope label in mlx-generate-ctmr masks), so the
# lookup tables always cover at least that range.
_LOOKUP_MIN = 201


def sample_activity(
    ids: array, profile: FdgOrganProfile, *, seed: int, mx: FdgRuntime
) -> array:
    """Sample activity from nonempty int32 IDs already validated by the caller."""
    profile_labels = [label for spec in profile.tissues.values() for label in spec.maisi_labels]
    size = max(max(profile_labels, default=0) + 1, _LOOKUP_MIN)
    # Send every ID >= size to one extra table slot holding the unclassified values, so
    # table size does not depend on the largest input ID.
    ids = mx.where(ids < size, ids, size)
    default = profile.tissues["unclassified_soft_tissue"]
    means = mx.array((*profile.lookup_table(size), default.suv_mean), dtype=mx.float32)
    stds = mx.array((*_std_table(profile, size), default.suv_std), dtype=mx.float32)
    base = means[ids]
    noise = mx.random.normal(ids.shape, dtype=mx.float32, key=mx.random.key(seed))
    activity = mx.maximum(base + stds[ids] * noise, mx.array(0.0, dtype=mx.float32))
    mx.eval(activity)
    return activity


def _std_table(profile: FdgOrganProfile, size: int) -> tuple[float, ...]:
    # Standard-deviation counterpart of ``profile.lookup_table(size)``.
    unclassified = profile.tissues["unclassified_soft_tissue"].suv_std
    table = [unclassified] * size
    for spec in profile.tissues.values():
        for label in spec.maisi_labels:
            if label < size:
                table[label] = spec.suv_std
    return tuple(table)


def acquire_activity(
    activity: array,
    ct_hu: array,
    *,
    scanner: FdgScannerProfile,
    count_level: float,
    spacing_mm: tuple[float, float, float],
    seed: int,
    mx: FdgRuntime,
    validate_ct: bool,
) -> array:
    """Acquire PET, omitting the CT scan only for the runner's validated host input.

    Generated activity is still checked: even finite profile values can overflow
    when converted to the native float32 arithmetic.
    """

    from fdg_pet_ct_mlx.acquisition import gaussian_blur, validate_count_scale

    if activity.ndim != 3 or activity.size == 0 or activity.shape != ct_hu.shape:
        raise ValueError("activity and CT must share the same 3D grid and be nonempty")
    fractions = (scanner.scatter_fraction, scanner.randoms_fraction)
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in fractions):
        raise ValueError("scatter and randoms fractions must be finite and in [0, 1]")
    counts_per = validate_count_scale(scanner.counts_per_suv, count_level)
    valid_activity = mx.all(mx.isfinite(activity) & (activity >= 0))
    valid_ct = mx.all(mx.isfinite(ct_hu)) if validate_ct else None
    mx.eval(*(value for value in (valid_activity, valid_ct) if value is not None))
    if not bool(valid_activity.item()):
        raise ValueError("activity voxels must be finite and nonnegative")
    if valid_ct is not None and not bool(valid_ct.item()):
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
