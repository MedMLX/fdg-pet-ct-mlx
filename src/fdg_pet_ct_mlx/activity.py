"""Per-voxel FDG activity (SUV) drawn from tissue statistics selected by label ID."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from fdg_pet_ct_mlx._core import sample_activity
from fdg_pet_ct_mlx._typing import FdgRuntime
from fdg_pet_ct_mlx.profiles import FdgOrganProfile

if TYPE_CHECKING:
    from mlx.core import array



def validated_label_ids(labels: array, mx: FdgRuntime, *, context: str) -> array:
    """Return int32 MAISI IDs after validating a nonempty 3-D label grid.

    Boolean, complex, nonfinite, fractional and out-of-range values raise
    ``ValueError`` before casting. ``context`` identifies the caller in the error.
    """
    if labels.ndim != 3:
        raise ValueError(f"{context} labels must be a 3D volume")
    if labels.size == 0:
        raise ValueError(f"{context} labels must be nonempty")
    if labels.dtype == mx.complex64 or labels.dtype == mx.bool_:
        raise ValueError(f"{context} labels must contain real integral identifiers")
    finite = mx.all(mx.isfinite(labels))
    lowest = mx.min(labels)
    highest = mx.max(labels)
    integral = mx.all(labels == mx.floor(labels))
    mx.eval(finite, lowest, highest, integral)
    if not bool(finite.item()):
        raise ValueError(f"{context} labels must be finite")
    lowest_value = cast(int | float, lowest.item())
    highest_value = cast(int | float, highest.item())
    if lowest_value < 0 or highest_value > 2**31 - 1:
        raise ValueError(f"{context} labels must be in the nonnegative int32 range")
    if not bool(integral.item()):
        raise ValueError(f"{context} labels must be integral")
    return labels.astype(mx.int32)


def synthesize_activity_map(
    labels: array,
    profile: FdgOrganProfile,
    *,
    seed: int,
    mx: FdgRuntime,
) -> array:
    """Draw a nonnegative float32 SUV volume on the label grid.

    Each voxel is ``max(mean[label] + std[label] * N(0, 1), 0)``, with independent
    unit-normal draws per voxel and the profile's SUV mean and standard deviation for
    that label. Means come from ``profile.lookup_table`` and standard deviations from
    ``profile.tissues``. IDs that no tissue lists get the ``unclassified_soft_tissue``
    standard deviation and, by the lookup-table convention, its mean. IDs at or above
    the lookup size use the unclassified mean and standard deviation directly.
    Profile values and lookup-table contents are caller-owned and are not
    checked for finiteness or valid activity statistics.

    Args:
        labels: MAISI label volume, shape ``(X, Y, Z)``. Any real integer or float
            dtype is accepted if every value is finite, integral and in
            ``[0, 2**31 - 1]``.
        profile: Organ activity profile.
        seed: Seed for the noise draw.
        mx: The ``mlx.core`` module.

    Returns:
        Float32 SUV array with the shape of ``labels``, already evaluated.

    Raises:
        ValueError: If ``labels`` is not 3-D, is empty, is boolean or complex, or has
            non-finite, negative, out-of-range or non-integral values.
        KeyError: If ``profile.tissues`` has no ``"unclassified_soft_tissue"`` entry.
    """

    ids = validated_label_ids(labels, mx, context="activity")
    return sample_activity(ids, profile, seed=seed, mx=mx)
