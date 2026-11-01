"""FDG-avid lesion uptake applied to an activity volume, selected by label ID.

Lesion geometry comes only from the label volume. A requested necrotic core that
the labeled lesion cannot support raises ``InvalidInputError``.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

from medmlx_core.errors import InvalidInputError

from fdg_pet_ct_mlx._typing import FdgRuntime
from fdg_pet_ct_mlx.acquisition import gaussian_blur
from fdg_pet_ct_mlx.activity import validated_label_ids
from fdg_pet_ct_mlx.profiles import FdgLesionProfile, FdgOrganProfile

if TYPE_CHECKING:
    from mlx.core import array


def apply_lesion_uptake(
    activity: array,
    labels: array,
    *,
    organ: FdgOrganProfile,
    lesion: FdgLesionProfile,
    spacing_mm: tuple[float, float, float],
    seed: int,
    mx: FdgRuntime,
) -> array:
    """Replace activity in FDG-avid lesion voxels with lesion uptake.

    Let ``host`` be the ``suv_mean`` of the tissue that ``lesion.host_tissue`` assigns
    to each avid voxel's label. In avid voxels the input activity is discarded and
    replaced by:

    1. ``host * tumor_to_background``;
    2. if ``heterogeneity > 0``, times ``max(1 + heterogeneity * N(0, 1), 0)``, with
       independent draws per voxel;
    3. if ``necrosis_fraction > 0``, ``0.2 * host`` in the necrotic core (the 0.2 is a
       fixed constant, not a profile field). The core is what remains after
       repeatedly eroding the union of all avid voxels with a full 3x3x3 neighborhood
       until its voxel count is at most ``necrosis_fraction`` times the avid count.
       Erosion removes whole shells, so the core can be smaller than requested.
       Voxels at the volume border erode because the mask is zero-padded;
    4. if ``spillover_fwhm_mm > 0``, avid voxels are reset to ``host`` and the excess
       ``(step 3 result - host)`` (negative in a necrotic core), blurred with an
       isotropic Gaussian (``gaussian_blur``), is added to the whole volume, so lesion
       SUV leaks into neighbors.

    The result is clipped at zero. Voxels whose label is not in
    ``avid_maisi_labels`` (for example non-avid cysts) keep their input activity before
    clipping, apart from spill-over.

    Args:
        activity: SUV volume, shape ``(X, Y, Z)``, as from ``synthesize_activity_map``.
        labels: Nonempty MAISI label volume on the same grid and axis order as
            ``activity``, with finite, integral IDs in ``[0, 2**31 - 1]``.
        organ: Organ profile; supplies the host SUV means.
        lesion: Lesion profile.
        spacing_mm: Voxel spacing in mm for each array axis (used for spill-over).
        seed: Seed for the heterogeneity noise; unused when ``heterogeneity <= 0``
            or there are no avid voxels.
        mx: The ``mlx.core`` module.

    Returns:
        When avid voxels are present, an evaluated float32 SUV array. If no voxel
        carries an avid label and ``necrosis_fraction <= 0``, ``activity`` itself is
        returned unchanged: its dtype and evaluation state are retained, and it is not copied.

    Raises:
        InvalidInputError: If ``necrosis_fraction > 0`` and there are no avid voxels,
            or the lesion is too small to erode down to the requested core. This is a
            ``ValueError`` subclass.
        ValueError: If the arrays do not share a 3-D grid, the labels are invalid,
            or the lesion profile lacks a host or has out-of-range values.
        KeyError: If ``lesion.host_tissue`` names a tissue missing from ``organ.tissues``.
    """

    if activity.ndim != 3 or activity.shape != labels.shape:
        raise ValueError("activity and labels must share the same 3D grid")
    settings = (
        ("tumor_to_background", lesion.tumor_to_background),
        ("heterogeneity", lesion.heterogeneity),
        ("spillover_fwhm_mm", lesion.spillover_fwhm_mm),
    )
    for name, value in settings:
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"lesion {name} must be finite and nonnegative")
    if not math.isfinite(lesion.necrosis_fraction) or not 0 <= lesion.necrosis_fraction <= 1:
        raise ValueError("lesion necrosis_fraction must be finite and in [0, 1]")
    missing_hosts = [label for label in lesion.avid_maisi_labels if label not in lesion.host_tissue]
    if missing_hosts:
        raise ValueError(f"lesion host_tissue is missing avid label IDs: {missing_hosts}")
    ids = validated_label_ids(labels, mx, context="lesion")
    avid = _label_mask(ids, lesion.avid_maisi_labels, mx)
    mx.eval(avid)
    avid_voxels = int(cast(int, mx.sum(avid).item()))
    if avid_voxels == 0:
        if lesion.necrosis_fraction > 0.0:
            raise InvalidInputError(
                "necrosis_fraction requires FDG-avid lesion voxels in the label volume"
            )
        return activity

    host = _host_map(ids, organ, lesion, mx)
    target = host * mx.array(lesion.tumor_to_background, dtype=mx.float32)
    if lesion.heterogeneity > 0.0:
        noise = mx.random.normal(ids.shape, dtype=mx.float32, key=mx.random.key(seed))
        scale = (
            mx.array(1.0, dtype=mx.float32)
            + mx.array(lesion.heterogeneity, dtype=mx.float32) * noise
        )
        target = target * mx.maximum(scale, mx.array(0.0, dtype=mx.float32))
    if lesion.necrosis_fraction > 0.0:
        necrotic = _necrosis_mask(avid, lesion.necrosis_fraction, mx)
        target = mx.where(
            necrotic,
            host * mx.array(0.2, dtype=mx.float32),
            target,
        )
    updated = mx.where(avid, target, activity)
    if lesion.spillover_fwhm_mm > 0.0:
        excess = mx.where(avid, updated - host, mx.zeros_like(updated))
        spilled = gaussian_blur(
            excess,
            (
                lesion.spillover_fwhm_mm,
                lesion.spillover_fwhm_mm,
                lesion.spillover_fwhm_mm,
            ),
            spacing_mm,
            mx,
        )
        updated = mx.where(avid, host, activity) + spilled
    result = mx.maximum(updated, mx.array(0.0, dtype=mx.float32))
    mx.eval(result)
    return result


def _label_mask(ids: array, labels: tuple[int, ...], mx: FdgRuntime) -> array:
    mask = mx.zeros(ids.shape, dtype=mx.bool_)
    for label in labels:
        mask = mx.logical_or(mask, ids == label)
    return mask


def _host_map(
    ids: array,
    organ: FdgOrganProfile,
    lesion: FdgLesionProfile,
    mx: FdgRuntime,
) -> array:
    # Per-voxel host-tissue SUV mean; voxels whose label is not a key of
    # ``lesion.host_tissue`` are 0.
    host = mx.zeros(ids.shape, dtype=mx.float32)
    for label, tissue in lesion.host_tissue.items():
        value = mx.array(organ.tissues[tissue].suv_mean, dtype=mx.float32)
        host = mx.where(ids == label, value, host)
    return host


def _necrosis_mask(avid: array, fraction: float, mx: FdgRuntime) -> array:
    """Return the avid voxels that survive repeated 3x3x3 erosion down to ``fraction``.

    Erosion stops at the first shell count not above ``fraction`` of the avid voxels.
    Raises ``InvalidInputError`` if erosion would empty the lesion or stops shrinking.
    """

    remaining = avid.astype(mx.float32)
    # The mask is float32, so .item() returns a Python float; the cast only narrows the
    # type for the checker.
    core_target = float(cast(float, mx.sum(remaining).item())) * fraction
    if core_target <= 0.0:
        raise InvalidInputError("necrosis_fraction leaves no remaining lesion voxels")
    while float(cast(float, mx.sum(remaining).item())) > core_target:
        eroded = _erode(remaining, mx)
        eroded_count = float(cast(float, mx.sum(eroded).item()))
        current = float(cast(float, mx.sum(remaining).item()))
        if eroded_count <= 0.0 or eroded_count >= current:
            raise InvalidInputError(
                "necrosis_fraction requires a lesion large enough to erode to the "
                "requested necrotic core fraction"
            )
        remaining = eroded
    return mx.logical_and(avid, remaining >= mx.array(0.5, dtype=mx.float32))


def _erode(mask: array, mx: FdgRuntime) -> array:
    # Binary erosion by a full 3x3x3 box on a 3-D mask; outside the volume counts as 0.
    volume = mask.astype(mx.float32)[None, ..., None]
    weight = mx.ones((1, 3, 3, 3, 1), dtype=mx.float32)
    padded = mx.pad(volume, [(0, 0), (1, 1), (1, 1), (1, 1), (0, 0)])
    neighbor_sum = mx.conv3d(padded, weight)
    # A voxel survives only if all 27 voxels of its 3x3x3 box are set; 26.5 avoids
    # an exact float comparison against 27.
    return (neighbor_sum[0, ..., 0] >= mx.array(26.5, dtype=mx.float32)).astype(mx.float32)
