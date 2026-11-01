"""Structural (duck-typed) profile inputs for activity, scanner and lesion settings.

The package ships no profile data. Callers pass objects with these attributes;
the acquisition function validates scanner values, while activity and lesion
functions document their own input requirements. The registered
``fdg_pet_ct_mlx.medmlx`` adapter reads and validates caller-supplied JSON profiles.
Label IDs use the NV-Generate/MAISI vocabulary. SUV values follow the body-weight
convention, but no weight or dose normalization is applied here.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol


class TissueActivity(Protocol):
    """Activity distribution for one tissue class.

    Attributes:
        suv_mean: Mean SUV of the tissue.
        suv_std: Standard deviation of the per-voxel Gaussian SUV noise, in SUV.
        maisi_labels: MAISI label IDs belonging to this tissue.
    """

    @property
    def suv_mean(self) -> float:
        """Return this tissue's mean SUV under the body-weight convention."""
        ...
    @property
    def suv_std(self) -> float:
        """Return the per-voxel Gaussian SUV standard deviation for this tissue."""
        ...
    @property
    def maisi_labels(self) -> tuple[int, ...]:
        """Return the MAISI label IDs assigned to this tissue."""
        ...


class FdgOrganProfile(Protocol):
    """Organ-level activity: tissue classes keyed by name.

    ``tissues`` must contain ``"unclassified_soft_tissue"``, which supplies the mean
    and standard deviation for label IDs that no tissue lists. If a label ID is
    listed by several tissues, the one later in ``tissues`` supplies the standard
    deviation; the mean comes from ``lookup_table``.
    """

    @property
    def tissues(self) -> Mapping[str, TissueActivity]:
        """Map tissue names to activity specs; include ``unclassified_soft_tissue``."""
        ...
    def lookup_table(self, size: int) -> tuple[float, ...]:
        """Return the mean SUV for label IDs ``0..size-1`` as a tuple of length ``size``."""
        ...


class FdgScannerProfile(Protocol):
    """Image-space scanner approximation settings.

    Attributes:
        scatter_fraction: Finite fraction in [0, 1]; with ``randoms_fraction``, sets a
            uniform offset equal to the volume mean times their sum when that sum is positive.
        randoms_fraction: Unitless; see ``scatter_fraction``.
        fwhm_mm: Finite nonnegative Gaussian blur FWHM in mm, one per array axis.
            A value of 0 skips that axis.
        counts_per_suv: Positive finite expected counts per voxel per unit SUV at
            ``count_level=1``.
    """

    @property
    def scatter_fraction(self) -> float:
        """Return the finite scatter fraction in [0, 1] added as a uniform offset."""
        ...
    @property
    def randoms_fraction(self) -> float:
        """Return the finite randoms fraction in [0, 1] added as a uniform offset."""
        ...
    @property
    def fwhm_mm(self) -> tuple[float, float, float]:
        """Return the Gaussian blur FWHM in mm per array axis; 0 skips that axis."""
        ...
    @property
    def counts_per_suv(self) -> float:
        """Return positive finite expected counts per voxel per SUV unit at ``count_level=1``."""
        ...


class FdgLesionProfile(Protocol):
    """FDG-avid lesion settings.

    Attributes:
        avid_maisi_labels: MAISI label IDs whose voxels are treated as FDG-avid lesion.
        host_tissue: Maps each avid label ID to a key of ``FdgOrganProfile.tissues``;
            that tissue's ``suv_mean`` is the voxel's host (background) SUV. Every
            configured avid label must have a host entry.
        tumor_to_background: Finite nonnegative lesion mean SUV as a multiple of host SUV.
        heterogeneity: Finite nonnegative relative standard deviation of multiplicative
            Gaussian noise on lesion voxels; 0 disables it.
        necrosis_fraction: Target fraction in [0, 1] (by voxel count, over all avid
            voxels) of the lesion set to the necrotic core; 0 disables it.
        spillover_fwhm_mm: Finite nonnegative FWHM in mm of an isotropic Gaussian that
            spreads the lesion's excess SUV into neighboring voxels; 0 disables it.
    """

    @property
    def avid_maisi_labels(self) -> tuple[int, ...]:
        """Return the MAISI label IDs whose voxels are treated as FDG-avid lesion."""
        ...
    @property
    def host_tissue(self) -> Mapping[int, str]:
        """Map each avid label ID to a key of ``FdgOrganProfile.tissues``."""
        ...
    @property
    def tumor_to_background(self) -> float:
        """Return the lesion mean SUV as a finite nonnegative multiple of host SUV."""
        ...
    @property
    def heterogeneity(self) -> float:
        """Return the relative standard deviation of multiplicative lesion noise."""
        ...
    @property
    def necrosis_fraction(self) -> float:
        """Return the target necrotic-core fraction of avid voxels, by voxel count."""
        ...
    @property
    def spillover_fwhm_mm(self) -> float:
        """Return the finite nonnegative isotropic spill-over Gaussian FWHM in mm."""
        ...
