"""Kernel tests on small synthetic volumes.

Gaussian blur is compared with SciPy; activity, acquisition and lesion behavior is
checked by property (shape, ordering, error cases) rather than against a reference
PET implementation.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar, cast

import numpy as np
import pytest
from medmlx_core.runtime import import_mlx
from scipy.ndimage import gaussian_filter

from fdg_pet_ct_mlx._typing import FdgRuntime
from fdg_pet_ct_mlx.acquisition import apply_acquisition_model, gaussian_blur
from fdg_pet_ct_mlx.activity import synthesize_activity_map
from fdg_pet_ct_mlx.lesion import apply_lesion_uptake


@dataclass(frozen=True)
class Tissue:
    suv_mean: float
    suv_std: float
    maisi_labels: tuple[int, ...]


class Organ:
    _tissues: ClassVar[dict[str, Tissue]] = {
        "unclassified_soft_tissue": Tissue(0.5, 0.0, ()),
        "liver": Tissue(2.0, 0.0, (1,)),
    }

    @property
    def tissues(self) -> Mapping[str, Tissue]:
        return self._tissues

    def lookup_table(self, size: int) -> tuple[float, ...]:
        table = [self.tissues["unclassified_soft_tissue"].suv_mean] * size
        for tissue in self.tissues.values():
            for label in tissue.maisi_labels:
                if label < size:
                    table[label] = tissue.suv_mean
        return tuple(table)


def test_blur_matches_scipy_zero_padding() -> None:
    mx = cast(FdgRuntime, import_mlx())
    values = np.random.default_rng(21).normal(size=(7, 8, 9)).astype(np.float32)
    expected = gaussian_filter(values, sigma=1.0, radius=3, mode="constant", cval=0.0)
    result = gaussian_blur(
        mx.array(values), (2.354820045, 2.354820045, 2.354820045), (1.0, 1.0, 1.0), mx
    )
    np.testing.assert_allclose(np.asarray(result, dtype=np.float32), expected, atol=2e-7, rtol=2e-6)


def test_activity_retains_label_grid_and_zero_variance_means() -> None:
    mx = cast(FdgRuntime, import_mlx())
    labels = np.array([[[0, 1], [2, 1]]], dtype=np.int32)
    result = synthesize_activity_map(mx.array(labels), Organ(), seed=9, mx=mx)
    np.testing.assert_array_equal(
        np.asarray(result, dtype=np.float32), np.array([[[0.5, 2.0], [0.5, 2.0]]], dtype=np.float32)
    )


@dataclass(frozen=True)
class Scanner:
    fwhm_mm: tuple[float, float, float]
    counts_per_suv: float
    scatter_fraction: float = 0.0
    randoms_fraction: float = 0.0


class NecrosisOrgan(Organ):
    _tissues: ClassVar[dict[str, Tissue]] = {
        "unclassified_soft_tissue": Tissue(0.5, 0.0, ()),
        "liver": Tissue(2.3, 0.0, (1,)),
    }


@dataclass(frozen=True)
class Lesion:
    avid_maisi_labels: tuple[int, ...] = (26,)
    _host_tissue: ClassVar[dict[int, str]] = {26: "liver"}
    tumor_to_background: float = 4.0
    heterogeneity: float = 0.0
    necrosis_fraction: float = 0.4
    spillover_fwhm_mm: float = 0.0

    @property
    def host_tissue(self) -> Mapping[int, str]:
        return self._host_tissue


def test_scanner_profile_changes_resolution_and_noise() -> None:
    mx = cast(FdgRuntime, import_mlx())
    impulse = np.zeros((24, 24, 12), dtype=np.float32)
    impulse[12, 12, 6] = 8.0
    activity = mx.array(impulse)
    ct = mx.zeros((24, 24, 12), dtype=mx.float32)
    sharp = Scanner(
        fwhm_mm=(2.0, 2.0, 2.0),
        counts_per_suv=400.0,
        scatter_fraction=0.0,
        randoms_fraction=0.0,
    )
    wide = Scanner(
        fwhm_mm=(12.0, 12.0, 12.0),
        counts_per_suv=400.0,
        scatter_fraction=0.0,
        randoms_fraction=0.0,
    )
    low_count = Scanner(
        fwhm_mm=(2.0, 2.0, 2.0),
        counts_per_suv=20.0,
        scatter_fraction=0.0,
        randoms_fraction=0.0,
    )
    high_count = Scanner(
        fwhm_mm=(2.0, 2.0, 2.0),
        counts_per_suv=2000.0,
        scatter_fraction=0.0,
        randoms_fraction=0.0,
    )
    sharp_pet = apply_acquisition_model(
        activity, ct, scanner=sharp, count_level=1.0, spacing_mm=(2.0, 2.0, 2.0), seed=4, mx=mx
    )
    wide_pet = apply_acquisition_model(
        activity, ct, scanner=wide, count_level=1.0, spacing_mm=(2.0, 2.0, 2.0), seed=4, mx=mx
    )
    uniform = mx.full((24, 24, 12), 2.0, dtype=mx.float32)
    noisy_pet = apply_acquisition_model(
        uniform, ct, scanner=low_count, count_level=1.0, spacing_mm=(2.0, 2.0, 2.0), seed=4, mx=mx
    )
    quiet_pet = apply_acquisition_model(
        uniform, ct, scanner=high_count, count_level=1.0, spacing_mm=(2.0, 2.0, 2.0), seed=4, mx=mx
    )
    mx.eval(sharp_pet, wide_pet, noisy_pet, quiet_pet)
    sharp_host = np.asarray(sharp_pet, dtype=np.float32)
    wide_host = np.asarray(wide_pet, dtype=np.float32)
    noisy_host = np.asarray(noisy_pet, dtype=np.float32)
    quiet_host = np.asarray(quiet_pet, dtype=np.float32)
    assert sharp_host.shape == wide_host.shape == (24, 24, 12)
    assert float(sharp_host[12, 12, 6]) > float(wide_host[12, 12, 6])
    assert float(np.std(noisy_host)) > float(np.std(quiet_host))


def test_necrosis_is_photopenic_core_not_rim() -> None:
    mx = cast(FdgRuntime, import_mlx())
    organ = NecrosisOrgan()
    lesion = Lesion(
        tumor_to_background=4.0,
        heterogeneity=0.0,
        necrosis_fraction=0.4,
        spillover_fwhm_mm=0.0,
    )
    labels = np.zeros((16, 16, 16), dtype=np.int32)
    labels[4:12, 4:12, 4:12] = 26
    activity = mx.full((16, 16, 16), 2.3, dtype=mx.float32)
    updated = apply_lesion_uptake(
        activity,
        mx.array(labels),
        organ=organ,
        lesion=lesion,
        spacing_mm=(2.0, 2.0, 2.0),
        seed=0,
        mx=mx,
    )
    mx.eval(updated)
    host = np.asarray(updated, dtype=np.float32)
    core = host[7:9, 7:9, 7:9]
    rim = host[4, 4:12, 4:12]
    assert float(np.mean(core)) < float(np.mean(rim))
    assert float(np.mean(core)) < 1.0
    assert float(np.mean(rim)) > 6.0


@pytest.mark.parametrize("label_value", [0, 26])
def test_lesion_rejects_broadcastable_mismatched_grids(label_value: int) -> None:
    mx = cast(FdgRuntime, import_mlx())
    with pytest.raises(ValueError, match="same 3D grid"):
        apply_lesion_uptake(
            mx.ones((2, 3, 4), dtype=mx.float32),
            mx.full((1, 3, 4), label_value, dtype=mx.int32),
            organ=Organ(),
            lesion=Lesion(necrosis_fraction=0.0),
            spacing_mm=(1.0, 1.0, 1.0),
            seed=0,
            mx=mx,
        )


@pytest.mark.parametrize("value", [26.9, -1.0, float("nan"), float("inf"), 2**31])
def test_lesion_rejects_invalid_label_ids_before_cast(value: float) -> None:
    """A fractional or invalid ID must not become an avid or background label."""
    mx = cast(FdgRuntime, import_mlx())
    with pytest.raises(ValueError, match="lesion labels"):
        apply_lesion_uptake(
            mx.ones((1, 1, 1), dtype=mx.float32),
            mx.full((1, 1, 1), value, dtype=mx.float32),
            organ=Organ(),
            lesion=Lesion(necrosis_fraction=0.0),
            spacing_mm=(1.0, 1.0, 1.0),
            seed=0,
            mx=mx,
        )


@pytest.mark.parametrize("label", [0, 26])
@pytest.mark.parametrize(
    "lesion",
    [
        Lesion(tumor_to_background=-1.0, necrosis_fraction=0.0),
        Lesion(tumor_to_background=float("inf"), necrosis_fraction=0.0),
        Lesion(heterogeneity=-0.1, necrosis_fraction=0.0),
        Lesion(heterogeneity=float("nan"), necrosis_fraction=0.0),
        Lesion(necrosis_fraction=-0.1),
        Lesion(necrosis_fraction=1.1),
        Lesion(necrosis_fraction=float("nan")),
        Lesion(spillover_fwhm_mm=-0.1, necrosis_fraction=0.0),
        Lesion(spillover_fwhm_mm=float("nan"), necrosis_fraction=0.0),
    ],
)
def test_lesion_rejects_invalid_profile_values(label: int, lesion: Lesion) -> None:
    """Invalid fractions, uptake and blur cannot silently disable or alter a lesion."""
    mx = cast(FdgRuntime, import_mlx())
    with pytest.raises(ValueError, match="lesion"):
        apply_lesion_uptake(
            mx.ones((1, 1, 1), dtype=mx.float32),
            mx.full((1, 1, 1), label, dtype=mx.int32),
            organ=Organ(),
            lesion=lesion,
            spacing_mm=(1.0, 1.0, 1.0),
            seed=0,
            mx=mx,
        )


@pytest.mark.parametrize("label", [0, 26])
def test_lesion_requires_host_tissue_for_each_avid_label(label: int) -> None:
    """An unspecified host must not turn an avid lesion into zero activity."""
    class MissingHost(Lesion):
        @property
        def host_tissue(self) -> Mapping[int, str]:
            return {}

    mx = cast(FdgRuntime, import_mlx())
    with pytest.raises(ValueError, match=r"host_tissue.*26"):
        apply_lesion_uptake(
            mx.ones((1, 1, 1), dtype=mx.float32),
            mx.full((1, 1, 1), label, dtype=mx.int32),
            organ=Organ(),
            lesion=MissingHost(necrosis_fraction=0.0),
            spacing_mm=(1.0, 1.0, 1.0),
            seed=0,
            mx=mx,
        )


def test_acquisition_rejects_non_3d_grids() -> None:
    mx = cast(FdgRuntime, import_mlx())
    with pytest.raises(ValueError, match="same 3D grid"):
        apply_acquisition_model(
            mx.ones((2, 3), dtype=mx.float32), mx.zeros((2, 3), dtype=mx.float32),
            scanner=Scanner(fwhm_mm=(0.0, 0.0, 0.0), counts_per_suv=100.0),
            count_level=1.0, spacing_mm=(1.0, 1.0, 1.0), seed=0, mx=mx,
        )


@pytest.mark.parametrize(
    ("counts_per_suv", "count_level"),
    [(0.0, 1.0), (-1.0, 1.0), (float("inf"), 1.0), (1e-40, 1.0), (1e30, 1e30),
     (100.0, 0.0), (100.0, -1.0), (100.0, float("nan"))],
)
def test_acquisition_rejects_invalid_count_scales(
    counts_per_suv: float, count_level: float
) -> None:
    """Invalid or float32-underflowing count scales must not produce NaN SUV maps."""
    mx = cast(FdgRuntime, import_mlx())
    with pytest.raises(ValueError, match="count"):
        apply_acquisition_model(
            mx.ones((2, 2, 2), dtype=mx.float32), mx.zeros((2, 2, 2), dtype=mx.float32),
            scanner=Scanner(fwhm_mm=(0.0, 0.0, 0.0), counts_per_suv=counts_per_suv),
            count_level=count_level, spacing_mm=(1.0, 1.0, 1.0), seed=0, mx=mx,
        )


@pytest.mark.parametrize(
    ("scatter", "randoms"), [(-0.1, 0.0), (0.0, 1.1), (float("inf"), 0.0)]
)
def test_acquisition_rejects_invalid_scanner_fractions(scatter: float, randoms: float) -> None:
    """Scatter and randoms fractions must be finite values between zero and one."""
    mx = cast(FdgRuntime, import_mlx())
    with pytest.raises(ValueError, match="fraction"):
        apply_acquisition_model(
            mx.ones((2, 2, 2), dtype=mx.float32), mx.zeros((2, 2, 2), dtype=mx.float32),
            scanner=Scanner(
                fwhm_mm=(0.0, 0.0, 0.0), counts_per_suv=100.0,
                scatter_fraction=scatter, randoms_fraction=randoms,
            ),
            count_level=1.0, spacing_mm=(1.0, 1.0, 1.0), seed=0, mx=mx,
        )


@pytest.mark.parametrize(
    ("activity_value", "ct_value"),
    [(1.0, float("inf")), (1.0, -float("inf")), (1.0, float("nan")),
     (-1.0, 0.0), (float("nan"), 0.0)],
)
def test_acquisition_rejects_invalid_voxels(activity_value: float, ct_value: float) -> None:
    """Invalid activity or CT cannot become plausible finite output through clipping."""
    mx = cast(FdgRuntime, import_mlx())
    with pytest.raises(ValueError, match=r"activity|CT"):
        apply_acquisition_model(
            mx.full((2, 2, 2), activity_value, dtype=mx.float32),
            mx.full((2, 2, 2), ct_value, dtype=mx.float32),
            scanner=Scanner(fwhm_mm=(0.0, 0.0, 0.0), counts_per_suv=100.0),
            count_level=1.0, spacing_mm=(1.0, 1.0, 1.0), seed=0, mx=mx,
        )


def test_acquisition_rejects_float32_overflow() -> None:
    """Overflow in otherwise finite inputs cannot be returned as an SUV image."""
    mx = cast(FdgRuntime, import_mlx())
    with pytest.raises(ValueError, match="non-finite SUV"):
        apply_acquisition_model(
            mx.full((2, 2, 2), 2e38, dtype=mx.float32),
            mx.full((2, 2, 2), -1000.0, dtype=mx.float32),
            scanner=Scanner(
                fwhm_mm=(0.0, 0.0, 0.0), counts_per_suv=100.0,
                scatter_fraction=1.0, randoms_fraction=1.0,
            ),
            count_level=1.0, spacing_mm=(1.0, 1.0, 1.0), seed=0, mx=mx,
        )


@pytest.mark.parametrize(
    ("fwhm", "spacing"),
    [((-1.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
     ((float("nan"), 0.0, 0.0), (1.0, 1.0, 1.0)),
     ((0.0, 0.0, 0.0), (0.0, 1.0, 1.0)),
     ((0.0, 0.0, 0.0), (1.0, float("inf"), 1.0)),
     ((1e308, 0.0, 0.0), (1e-308, 1.0, 1.0))],
)
def test_blur_rejects_invalid_physical_parameters(
    fwhm: tuple[float, float, float], spacing: tuple[float, float, float]
) -> None:
    """Blur width and voxel spacing must describe a valid physical grid."""
    mx = cast(FdgRuntime, import_mlx())
    with pytest.raises(ValueError, match=r"FWHM|spacing"):
        gaussian_blur(mx.ones((2, 2, 2), dtype=mx.float32), fwhm, spacing, mx)


@pytest.mark.parametrize("values", [
    [-1, 1], [0.5, 1.9], [float("nan"), 1], [float("inf"), 1],
    [float("-inf"), 1], [2**31, 1],
])
def test_activity_rejects_invalid_labels_before_cast(values: list[float]) -> None:
    mx = cast(FdgRuntime, import_mlx())
    with pytest.raises(ValueError, match="activity labels"):
        synthesize_activity_map(mx.reshape(mx.array(values), (1, 1, 2)), Organ(), seed=9, mx=mx)


def test_activity_rejects_empty_labels() -> None:
    mx = cast(FdgRuntime, import_mlx())
    with pytest.raises(ValueError, match="nonempty"):
        synthesize_activity_map(mx.zeros((0, 1, 1), dtype=mx.int32), Organ(), seed=9, mx=mx)


@pytest.mark.parametrize("dtype", ["int32", "uint32", "int64"])
def test_activity_preserves_large_unknown_integral_labels(dtype: str) -> None:
    mx = cast(FdgRuntime, import_mlx())
    label_dtype = {"int32": mx.int32, "uint32": mx.uint32, "int64": mx.int64}[dtype]
    labels = mx.reshape(mx.array([0, 2**31 - 1], dtype=label_dtype), (1, 1, 2))
    result = synthesize_activity_map(labels, Organ(), seed=9, mx=mx)
    np.testing.assert_array_equal(np.asarray(result, dtype=np.float32), np.full((1, 1, 2), 0.5))
