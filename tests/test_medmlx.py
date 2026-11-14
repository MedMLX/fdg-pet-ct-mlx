"""Standalone FDG adapter contracts; Metal execution is explicit and uses tiny arrays."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TypedDict, cast

import nibabel as nib
import numpy as np
import pytest
from medmlx_core.errors import InvalidInputError
from medmlx_core.outputs import validate_outputs

from fdg_pet_ct_mlx.medmlx import run


class FdgInputs(TypedDict):
    image_path: str
    labels_path: str
    profile_path: str
    output_dir: str


@pytest.fixture
def fdg_inputs(tmp_path: Path) -> FdgInputs:
    cache = Path.home() / ".cache" / "medmlx" / "test-data"
    ct_path = os.environ.get("MEDMLX_TEST_CT", str(cache / "hcc_063_ct.nii.gz"))
    mask_path = os.environ.get("MEDMLX_TEST_MASK", str(cache / "hcc_063_liver.nii.gz"))
    ct, mask = nib.load(ct_path), nib.load(mask_path)
    assert isinstance(ct, nib.Nifti1Image) and isinstance(mask, nib.Nifti1Image)
    assert ct.affine is not None and mask.affine is not None
    assert ct.shape == mask.shape and np.allclose(ct.affine, mask.affine)
    crop = np.s_[150:154, 250:255, 40:46]
    affine = np.diag([2.0, 3.0, 4.0, 1.0])
    affine[:3, 3] = [-10, 20, 5]
    for name, data in (("ct", np.asarray(ct.dataobj)[crop]),
                       ("labels", np.asarray(mask.dataobj)[crop])):
        image = nib.Nifti1Image(data.astype(np.float32), affine)
        image.header.set_xyzt_units("mm")
        # A CT display range in HU must not label the SUV output.
        image.header["cal_min"], image.header["cal_max"] = -1024.0, 3071.0
        nib.save(image, tmp_path / f"{name}.nii.gz")
    profile: dict[str, object] = {
        "tissues": {
            "unclassified_soft_tissue": {
                "suv_mean": 0.0,
                "suv_std": 0.0,
                "maisi_labels": [],
            }
        },
        "scanner": {
            "scatter_fraction": 0.0,
            "randoms_fraction": 0.0,
            "fwhm_mm": [0.0, 0.0, 0.0],
            "counts_per_suv": 100.0,
        },
    }
    (tmp_path / "profile.json").write_text(json.dumps(profile))
    return {
        "image_path": str(tmp_path / "ct.nii.gz"),
        "labels_path": str(tmp_path / "labels.nii.gz"),
        "profile_path": str(tmp_path / "profile.json"),
        "output_dir": str(tmp_path / "output"),
    }


def test_fdg_runner_preserves_geometry(fdg_inputs: FdgInputs) -> None:
    result = run(**fdg_inputs)
    destination = Path(fdg_inputs["output_dir"]) / "pet.nii.gz"
    assert result["outputs"] == {"pet": str(destination)}
    _, artifacts = validate_outputs(result)
    assert artifacts["pet"]["kind"] == "synthetic_image"
    assert artifacts["pet"].get("grid_of") == "image_path"
    assert artifacts["pet"].get("units") == "SUV"
    output = nib.load(destination)
    source = nib.load(fdg_inputs["image_path"])
    assert isinstance(output, nib.Nifti1Image)
    assert isinstance(source, nib.Nifti1Image)
    assert output.affine is not None and source.affine is not None
    assert output.shape == (4, 5, 6)
    assert output.get_data_dtype() == np.dtype(np.float32)
    np.testing.assert_array_equal(output.affine, source.affine)
    assert (float(output.header["cal_min"]), float(output.header["cal_max"])) == (0.0, 0.0)
    # A zero activity profile must remain zero through the real acquisition path.
    np.testing.assert_array_equal(output.get_fdata(), np.zeros((4, 5, 6)))


@pytest.mark.parametrize("problem", ["missing_ct", "count_level", "overwrite", "existing_output"])
def test_fdg_rejects_invalid_arguments(fdg_inputs: FdgInputs, problem: str) -> None:
    count_level, overwrite = 1.0, False
    if problem == "missing_ct":
        Path(fdg_inputs["image_path"]).unlink()
    elif problem == "count_level":
        count_level = 0.0
    elif problem == "overwrite":
        overwrite = cast(bool, "true")
    else:
        Path(fdg_inputs["output_dir"]).mkdir()
        (Path(fdg_inputs["output_dir"]) / "pet.nii.gz").write_bytes(b"existing output")
    with pytest.raises(InvalidInputError):
        run(**fdg_inputs, count_level=count_level, overwrite=overwrite)


@pytest.mark.parametrize("suffix", [".nii", ".nii.gz"])
@pytest.mark.parametrize("value", [1.0 + 2.0**-30, float("nan"), float("inf"), -1.0, 201.0])
def test_fdg_rejects_invalid_source_labels(
    fdg_inputs: FdgInputs, suffix: str, value: float
) -> None:
    """Validate source precision before compacting label IDs or entering Metal."""
    source = nib.load(fdg_inputs["labels_path"])
    assert isinstance(source, nib.Nifti1Image)
    values = np.ones(source.shape, dtype=np.float64)
    values[-1, -1, -1] = value
    labels = nib.Nifti1Image(values, source.affine, source.header)
    labels.header.set_data_dtype(np.float64)
    path = Path(fdg_inputs["labels_path"]).with_name("invalid" + suffix)
    nib.save(labels, path)
    fdg_inputs["labels_path"] = str(path)
    with pytest.raises(InvalidInputError, match="finite integer IDs"):
        run(**fdg_inputs)


def test_fdg_rejects_fractional_nifti_scaling(fdg_inputs: FdgInputs) -> None:
    """A sub-float32 fractional intercept cannot turn an integer source into valid IDs."""
    source = nib.load(fdg_inputs["labels_path"])
    assert isinstance(source, nib.Nifti1Image)
    labels = nib.Nifti1Image(np.ones(source.shape, dtype=np.int16), source.affine, source.header)
    labels.header.set_data_dtype(np.int16)
    labels.header["scl_slope"], labels.header["scl_inter"] = 1.0, 2.0**-30
    nib.save(labels, fdg_inputs["labels_path"])
    with pytest.raises(InvalidInputError, match="finite integer IDs"):
        run(**fdg_inputs)


@pytest.mark.parametrize("suffix", [".nii", ".nii.gz"])
def test_scaled_labels_preserve_seeded_pet(fdg_inputs: FdgInputs, suffix: str) -> None:
    """Accept exact scaled IDs and preserve their nonzero seeded tissue activity."""
    profile_path = Path(fdg_inputs["profile_path"])
    profile = cast(dict[str, object], json.loads(profile_path.read_text()))
    tissues = cast(dict[str, object], profile["tissues"])
    tissues["organ"] = {"suv_mean": 2.0, "suv_std": 0.3, "maisi_labels": [1]}
    profile_path.write_text(json.dumps(profile))
    original = run(**fdg_inputs, seed=17)
    expected = nib.load(cast(dict[str, str], original["outputs"])["pet"])
    assert isinstance(expected, nib.Nifti1Image)
    reference = expected.get_fdata(dtype=np.float32)
    assert (reference > 0).any()
    source = nib.load(fdg_inputs["labels_path"])
    assert isinstance(source, nib.Nifti1Image)
    values = np.asarray(source.dataobj, dtype=np.int16) * 2
    labels = nib.Nifti1Image(values, source.affine, source.header)
    labels.header.set_data_dtype(np.int16)
    labels.header["scl_slope"], labels.header["scl_inter"] = 0.5, 0.0
    path = Path(fdg_inputs["labels_path"]).with_name("scaled" + suffix)
    nib.save(labels, path)
    fdg_inputs["labels_path"] = str(path)
    fdg_inputs["output_dir"] += "-scaled"
    actual = run(**fdg_inputs, seed=17)
    image = nib.load(cast(dict[str, str], actual["outputs"])["pet"])
    assert isinstance(image, nib.Nifti1Image)
    np.testing.assert_array_equal(image.get_fdata(dtype=np.float32), reference)
