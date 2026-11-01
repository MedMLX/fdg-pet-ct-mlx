"""Standalone FDG adapter contracts; Metal execution is explicit and uses tiny arrays."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict, cast

import nibabel as nib
import numpy as np
import pytest
from medmlx_core.errors import InvalidInputError

from fdg_pet_ct_mlx.medmlx import run


class FdgInputs(TypedDict):
    image_path: str
    labels_path: str
    profile_path: str
    output_dir: str


@pytest.fixture
def fdg_inputs(tmp_path: Path) -> FdgInputs:
    affine = np.diag([2.0, 3.0, 4.0, 1.0])
    affine[:3, 3] = [-10, 20, 5]
    for name, data in (("ct", np.zeros((4, 5, 6))), ("labels", np.ones((4, 5, 6)))):
        image = nib.Nifti1Image(data.astype(np.float32), affine)
        image.header.set_xyzt_units("mm")
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
    output = nib.load(destination)
    source = nib.load(fdg_inputs["image_path"])
    assert isinstance(output, nib.Nifti1Image)
    assert isinstance(source, nib.Nifti1Image)
    assert output.affine is not None and source.affine is not None
    assert output.shape == (4, 5, 6)
    assert output.get_data_dtype() == np.dtype(np.float32)
    np.testing.assert_array_equal(output.affine, source.affine)
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
