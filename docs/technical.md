# fdg-pet-ct-mlx technical notes

See the [README](../README.md) for installation and the registered file runner.

## Computational scope

This is a tensor library for image-space FDG PET synthesis. Callers supply organ,
scanner and optional lesion profiles, aligned image arrays and explicit random
seeds. The acquisition and lesion functions validate the settings described below.
The registered MedMLX adapter reads and validates profile JSON and NIfTI inputs.
Registration and resampling remain caller-owned. No pretrained weights are used.

[`synthesize_activity_map`](../src/fdg_pet_ct_mlx/activity.py) maps organ labels
to activity. [`apply_acquisition_model`](../src/fdg_pet_ct_mlx/acquisition.py)
applies a CT-dependent residual, scatter/randoms offset, separable Gaussian blur
and a Gaussian approximation to count noise. Spacing and FWHM follow array-axis
order. The result is an MLX float32 SUVbw array on the input grid.
This does not simulate projection physics or reconstruction.

## Inputs and conventions

- **Axis order.** Volumes are 3-D; `spacing_mm` and `fwhm_mm` are matched to array
  axes by position, whatever anatomical orientation the caller uses. Label, CT and
  activity volumes must share one grid. Registration and resampling are not done
  here. `apply_acquisition_model` and `apply_lesion_uptake` require their paired
  arrays to be 3-D with matching shapes. Shape equality does not establish physical
  alignment; the caller must ensure the grids correspond.
- **Labels.** Label IDs are MAISI label IDs. `synthesize_activity_map` and
  `apply_lesion_uptake` accept nonempty 3-D integer or floating-point volumes of
  finite, integral IDs in `[0, 2**31 - 1]`. Boolean, complex and invalid label
  volumes raise `ValueError` before casting. `synthesize_activity_map` uses the
  `unclassified_soft_tissue` statistics for IDs that no tissue lists.
- **Units.** Activity and outputs are SUV (body-weight convention; no weight or dose
  normalization is applied here). CT is in HU. Spacing and blur widths are in mm.
- **Profiles.** `fdg_pet_ct_mlx.profiles` defines structural protocols; the
  attributes and their meaning are documented there. The package ships no profile
  values. The acquisition function validates scanner fractions, blur widths,
  spacing and input voxels; its count-scale product must be a positive normal
  float32 value. Lesion uptake ratio, heterogeneity and spill-over width must be
  finite and nonnegative, the necrosis fraction must be in [0, 1], and each avid
  label must specify a host tissue. Organ profile values remain caller-owned.
  The file adapter additionally validates its JSON profiles and image geometry.
- **Order of operations in `apply_acquisition_model`.** CT-dependent factor,
  scatter/randoms offset, Gaussian blur (zero-padded at the borders, truncated at
  3 sigma), then Gaussian approximation to count noise. The CT factor
  `exp(-0.02 * clip((HU + 1000) / 1000, 0, 3))` is a fixed heuristic, not a measured
  attenuation model.
- **Lesions.** `apply_lesion_uptake` replaces activity in avid-label voxels with the
  host tissue mean times the tumor-to-background ratio, optionally with multiplicative
  noise, a necrotic core fixed at 0.2 times the host mean, and Gaussian spill-over.
  If there are no avid voxels, a positive requested necrosis fraction raises an
  error; otherwise the input activity array is returned unchanged, including its dtype.
- **Randomness.** The functions that add noise take an integer `seed` and draw one MLX
  normal field with `mx.random.key(seed)`. `apply_lesion_uptake` draws only when
  `heterogeneity > 0`.

## MedMLX file runner

`fdg_pet_ct_mlx.medmlx.run` accepts keyword-only `image_path`, `labels_path`,
`profile_path` and `output_dir`. The CT must be a finite scalar 3-D NIfTI volume
in HU with millimetre spatial units. Labels must be integer MAISI IDs in
`[0, 200]`, with the same shape, affine (absolute tolerance `1e-5`) and units.
The adapter validates these inputs and writes float32 `pet.nii.gz`, retaining
the CT geometry. It does not register or resample volumes.

The profile file is a JSON object with exactly two fields:

- `tissues`: an object keyed by tissue name. Each tissue has finite nonnegative
  `suv_mean` and `suv_std` values, plus a `maisi_labels` list of integer IDs in
  `[0, 200]`. IDs must be unique across tissues. `unclassified_soft_tissue` is
  required for labels without an explicit assignment.
- `scanner`: an object containing `scatter_fraction` and `randoms_fraction` in
  `[0, 1]`, three finite nonnegative `fwhm_mm` values in array-axis order, and a
  finite positive `counts_per_suv` value.

Profiles remain caller-owned; the adapter supplies no tissue statistics or
scanner presets. `count_level` must be finite and positive, and its product with
`counts_per_suv` must be a positive normal float32 value. `seed` selects the
activity draw, with `seed + 1` used for acquisition. Existing outputs require
`overwrite=True`.

## Low-level array example

A synthetic example with organ and scanner profiles:

```python
from collections.abc import Mapping
from dataclasses import dataclass

import mlx.core as mx

from fdg_pet_ct_mlx.acquisition import apply_acquisition_model
from fdg_pet_ct_mlx.activity import synthesize_activity_map


@dataclass(frozen=True)
class Tissue:
    suv_mean: float
    suv_std: float
    maisi_labels: tuple[int, ...]


@dataclass(frozen=True)
class Organs:
    tissues: Mapping[str, Tissue]

    def lookup_table(self, size: int) -> tuple[float, ...]:
        table = [self.tissues["unclassified_soft_tissue"].suv_mean] * size
        for tissue in self.tissues.values():
            for label in tissue.maisi_labels:
                if label < size:
                    table[label] = tissue.suv_mean
        return tuple(table)


@dataclass(frozen=True)
class Scanner:
    scatter_fraction: float = 0.05
    randoms_fraction: float = 0.02
    fwhm_mm: tuple[float, float, float] = (5.0, 5.0, 5.0)
    counts_per_suv: float = 100.0


organ = Organs({
    "background": Tissue(0.0, 0.0, (0,)),
    "unclassified_soft_tissue": Tissue(1.0, 0.1, (1,)),
})
labels = mx.ones((16, 16, 16), dtype=mx.int32)
ct_hu = mx.zeros(labels.shape, dtype=mx.float32)
activity = synthesize_activity_map(labels, organ, seed=7, mx=mx)
pet = apply_acquisition_model(
    activity,
    ct_hu,
    scanner=Scanner(),
    count_level=1.0,
    spacing_mm=(2.0, 2.0, 2.0),
    seed=8,
    mx=mx,
)
mx.eval(pet)
print(pet.shape, pet.dtype)  # (16, 16, 16), float32
```

The profile values above are synthetic demonstration inputs, not calibrated
tissue statistics or a scanner preset.

## Development checks

From a source checkout:

```bash
make env
make qa
make test
```

`make env` checks the lock before synchronizing the development environment.
`make qa` runs Ruff and strict Pyright through the existing project Python;
it does not run tests or install dependencies. Pyright uses the maintained
host Node runtime. Select affected tests with `TESTS` and `PYTEST_ARGS`.

Synthetic checks cover numerical and API behavior. They do not establish
physical scanner fidelity, clinical validity or diagnostic performance.
