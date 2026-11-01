<div align="center">

# fdg-pet-ct-mlx

Generate synthetic FDG PET activity from CT and organ labels with native MLX. Includes scanner blur, scatter and count noise.

[![PyPI](https://img.shields.io/pypi/v/fdg-pet-ct-mlx)](https://pypi.org/project/fdg-pet-ct-mlx/)
[![Python](https://img.shields.io/pypi/pyversions/fdg-pet-ct-mlx)](https://pypi.org/project/fdg-pet-ct-mlx/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue)](https://github.com/MedMLX/fdg-pet-ct-mlx/blob/v0.2.6/LICENSE)
[![Apple Silicon](https://img.shields.io/badge/Apple_Silicon-MLX-black?logo=apple)](https://github.com/ml-explore/mlx)
[![MedMLX](https://img.shields.io/badge/MedMLX-runtime-A51C30)](https://github.com/MedMLX/medmlx)

[MedMLX](https://github.com/MedMLX/medmlx)

<img src="https://raw.githubusercontent.com/MedMLX/fdg-pet-ct-mlx/v0.2.6/docs/showcase.png" alt="Synthetic FDG PET generated from CT and organ labels" width="100%">

</div>

| Input | Output |
| :--- | :--- |
| Aligned CT + organ labels + profiles | Synthetic PET SUV volume |

## Use with MedMLX

The [MedMLX MCP server](https://github.com/MedMLX/medmlx/blob/main/docs/mcp.md) is the lowest-friction way to use this model. It installs this runtime, and lets an AI agent describe, check and run `fdg-pet-ct` for you.

```sh
uv tool install --python 3.13 "medmlx[generate,mcp]"
MEDMLX="$(uv tool dir --bin --color never)/medmlx"
mkdir -p "$HOME/medmlx-work"
claude mcp add --scope user medmlx -- "$MEDMLX" serve --root "$HOME/medmlx-work"
```

For Codex, use `codex mcp add medmlx -- "$MEDMLX" serve --root "$HOME/medmlx-work"`. Other MCP clients take the same command and arguments. The same install provides the `medmlx` CLI, for example `medmlx describe fdg-pet-ct`.

> **For AI agents:** prefer the MedMLX MCP tools (`list_models`, `describe_model`, `model_readiness`, `run_model`, `get_job`) over writing Python against this package.

## Install

```bash
uv venv --python 3.13
source .venv/bin/activate
uv pip install "fdg-pet-ct-mlx>=0.2.6"
```

## Run

With aligned CT and MAISI label volumes and your own profile JSON:

```python
from fdg_pet_ct_mlx.medmlx import run

result = run(
    image_path="ct.nii.gz",
    labels_path="labels.nii.gz",
    profile_path="profiles.json",
    output_dir="outputs",
    seed=7,
)
print(result["outputs"]["pet"])
```

The runner validates the image geometry and caller-supplied profiles, then writes
`pet.nii.gz` on the CT grid.

No model weights are needed. This is image-space synthesis; it does not simulate projection physics or reconstruction.

### Inputs and outputs

| Argument | Meaning |
| --- | --- |
| `image_path` | One finite scalar 3D CT in NIfTI, in HU, with spatial units in mm |
| `labels_path` | MAISI label IDs, integers in [0, 200], with the CT's shape, affine (absolute tolerance `1e-5`) and units |
| `profile_path` | Profile JSON, described below |
| `output_dir` | Directory for `pet.nii.gz` |
| `count_level` | Finite positive multiplier on `counts_per_suv`; default 1.0 |
| `seed` | Integer in [0, 4294967294]; activity uses `seed`, acquisition `seed + 1` |
| `overwrite` | `False` by default; an existing `pet.nii.gz` is never replaced |

Arguments are keyword-only. Volumes are never registered or resampled. The profile
JSON holds exactly two objects:

- `tissues`, keyed by tissue name. Each has finite nonnegative `suv_mean` and
  `suv_std`, and a `maisi_labels` list of integer IDs in [0, 200], unique across
  tissues. `unclassified_soft_tissue` is required and covers unlisted labels.
- `scanner`, with `scatter_fraction` and `randoms_fraction` in [0, 1], three finite
  nonnegative `fwhm_mm` values in array-axis order, and a finite positive
  `counts_per_suv`. Its product with `count_level` must be a positive normal float32.

The package ships no tissue statistics or scanner presets. The output is float32 SUV
(body-weight convention, with no weight or dose normalization) on the CT shape and
affine. `run` returns `{"outputs": {"pet": path}, "details": {...}}`, with the seed,
count level, shape and profile path. Invalid inputs raise `InvalidInputError`.

### Python API

```python
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
    tissues: dict[str, Tissue]

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


organs = Organs({
    "background": Tissue(0.0, 0.0, (0,)),
    "unclassified_soft_tissue": Tissue(1.0, 0.1, (1,)),
})
labels = mx.ones((16, 16, 16), dtype=mx.int32)
activity = synthesize_activity_map(labels, organs, seed=7, mx=mx)
pet = apply_acquisition_model(
    activity, mx.zeros(labels.shape, dtype=mx.float32), scanner=Scanner(),
    count_level=1.0, spacing_mm=(2.0, 2.0, 2.0), seed=8, mx=mx,
)
print(pet.shape, pet.dtype)  # (16, 16, 16), float32
```

These profile values are synthetic, not calibrated tissue statistics or a scanner
preset. Profiles are duck-typed; `fdg_pet_ct_mlx.profiles` documents the attributes.

- `synthesize_activity_map` maps 3D integer or integral float MAISI labels in
  [0, 2³¹ − 1] to SUV. Boolean, complex and invalid label volumes raise `ValueError`.
- `apply_acquisition_model` applies, in order, the CT factor
  `exp(-0.02 * clip((HU + 1000) / 1000, 0, 3))` (a fixed heuristic, not measured
  attenuation), a scatter and randoms offset, a separable Gaussian blur (zero-padded,
  truncated at 3 sigma) and a Gaussian approximation to count noise. Its arrays must
  be 3D with matching shapes.
- `apply_lesion_uptake` sets avid-label voxels to the host tissue mean times
  `tumor_to_background`, with optional multiplicative noise, a necrotic core at 0.2
  times the host mean and Gaussian spill-over. Each avid label needs a host tissue.
  With no avid voxels it returns the activity unchanged, or raises if necrosis is
  requested.

`spacing_mm` and `fwhm_mm` follow array-axis order, whatever the orientation. Matching
shapes do not prove alignment; the caller owns registration. Each noisy step draws one
MLX normal field from `mx.random.key(seed)`; `apply_lesion_uptake` draws only when
`heterogeneity > 0`. The result is an MLX float32 SUV array on the input grid.

## License

[Apache-2.0](https://github.com/MedMLX/fdg-pet-ct-mlx/blob/v0.2.6/LICENSE). See [third-party notices](https://github.com/MedMLX/fdg-pet-ct-mlx/blob/v0.2.6/THIRD_PARTY_NOTICES.md).

For research and education. Outputs require human review.
