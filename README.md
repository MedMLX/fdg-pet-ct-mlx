# fdg-pet-ct-mlx

Generate synthetic FDG PET activity from CT and organ labels with native MLX. Includes scanner blur, scatter and count noise.

| Input | Output |
| :--- | :--- |
| Aligned CT + organ labels + profiles | Synthetic PET SUV volume |

**Apple Silicon · Metal · Python 3.13 / 3.14** · [uv](https://docs.astral.sh/uv/)

## Quick start

```bash
uv venv --python 3.13
source .venv/bin/activate
uv pip install "fdg-pet-ct-mlx>=0.2.4"
```

## MedMLX runner

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
`pet.nii.gz` on the CT grid. See the [profile format and array API](docs/technical.md)
for required fields and a synthetic tensor example.

No model weights are needed. This is image-space synthesis; it does not simulate projection physics or reconstruction.

## Documentation

[Profiles, array API and validation](docs/technical.md)

## License

Apache-2.0; license and third-party notices are included in the distribution.

For research and education. Outputs require human review.
