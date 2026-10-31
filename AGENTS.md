# fdg-pet-ct-mlx

Native MLX kernels for weight-free synthetic FDG PET activity from CT and
organ labels, with scanner blur, scatter and count noise.

## Rules

- Allow `Any` at untyped third-party library boundaries; keep package interfaces typed.

- Model execution runs in MLX on Metal (macOS arm64). There is no PyTorch,
  MPS, NumPy or CPU fallback; host I/O and preprocessing may use CPU
  libraries.
- Unsupported checkpoints, options and inputs raise a clear error. Do not add
  compatibility shims or silent fallbacks.
- Never commit model weights, medical images, patient data or generated
  outputs.
- Keep upstream notices in THIRD_PARTY_NOTICES.md; model weights retain their
  publisher terms.
- Profiles are explicit caller inputs; do not invent tissue statistics or
  scanner presets.
- This is image-space synthesis, not projection physics or reconstruction.
  Keep application registries and run records outside the package.

## Checks

- `make env` creates the locked development environment.
- `make qa` runs Ruff and Pyright; it does not run models.
- `make test TESTS=tests/test_name.py` runs selected tests; numerical changes
  need the affected comparison against the upstream reference.
- There is no hosted CI; local checks are authoritative.
