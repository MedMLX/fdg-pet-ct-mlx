# Third-party notices

The image-space FDG synthesis implementation is MedMLX-authored code licensed
under [Apache-2.0](LICENSE). It does not distribute a pretrained model.

## Dependencies

MLX and NumPy provide array operations; the runtime and development
dependencies are declared in [pyproject.toml](pyproject.toml). These packages are
installed separately and retain their respective licenses. SciPy is used as an
independent numerical reference in the tests.

Input medical images and label maps are supplied by the caller. Their original
dataset or provider terms apply independently of this package's source license.

## Showcase image

The abdominal CT view is rendered from the public defaced images in: Gatidis,
S. and Kuestner, T. (2022), *A whole-body FDG-PET/CT dataset with manually
annotated tumor lesions (FDG-PET-CT-Lesions)*, Version 2, The Cancer Imaging
Archive, [doi:10.7937/gkr0-xv29](https://doi.org/10.7937/gkr0-xv29). The
defaced images are licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The PET overlay and uptake focus are simulated with this package and do not represent measured PET.
