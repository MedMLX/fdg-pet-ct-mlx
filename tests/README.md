# Runner test input

The runner tests use the converted HCC-TACE-Seg HCC_063 contrast CT and liver
mask at `~/.cache/medmlx/test-data/hcc_063_ct.nii.gz` and
`~/.cache/medmlx/test-data/hcc_063_liver.nii.gz` by default. `MEDMLX_TEST_CT`
and `MEDMLX_TEST_MASK` override those paths. Tests use matching crops of both
and assign controlled geometry for the runner assertions.
