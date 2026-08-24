# multimodal-behavioral-biometrics

## Unarchive the HMOG dataset

1. Put the downloaded HMOG dataset at the `input_data/hmog_dataset.zip` path in this repository.
2. Under the installed conda environment, run `python unarchive_hmog.py`.
3. Manually delete the `input_data/hmog_dataset.zip` file if necessary.

## (Optional) Validate the unarchived HMOG dataset

Run `python validate_hmog_folders.py`.

Note: For an unknown reason, many `Reading1answer`, `Reading2answer`, `Reading3answer` 
and similar `Writing...answer` files are missing. But the rest of the files are fine.

