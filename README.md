# multimodal-behavioral-biometrics

## Unarchive the HMOG dataset

1. Put the downloaded HMOG dataset at the `input_data/hmog_dataset.zip` path in this repository.
2. Under the installed conda environment, run `python unarchive_hmog.py`.
3. Manually delete the `input_data/hmog_dataset.zip` file if necessary.

## (Optional) Validate the unarchived HMOG dataset

Run `python validate_hmog_folders.py`.

Warning: The session folders within the user folder `207969` are named incorrectly, e.g. `207969\207696_session_1` (the
user ID is wrong in the session folder). To overcome this in a simple way, the code shouldn't rely on the user ID within
the session folder name because it is already known from the upper level folder name.

Note: For an unknown reason, many `Reading1answer`, `Reading2answer`, `Reading3answer`
and similar `Writing...answer` files are missing. But the rest of the files are fine.

