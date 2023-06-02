# Classification of chemically modified red blood cells in microflow using machine learning video analysis

R. K. Rajaram Baskaran, A. Link, B. Porr and T. Franke

# Usage

 1. run `main.py` to train, create, validate and test the model.
 2. run `plots.py` to generate the plots as seen in the paper.

## `main.py --option=<option_name>`
Train, validate and test (native vs chem. mod.) RBCs.

Options:
 - FA: Classification of native vs formaldehyde
 - DA: Classification of native vs diamide
 - GA: Classification of native vs ????
 - MIX: Classification of native vs a max of formaldehyde, diamide, ???
 - TEST: ???????? 

This generates `accuracy_and_loss_values.json`
which is then imported into `plots.py`.


## `plots.py`
Loads `accuracy_and_loss_values.json` and
plots accuracy, loss and probability predictions.

# Modules


## `video_processor.py`
Labels the videos, subtracts the background, and 
returns them as NumPy arrays.


# Tests

## `test_get_videos.py`
Tests loading videos from the file directory.

## `test_bg_sub.py`
Performs background subtraction, displays processed video.
