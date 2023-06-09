# Classification of chemically modified red blood cells in microflow using machine learning video analysis

R. K. Rajaram Baskaran, A. Link, B. Porr and T. Franke

[![DOI](https://zenodo.org/badge/570490201.svg)](https://zenodo.org/badge/latestdoi/570490201)

# Prerequisites

 - Python 3.10.6
 - Tensorflow 2.11.0 
 - Keras
 - OpenCV
 - NumPy
 - Matplotlib
 - tqdm

# Usage

 1. run `main.py` to train, create, validate and test the model.
 2. run `plots.py` to generate the plots as seen in the paper.

## `main.py <option_name>`
Train, validate and test (native vs chem. mod.) RBCs.

Options:
 - FA: Classification of native vs formaldehyde
 - DA: Classification of native vs diamide
 - GA: Classification of native vs glutaraldehyde
 - MIX: Classification of native vs random mix of formaldehyde, diamide, glutaraldehyde

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
