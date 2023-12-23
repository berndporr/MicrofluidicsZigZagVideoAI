# Classification of chemically modified red blood cells in microflow using machine learning video analysis

Baskaran R K R, Link A, Porr B, Franke T (2024)

Soft Matter

[DOI of the code: 10.5281/zenodo.8126539](https://zenodo.org/badge/latestdoi/570490201)

The repository has been archived and is read-only.

# Prerequisites

 - Python 3.10
 - Tensorflow & Keras 2.13.0
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

This generates all results in the directory `results_<option>`.

# runall.sh

Runs all option: FA, DA, GA and MIX. 
 - Foreground: Shows the accuracy and loss.
 - Background: `nohup ./runall.sh &`. You can log out and it will continue.


# Modules

## `plots.py`
Loads `accuracy_and_loss_values.json` and
plots accuracy, loss and probability predictions.

## `video_processor.py`
Labels the videos, subtracts the background, and 
returns them as NumPy arrays.


# Tests

## `test_get_videos.py`
Tests loading videos from the file directory.

## `test_bg_sub.py`
Performs background subtraction, displays processed video.
