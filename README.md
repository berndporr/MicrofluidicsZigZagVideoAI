# Droplets Tensorflow classification

This repo does video classification

## framegenerator.py
Class which turns the raw AVI video files into
a random access stream for TF. It does also
cropping of the video and other on the fly tasks.
These are being added as I go along.


# test_framegenerator.py
Tests the framegenerator if it generates the right
dateformat for TF and displays the data.

# training_beads.py
Tensorflow classification bewtween the plastic
beads before we had the blood samples.

# viewer_blood.py
Viewer shows a couple of clips from the red blood cells.

# test_bgframegenerator.py
Tests background substraction (work in progress and not working at the moment)

# testopencv.py
Tests basic functionality of openCV and matplotlib if it's able
to load the video and display it.

# training_blood.py
Training of the healthy vs ill red blood cells.
