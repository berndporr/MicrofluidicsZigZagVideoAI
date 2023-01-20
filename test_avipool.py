#!/usr/bin/python3

import framegenerator
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

#devices = tf.config.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(devices[0],True)

paths_to_healthy = [ "/data/RBC-ZigZag/ALL/60xPhotron_20mBar_C001H001S0008.avi",
                     "/data/RBC-ZigZag/ALL/60xPhotron_20mBar_C001H001S0009.avi",
                     "/data/RBC-ZigZag/ALL/60xPhotron_20mBar_C001H001S0010.avi" ]

background_subtraction_method = "rect"

crop = [[100,0],[500,120]]
clip_len = 50
fret = False
do_background_subtraction = "rect"

avipool = framegenerator.AVIpool(
    paths_to_healthy,
    "Healthy", 
    crop_rect = crop, 
    clip_length = clip_len, 
    subtract_background = background_subtraction_method)

f1 = avipool.get_frames_of_clip(10)
f2 = avipool.get_frames_of_clip(300)
f3 = avipool.get_frames_of_clip(800)
