#!/usr/bin/python3

import framegenerator
import cv2
import tensorflow as tf


import matplotlib.pyplot as plt
import matplotlib.animation as animation

path_to_healthy = "/data/ZigZag-Channel-beads4_5um/60xPhotron_C001H001S0005.avi"
path_to_ill = "/data/ZigZag-Channel-beads4_5um/60xPhotron_C001H001S0004.avi"

crop = [[100,0],[600,138]]
avi_healthy = framegenerator.AVIfile(path_to_healthy,"Healthy", crop_rect = crop)
avi_ill  = framegenerator.AVIfile(path_to_ill,"Ill", crop_rect = crop)

avi_files = [avi_healthy,avi_ill]

healthy_first_clip = avi_healthy.get_frames_of_clip(0)
#print(healthy_first_clip)
#print(healthy_first_clip.shape)

train_clips_list = range(0,300)
val_clips_list = range(300,450)
test_clips_list = range(450,600)

# Create the training set
output_signature = (tf.TensorSpec(shape = (None, None, None, 3), dtype = tf.float32),
                    tf.TensorSpec(shape = (), dtype = tf.int16))

fg_train = framegenerator.FrameGenerator(avi_files, train_clips_list, training=True)
train_ds = tf.data.Dataset.from_generator(fg_train,
                                          output_signature = output_signature)

for frames, labels in train_ds.take(10):
  print(labels)

fg_val = framegenerator.FrameGenerator(avi_files, val_clips_list, training=True)
val_ds = tf.data.Dataset.from_generator(fg_val,
                                          output_signature = output_signature)

# Print the shapes of the data
train_frames, train_labels = next(iter(train_ds))
print(f'Shape of training set of frames: {train_frames.shape}')
print(f'Shape of training labels: {train_labels.shape}')

val_frames, val_labels = next(iter(val_ds))
print(f'Shape of validation set of frames: {val_frames.shape}')
print(f'Shape of validation labels: {val_labels.shape}')
