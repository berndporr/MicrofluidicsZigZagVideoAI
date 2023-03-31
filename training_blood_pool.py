#!/usr/bin/python3

import glob
import os
import framegenerator
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#devices = tf.config.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(devices[0],True)

##LEARN
paths_to_healthy_learn = ["/data/RBC_Phantom_60xOlympus/Donor_1/Native5_focused",
                           "/data/RBC_Phantom_60xOlympus/Donor_1/Native5_overfocused2ticks",
                           "/data/RBC_Phantom_60xOlympus/Donor_1/Native5_underfocused2ticks",
                           "/data/RBC_Phantom_60xOlympus/Donor_2/RBC_9March2023_Donor2_3_focused",
                           "/data/RBC_Phantom_60xOlympus/Donor_2/RBC_9March2023_Donor2_2_underfocused",
                           "/data/RBC_Phantom_60xOlympus/Donor_2/RBC_9March2023_Donor2_4_overfocused"]

paths_to_ill_learn = ["/data/RBC_Phantom_60xOlympus/Donor_1/FA_0.37wtPercent",
                       "/data/RBC_Phantom_60xOlympus/Donor_2/RBC10March2023_Donor2_2ndDay_1mMDiamide_Split_focused",
                       "/data/RBC_Phantom_60xOlympus/Donor_2/RBC10March2023_Donor2_2ndDay_1mMDiamide_Split_Overfocused",
                       "/data/RBC_Phantom_60xOlympus/Donor_2/RBC10March2023_Donor2_2ndDay_1mMDiamide_Split_Underfocused"]

# Create empty lists to store the file paths
healthy_learn_files = []
ill_learn_files = []

# Loop over the directories and find all .avi files
for path in paths_to_healthy_learn:
    for file in os.listdir(path):
        if file.endswith('.avi'):
            healthy_learn_files.append(os.path.join(path, file))

for path in paths_to_ill_learn:
    for file in os.listdir(path):
        if file.endswith('.avi'):
            ill_learn_files.append(os.path.join(path, file))

print("Healthy Train Files:", healthy_learn_files)
print("Ill Train Files:", ill_learn_files)

##VALIDATION
path_to_healthy_val = "/data/RBC_Phantom_60xOlympus/Donor_1/Native5_focused"
path_to_ill_val = "/data/RBC_Phantom_60xOlympus/Donor_1/FA_0.37wtPercent"

# Read all .avi files in path_to_healthy_val directory
healthy_val_files = []
for file in glob.glob(os.path.join(path_to_healthy_val, "*.avi")):
    healthy_val_files.append(file)

# Read all .avi files in path_to_ill_val directory
ill_val_files = []
for file in glob.glob(os.path.join(path_to_ill_val, "*.avi")):
    ill_val_files.append(file)

print("Healthy Val Files:", healthy_val_files)
print("Ill Val Files:", ill_val_files)

background_subtraction_method = "opencv,rect"

crop = [[100,0],[500,120]]
clip_len = 60

avi_healthy_learn = framegenerator.AVIpool(
  paths_to_healthy_learn,
  "Healthy",
  crop_rect = crop,
  clip_length = clip_len,
  subtract_background = background_subtraction_method)

avi_ill_learn  = framegenerator.AVIpool(
  paths_to_ill_learn,
  "Ill",
  crop_rect = crop,
  clip_length = clip_len,
  subtract_background = background_subtraction_method)

avi_files_learn = [avi_healthy_learn,avi_ill_learn]

# Create the training set
output_signature = (tf.TensorSpec(shape = (None, None, None, 3), dtype = tf.float32),
                    tf.TensorSpec(shape = (), dtype = tf.int16))

maxClips = 100

fg_train = framegenerator.FrameGenerator(avi_files_learn, training=True, max_clips=maxClips)
train_ds = tf.data.Dataset.from_generator(fg_train,
                                          output_signature = output_signature)



avi_healthy_val = framegenerator.AVIfile(
  path_to_healthy_val,
  "Healthy",
  crop_rect = crop,
  clip_length = clip_len,
  subtract_background = background_subtraction_method)

avi_ill_val  = framegenerator.AVIfile(
  path_to_ill_val,
  "Ill",
  crop_rect = crop,
  clip_length = clip_len,
  subtract_background = background_subtraction_method)

avi_files_val = [avi_healthy_val,avi_ill_val]

fg_val = framegenerator.FrameGenerator(avi_files_val, training=True)
val_ds = tf.data.Dataset.from_generator(fg_val,
                                        output_signature = output_signature)

# Print the shapes of the data
train_frames, train_labels = next(iter(train_ds))
print(f'Shape of training set of frames: {train_frames.shape}')
print(f'Shape of training labels: {train_labels.shape}')

val_frames, val_labels = next(iter(val_ds))
print(f'Shape of validation set of frames: {val_frames.shape}')
print(f'Shape of validation labels: {val_labels.shape}')

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size = AUTOTUNE)
val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size = AUTOTUNE)

train_ds = train_ds.batch(2)
val_ds = val_ds.batch(2)

train_frames, train_labels = next(iter(train_ds))
print(f'Shape of training set of frames: {train_frames.shape}')
print(f'Shape of training labels: {train_labels.shape}')

val_frames, val_labels = next(iter(val_ds))
print(f'Shape of validation set of frames: {val_frames.shape}')
print(f'Shape of validation labels: {val_labels.shape}')





########
## Random Keras Model

net = tf.keras.applications.EfficientNetB0(include_top = False)
net.trainable = False

model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(scale=255),
    tf.keras.layers.TimeDistributed(net),
    tf.keras.layers.Dense(10),
    tf.keras.layers.GlobalAveragePooling3D()
])

model.compile(optimizer = 'adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
              metrics=['accuracy'])

model.fit(train_ds, 
          epochs = 10,
          validation_data = val_ds,
          callbacks = tf.keras.callbacks.EarlyStopping(patience = 2, monitor = 'val_loss'))

model.summary()

# model.save('models/blood')

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('model.tflite', 'wb') as f:
  f.write(tflite_model)
