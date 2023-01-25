#!/usr/bin/python3

import framegenerator
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#devices = tf.config.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(devices[0],True)

paths_to_healthy_learn = [ "/data/RBC-ZigZag/ALL/60xPhotron_20mBar_C001H001S0008.avi",
                           "/data/RBC-ZigZag/ALL/60xPhotron_20mBar_C001H001S0009.avi",
                           "/data/RBC-ZigZag/ALL/60xPhotron_20mBar_C001H001S0010.avi" ]

paths_to_ill_learn = [ "/data/RBC-ZigZag/ALL/60xPhotron_20mBar_2___FA3_7percent_C001H001S0001.avi",
                       "/data/RBC-ZigZag/ALL/60xPhotron_20mBar_2___FA3_7percent_C001H001S0002.avi",
                       "/data/RBC-ZigZag/ALL/60xPhotron_20mBar_2___FA3_7percent_C001H001S0004.avi"
]


path_to_healthy_val = "/data/RBC-ZigZag/ALL/60xPhotron_20mBar_C001H001S0001.avi"
path_to_ill_val = "/data/RBC-ZigZag/ALL/60xPhotron_20mBar_2___FA3_7percent_C001H001S0003.avi"


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
