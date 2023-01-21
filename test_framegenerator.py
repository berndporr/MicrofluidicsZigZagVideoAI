#!/usr/bin/python3

import framegenerator
import cv2

import matplotlib.pyplot as plt
import matplotlib.animation as animation

path_to_healthy = "/data/RBC-ZigZag/Selection/60xPhotron_20mBar_2_C001H001S0001.avi"
healthy_clip_len = 50

path_to_ill = "/data/RBC-ZigZag/Selection/60xPhotron_20mBar_2___1percentGA_C001H001S0001.avi"
ill_clip_len = 60

crop = [[000,0],[500,120]]

avi_healthy = framegenerator.AVIfile(path_to_healthy,"Healthy", crop_rect = crop, clip_length = healthy_clip_len)
avi_ill  = framegenerator.AVIfile(path_to_ill,"Ill", crop_rect = crop, clip_length = ill_clip_len)

avi_files = [avi_healthy,avi_ill]

healthy_first_clip = avi_healthy.get_frames_of_clip(0)
#print(healthy_first_clip)
#print(healthy_first_clip.shape)

train_clips_list = range(0,300)

fg = framegenerator.FrameGenerator(avi_files, training=True)

for i in range(3):
  frames, label = next(fg())
  print(f"i:{i}, Shape: {frames.shape}, Label: {label}")
  img = frames[0]
  print(img.shape)

animations = []

for i in range(6):
  print("Creating animation #",i)
  frames, label = next(fg())
  fig, ax = plt.subplots()
  ax.set_title("Label = {}".format(label))
  ims = []
  for frame in frames:
    im = ax.imshow(frame, animated=True)
    ims.append([im])

  animations.append(animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                repeat_delay=0)
                                )


plt.show()
