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

clip_index = 0

healthy_first_clip_with_bg = avi_healthy.get_frames_of_clip(clip_index)

fig1, ax1 = plt.subplots()
ims1 = []

for frame in healthy_first_clip_with_bg:
      im = ax1.imshow(frame, animated=True)
      ims1.append([im])

ani1 = animation.ArtistAnimation(fig1, ims1, interval=1000, blit=True,
                                repeat_delay=0)


bg = avi_healthy.calcBackground(healthy_first_clip_with_bg)

plt.figure("Background")

plt.imshow(bg)

avi_without_bg_healthy = framegenerator.AVIfile(
      path_to_healthy, "Healthy", crop_rect = crop, clip_length = healthy_clip_len, subtract_background = True)

healthy_first_clip_bg_subtracted = avi_without_bg_healthy.get_frames_of_clip(clip_index)

fig2, ax2 = plt.subplots()
ims2 = []

for frame in healthy_first_clip_bg_subtracted:
      im = ax2.imshow(frame, animated=True)
      ims1.append([im])

ani2 = animation.ArtistAnimation(fig2, ims2, interval=50, blit=True,
                              repeat_delay=0)


plt.show()
