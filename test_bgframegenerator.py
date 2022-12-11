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

healthy_first_clip = avi_healthy.get_frames_of_clip(0)

fig, ax = plt.subplots()
ims = []

for frame in healthy_first_clip:
      im = ax.imshow(frame, animated=True)
      ims.append([im])

ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                repeat_delay=0)


bg = avi_healthy.calcBackground(0)

figure("Background")

plt.imshow(bg)

plt.show()
