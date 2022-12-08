#!/usr/bin/python3

import tqdm
import random
import pathlib
import itertools
import collections

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation



src = cv2.VideoCapture("/data/ZigZag-Channel-beads4_5um/60xPhotron_C001H001S0005.avi")
video_length = int(src.get(cv2.CAP_PROP_FRAME_COUNT))
width  = int(src.get(cv2.CAP_PROP_FRAME_WIDTH))   # float `width`
height = int(src.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float `height`
print(width,"x",height)

fig, ax = plt.subplots()
ims = []

for i in range(100):
      ret, frame = src.read()
      im = ax.imshow(frame, animated=True)
      ims.append([im])

src.release()

ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                repeat_delay=0)


plt.show()
