#!/usr/bin/python3

import tqdm
import random
import pathlib
import itertools
import collections
import sys

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


if len(sys.argv) < 2:
      print("Usage: {} <avifile>".format(sys.argv[0]))
      quit()

src = cv2.VideoCapture(sys.argv[1])
video_length = int(src.get(cv2.CAP_PROP_FRAME_COUNT))
width  = int(src.get(cv2.CAP_PROP_FRAME_WIDTH))   # float `width`
height = int(src.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float `height`
print("Frame size: {}x{}".format(width,height))

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
