#!/usr/bin/python3

import framegenerator
import cv2

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import sys

path_to_healthy = "/data/RBC-ZigZag/Selection/60xPhotron_20mBar_2_C001H001S0001.avi"
healthy_clip_len = 50

crop = [[000,0],[500,120]]

avi_healthy = framegenerator.AVIfile(path_to_healthy,"Healthy", crop_rect = crop, clip_length = healthy_clip_len)

clip_index = 10
if len(sys.argv) > 1:
      clip_index = int(sys.argv[1])

healthy_first_clip_with_bg = avi_healthy.get_frames_of_clip(clip_index)

animate_orig = True

# expects normalised frames values between 0..1
def saveFrames(allframes, filename):
      sz = (allframes.shape[2], allframes.shape[1])
      vw = cv2.VideoWriter(filename+".avi",cv2.VideoWriter_fourcc(*'MJPG'), 10, sz)
      for f in allframes:
            # back to openCV BGR uint8
            f_sc = np.array(np.array(f) * 255.0, dtype=np.uint8)[...,[2,1,0]]
            #print("savefr",f_sc.shape,"Max=",np.max(f_sc)," min=",np.min(f_sc))
            vw.write(f_sc)


# saving the orig file back as a movie
saveFrames(healthy_first_clip_with_bg,"tmp/orig")
if animate_orig:
      fig1, ax1 = plt.subplots()
      ims1 = []

      for frame in healthy_first_clip_with_bg:
            im = ax1.imshow(frame, animated=True)
            ims1.append([im])

      ani1 = animation.ArtistAnimation(fig1, ims1, interval=50, blit=True, repeat_delay=0)


bg = avi_healthy.calcBackground(healthy_first_clip_with_bg)

show_background = True

if show_background:
      plt.figure("Background")
      plt.imshow(bg)

avi_without_bg_healthy = framegenerator.AVIfile(
      path_to_healthy, "Healthy", crop_rect = crop, clip_length = healthy_clip_len, subtract_background = "rect")

healthy_first_clip_bg_subtracted = avi_without_bg_healthy.get_frames_of_clip(clip_index)

fig2, ax2 = plt.subplots()
ims2 = []

m = np.max(healthy_first_clip_bg_subtracted)

print("Max brightness value after background subtraction =",m)

healthy_first_clip_bg_subtracted /= m

# save the movie without background
saveFrames(healthy_first_clip_bg_subtracted,"tmp/nobg")

for frame in healthy_first_clip_bg_subtracted:
      im = ax2.imshow(frame, animated=True)
      ims2.append([im])

ani2 = animation.ArtistAnimation(fig2, ims2, interval=50, blit=True,
                              repeat_delay=0)


plt.show()
