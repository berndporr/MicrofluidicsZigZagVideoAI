#!/usr/bin/python3

import framegenerator
import cv2

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

fg = framegenerator.FrameGenerator(avi_files, train_clips_list, training=True)

for i in range(10):
  frames, label = next(fg())
  print(f"i:{i}, Shape: {frames.shape}, Label: {label}")
  img = frames[0]
  print(img.shape)



frames, label = next(fg())
fig, ax = plt.subplots()
ims = []
isFirst = True
for frame in frames:
    im = ax.imshow(frame, animated=True)
    ims.append([im])

ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                repeat_delay=0)


plt.show()
