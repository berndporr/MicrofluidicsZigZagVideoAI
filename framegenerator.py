#!/usr/bin/python3

import tqdm
import random
import pathlib
import itertools
import collections

import os
import cv2
import numpy as np
import tensorflow as tf

class AVIfile:

  def __init__(self, video_path, label_name, clip_length = 30, crop_rect = False, frame_step = 1):
    self.clip_length = clip_length
    self.label_name = label_name
    self.crop_rect = crop_rect
    self.frame_step = frame_step
    self.src = cv2.VideoCapture(str(video_path))
    if not self.src:
      print("Could not open:",video_path)
    self.video_length = int(self.src.get(cv2.CAP_PROP_FRAME_COUNT))
    self.width  = int(self.src.get(cv2.CAP_PROP_FRAME_WIDTH))   # float `width`
    self.height = int(self.src.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float `height`
    print("{} opened: {} frames, {} clips at {}x{}".format(video_path,self.video_length,self.get_number_of_clips(),
    self.width,self.height))

  def __del__(self):
    self.src.release()

  def format_frames(self,frame):
    frame = tf.image.convert_image_dtype(frame, tf.float32)
    return frame

  def get_number_of_clips(self):
    return self.video_length // self.clip_length

  def get_frames_of_clip(self, clip_index):
    output_size = (self.width, self.height)

    framepos =  clip_index * self.clip_length
    # print("Going to frame pos:",framepos)
    self.src.set(cv2.CAP_PROP_POS_FRAMES,framepos)

    result = []

    ret, frame = self.src.read()
    if not ret:
      print("Frame read error")
      quit()
    frame = self.format_frames(frame)

    result.append(frame)

    for i in range(1,self.clip_length):
      ret, frame = self.src.read()
      if (i % self.frame_step) == 0:
        if ret:
          frame = self.format_frames(frame)
          result.append(frame)
        else:
          result.append(np.zeros_like(result[0]))
    ## returning the array but fixing openCV's odd BGR format to RGB
    result = np.array(result)[...,[2,1,0]]

    if (self.crop_rect):
      result = tf.image.crop_to_bounding_box(result, 
      self.crop_rect[0][1], self.crop_rect[0][0], 
      self.crop_rect[1][1]-self.crop_rect[0][1], self.crop_rect[1][0]-self.crop_rect[0][0])

    return result


class FrameGenerator:
  def __init__(self, avifiles, clips_list, training = False):
    self.avifiles = avifiles
    self.training = training
    self.clips_list = clips_list

  def __call__(self):
    pairs = []
    for clip_index in self.clips_list:
      for label_index in range(len(self.avifiles)):
        pair = (label_index,clip_index)
        pairs.append(pair)

    if self.training:
      random.shuffle(pairs)

    for label_index,clip_index in pairs:
      video_frames = self.avifiles[label_index].get_frames_of_clip(clip_index)
      label = label_index
      yield video_frames, label
