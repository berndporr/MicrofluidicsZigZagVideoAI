#!/usr/bin/python3

import tqdm
import random
import pathlib
import itertools
import collections
import copy
import ruptures as rpt

import os
import cv2
import numpy as np
import tensorflow as tf

class AVIfile:

  def __init__(self, video_path, label_name, clip_length, crop_rect = False, frame_step = 1, subtract_background = ""):
    """
    Wrapper for the raw AVI file.
    video_path: path to the AVI video clip
    label_name: label of the dataset, for example healthy or ill
    clip_length: the number of frames per clip
    crop_rect: rectangle to crop the frames
    frame_step: defines how many frames to skip to reduce the amount of data
    frames2ret: frames per clip to return if not the whole clip_length is taken
    subtract_background: if background subtraction should be performed or not
    """
    self.clip_length = clip_length
    self.label_name = label_name
    self.crop_rect = crop_rect
    self.frame_step = frame_step
    self.subtract_background = subtract_background
    self.frames2ret = clip_length
    self.src = cv2.VideoCapture(str(video_path))
    if not self.src:
      print("Could not open:",video_path)
      quit()
    self.video_length = int(self.src.get(cv2.CAP_PROP_FRAME_COUNT))
    self.width  = int(self.src.get(cv2.CAP_PROP_FRAME_WIDTH))   # float `width`
    self.height = int(self.src.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float `height`
    print("{} opened: {} frames, {} clips at {}x{}, bgsub={}".format(video_path,self.video_length,self.get_number_of_clips(),
    self.width,self.height,self.subtract_background))

  def calcMovement(self,frames):
    m = []
    for i in range(len(frames)-2):
      f1 = frames[i]
      f2 = frames[i+2]
      df = np.abs(f1 - f2)
      df = df**2
      m.append(np.max(df))
    m = np.array(m)
    m -= np.average(m)
    return m

  def calcBackground(self,allframes):
    halfwidth = allframes.shape[2]//2
    # print("allframes.shape",allframes.shape)
    # print(0,0,allframes.shape[1],halfwidth)
    left = tf.image.crop_to_bounding_box(allframes,0,0,allframes.shape[1],halfwidth)
    # print(0,halfwidth,allframes.shape[1],allframes.shape[2]-halfwidth)
    right = tf.image.crop_to_bounding_box(allframes,0,halfwidth,allframes.shape[1],allframes.shape[2]-halfwidth)
    m_left = self.calcMovement(left)
    m_right = self.calcMovement(right)
    # print("Left:",m_left)
    # print("Right:",m_right)
    m_diff = m_left-m_right
    # print("Diff:",m_diff)
    frame_no_cell_in_2nd_half = halfwidth
    for i in range(len(m_diff)-1):
      # print("Diff: ",i,m_diff[i])
      if (m_diff[i] < 0) and (m_diff[i+1] < 0):
        frame_no_cell_in_2nd_half = i
        break
    print("Calculating background: cell in 2nd half at frame #{}.".format(frame_no_cell_in_2nd_half))
    left_avg = np.zeros_like(left[0])
    n = 0
    a = frame_no_cell_in_2nd_half + ( len(allframes) - frame_no_cell_in_2nd_half ) // 2
    if a > (len(allframes) - 1):
      a = len(allframes) - 1
    for i in range(a,len(allframes)):
      left_avg += left[i]
      n += 1
    left_avg = left_avg / n
    right_avg = np.zeros_like(right[0])
    n = 0
    b = frame_no_cell_in_2nd_half // 2
    if b == 0:
      b = 1
    for i in range(0,b):
      right_avg += right[i]
      n += 1
    right_avg = right_avg / n
    avg = np.concatenate((left_avg, right_avg), axis=1)
    # print("BG avg.shape=",avg.shape)
    return avg

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
    print("---> {} going to frame pos {}, clip index = {}.".
      format(self.label_name,framepos,clip_index))
    self.src.set(cv2.CAP_PROP_POS_FRAMES,framepos)

    frames2ignore = 0
    if "opencv" in self.subtract_background:
      backSub = cv2.createBackgroundSubtractorMOG2(history= 100, varThreshold=1)
      frames2ignore = 5

    result = []
    
    for i in range(self.frames2ret):
      ret, frame = self.src.read()
      if not ret:
          print("Fatal error. Could not read frame #",i)
          quit()
      if "opencv" in self.subtract_background:
        fgMask = backSub.apply(frame)
        mask = cv2.cvtColor(fgMask, cv2.COLOR_GRAY2BGR)
        frame = cv2.bitwise_and(frame, mask)
        print("O",end="")
      if frames2ignore > 0:
        frames2ignore -= 1
      else:
        if (i % self.frame_step) == 0:
          frame = self.format_frames(frame)
          result.append(frame)

    ## returning the array but fixing openCV's odd BGR format to RGB
    result = np.array(result)[...,[2,1,0]]

    if self.crop_rect:
      result = tf.image.crop_to_bounding_box(result, 
      self.crop_rect[0][1], self.crop_rect[0][0], 
      self.crop_rect[1][1]-self.crop_rect[0][1], self.crop_rect[1][0]-self.crop_rect[0][0])

    if "split" in self.subtract_background:
      background = self.calcBackground(result)
      result = result - background

    if "abs" in self.subtract_background:
      result = np.abs(result)
    elif "rect" in self.subtract_background:
      result = np.maximum(result, 0)
    elif len(self.subtract_background) > 1:
      print("Fatal. No bg postprocesssing method")
      quit()

    return result


class FrameGenerator:
  """
  Class which feeds a list of label/video into the TF film classifier. Done as a stream.
  """
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
