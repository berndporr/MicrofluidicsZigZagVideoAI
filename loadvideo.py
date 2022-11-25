import tqdm
import random
import pathlib
import itertools
import collections

import os
import cv2
import numpy as np
import tensorflow as tf

def format_frames(frame, output_size):
  """
    Pad and resize an image from a video.

    Args:
      frame: Image that needs to resized and padded. 
      output_size: Pixel size of the output frame image.

    Return:
      Formatted frame with padding of specified output size.
  """
  print("output size",output_size)
  print(f"Frame type {type(frame)}")
  frame = tf.image.convert_image_dtype(frame, tf.float32)
  #frame = tf.image.resize_with_pad(frame, *output_size)
  return frame


def all_frames_from_video_file(video_path, output_scaling = False, frame_step = 1):
  """
    Creates frames from each video file present for each category.

    Args:
      video_path: File path to the video.
      n_frames: Number of frames to be created per video file.
      output_size: Pixel size of the output frame image.

    Return:
      An NumPy array of frames in the shape of (n_frames, height, width, channels).
  """
  # Read each video frame by frame
  result = []
  print("Opening:"+video_path)
  src = cv2.VideoCapture(str(video_path))  

  video_length = int(src.get(cv2.CAP_PROP_FRAME_COUNT))
  width  = src.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
  height = src.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`

  print("Capture of length={} {}x{}".format(video_length,width,height))

  if (output_scaling):
    output_size = (int(width * output_scaling), int(height * output_scaling))
  else:
    output_size = (width, height)

  src.set(cv2.CAP_PROP_POS_FRAMES, 0)
  # ret is a boolean indicating whether read was successful, frame is the image itself
  ret, frame = src.read()
  print("Result:",ret,frame)
  result.append(format_frames(frame, output_size))

  for i in range(video_length - 1):
    print(i)
    ret, frame = src.read()
    if ((i+1) % frame_step) == 0:
      if ret:
        frame = format_frames(frame, output_size)
        result.append(frame)
      else:
        result.append(np.zeros_like(result[0]))
  src.release()
  result = np.array(result)[..., [2, 1, 0]]

  return result


class FrameGenerator:
  def __init__(self, path, n_frames, training = False):
    """ Returns a set of frames with their associated label. 

      Args:
        path: Video file paths.
        n_frames: Number of frames. 
        training: Boolean to determine if training dataset is being created.
    """
    self.path = path
    self.n_frames = n_frames
    self.training = training
    self.class_names = sorted(set(p.name for p in self.path.iterdir() if p.is_dir()))
    self.class_ids_for_name = dict((name, idx) for idx, name in enumerate(self.class_names))

  def get_files_and_class_names(self):
    video_paths = list(self.path.glob('*/*.avi'))
    classes = [p.parent.name for p in video_paths] 
    return video_paths, classes

  def __call__(self):
    video_paths, classes = self.get_files_and_class_names()

    pairs = list(zip(video_paths, classes))

    if self.training:
      random.shuffle(pairs)

    for path, name in pairs:
      video_frames = all_frames_from_video_file(path) 
      label = self.class_ids_for_name[name] # Encode labels
      yield video_frames, label


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
all_frames_from_video_file("/data/ZigZag-Channel-beads4_5um/60xPhotron_C001H001S0004.avi")
