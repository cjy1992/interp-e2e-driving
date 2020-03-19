# Copyright (c) 2020: Jianyu Chen (jianyuchen@berkeley.edu).
#
# This file is modified from <https://github.com/alexlee-gk/slac>:
# Copyright (c) 2019 alexlee-gk
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
import tensorflow as tf
from tensorflow.core.framework import summary_pb2
from tensorflow.core.util import event_pb2
from tensorflow.python.ops import summary_ops_v2


def encode_gif(images, fps):
  """Encodes numpy images into gif string.
  Args:
    images: A 5-D `uint8` `np.array` (or a list of 4-D images) of shape
      `[batch_size, time, height, width, channels]` where `channels` is 1 or 3.
    fps: frames per second of the animation
  Returns:
    The encoded gif string.
  Raises:
    IOError: If the ffmpeg command returns an error.
  """
  from subprocess import Popen, PIPE
  h, w, c = images[0].shape
  cmd = ['ffmpeg', '-y',
         '-f', 'rawvideo',
         '-vcodec', 'rawvideo',
         '-r', '%.02f' % fps,
         '-s', '%dx%d' % (w, h),
         '-pix_fmt', {1: 'gray', 3: 'rgb24'}[c],
         '-i', '-',
         '-filter_complex',
         '[0:v]split[x][z];[z]palettegen[y];[x]fifo[x];[x][y]paletteuse',
         '-r', '%.02f' % fps,
         '-f', 'gif',
         '-']
  proc = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE)
  for image in images:
    proc.stdin.write(image.tostring())
  out, err = proc.communicate()
  if proc.returncode:
    err = '\n'.join([' '.join(cmd), err.decode('utf8')])
    raise IOError(err)
  del proc
  return out


def py_gif_summary(tag, images, max_outputs, fps):
  """Outputs a `Summary` protocol buffer with gif animations.
  Args:
    tag: Name of the summary.
    images: A 5-D `uint8` `np.array` of shape `[batch_size, time, height, width,
      channels]` where `channels` is 1 or 3.
    max_outputs: Max number of batch elements to generate gifs for.
    fps: frames per second of the animation
  Returns:
    The serialized `Summary` protocol buffer.
  Raises:
    ValueError: If `images` is not a 5-D `uint8` array with 1 or 3 channels.
  """
  is_bytes = isinstance(tag, bytes)
  if is_bytes:
    tag = tag.decode("utf-8")
  images = np.asarray(images)
  if images.dtype != np.uint8:
    raise ValueError("Tensor must have dtype uint8 for gif summary.")
  if images.ndim != 5:
    raise ValueError("Tensor must be 5-D for gif summary.")
  batch_size, _, height, width, channels = images.shape
  if channels not in (1, 3):
    raise ValueError("Tensors must have 1 or 3 channels for gif summary.")

  summ = tf.compat.v1.Summary()
  num_outputs = min(batch_size, max_outputs)
  for i in range(num_outputs):
    image_summ = tf.compat.v1.Summary.Image()
    image_summ.height = height
    image_summ.width = width
    image_summ.colorspace = channels  # 1: grayscale, 3: RGB
    try:
      image_summ.encoded_image_string = encode_gif(images[i], fps)
    except (IOError, OSError) as e:
      tf.logging.warning(
          "Unable to encode images to a gif string because either ffmpeg is "
          "not installed or ffmpeg returned an error: %s. Falling back to an "
          "image summary of the first frame in the sequence.", e)
      try:
        from PIL import Image  # pylint: disable=g-import-not-at-top
        import io  # pylint: disable=g-import-not-at-top
        with io.BytesIO() as output:
          Image.fromarray(images[i][0]).save(output, "PNG")
          image_summ.encoded_image_string = output.getvalue()
      except:
        tf.logging.warning(
            "Gif summaries requires ffmpeg or PIL to be installed: %s", e)
        image_summ.encoded_image_string = "".encode('utf-8') if is_bytes else ""
    if num_outputs == 1:
      summ_tag = "{}/gif".format(tag)
    else:
      summ_tag = "{}/gif/{}".format(tag, i)
    summ.value.add(tag=summ_tag, image=image_summ)
  summ_str = summ.SerializeToString()
  return summ_str


def gif_summary_v2(name, tensor, max_outputs, fps, family=None, step=None):

  def py_gif_event(step, tag, tensor, max_outputs, fps):
    summary = py_gif_summary(tag, tensor, max_outputs, fps)

    if isinstance(summary, bytes):
      summ = summary_pb2.Summary()
      summ.ParseFromString(summary)
      summary = summ

    event = event_pb2.Event(summary=summary)
    event.wall_time = time.time()
    event.step = step
    event_pb = event.SerializeToString()
    return event_pb

  def function(tag, scope):
    # Note the identity to move the tensor to the CPU.
    event = tf.numpy_function(
        py_gif_event,
        [_choose_step(step), tag, tf.identity(tensor), max_outputs, fps],
        tf.string)
    return summary_ops_v2.import_event(event, name=scope)

  return summary_ops_v2.summary_writer_function(
      name, tensor, function, family=family)


def _choose_step(step):
  if step is None:
    return tf.compat.v1.train.get_or_create_global_step()
  if not isinstance(step, tf.Tensor):
    return tf.convert_to_tensor(step, tf.int64)
  return step


def gif_summary(name, images, fps, saturate=False, step=None):
  images = tf.image.convert_image_dtype(images, tf.uint8, saturate=saturate)
  output = tf.concat(tf.unstack(images), axis=2)[None]
  gif_summary_v2(name, output, 1, fps, step=step)
