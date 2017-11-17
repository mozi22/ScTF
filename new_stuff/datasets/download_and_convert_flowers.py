# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Downloads and converts Flowers data to TFRecords of TF-Example protos.

This module downloads the Flowers data, uncompresses it, reads the files
that make up the Flowers data and creates two TFRecord datasets: one for train
and one for test. Each TFRecord dataset is comprised of a set of TF-Example
protocol buffers, each of which contain a single image and label.

The script should take about a minute to run.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import numpy as np

import tensorflow as tf

from datasets import dataset_utils

# The URL where the Flowers data can be downloaded.
_DATA_URL = 'http://download.tensorflow.org/example_images/flower_photos.tgz'

_TRAINING_DIR = ''
_TEST_DIR = ''

# The number of images in the validation set.
_NUM_VALIDATION = 350

# Seed for repeatability.
_RANDOM_SEED = 0

# The number of shards per dataset split.
_NUM_SHARDS = 1

class ImageReader(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

  def read_image_dims(self, sess, image_data):
    image = self.decode_jpeg(sess, image_data)
    return image.shape[0], image.shape[1]

  def decode_jpeg(self, sess, image_data):
    image = sess.run(self._decode_jpeg,
                     feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image

def _get_filenames_and_classes(data_dir,label_dir):
  """Returns a list of filenames and inferred class names.

  Args:
    data_dir: A directory containing a set of subdirectories with features(images).
    label_dir: A directory containing a set of subdirectories with labels(images).

  Returns:
    A list of image file paths, relative to `dataset_dir` and the list of
    subdirectories, representing class names.
  """
  data_root = os.path.join(data_dir)
  label_root = os.path.join(label_dir)
  directories = []
  class_names = []
  for filename in os.listdir(data_root):
    path_feature = os.path.join(data_root, filename)
    path_label = os.path.join(label_root, filename)
    if os.path.isdir(path_feature) and os.path.isdir(path_label):
      directories.append(path_feature)
      class_names.append(path_label)

  photo_filenames = []
  for directory in directories:
    for filename in os.listdir(directory):
      path = os.path.join(directory, filename)
      photo_filenames.append(path)

  flo_filenames = []
  for directory in class_names:
    for filename in os.listdir(directory):
      path = os.path.join(directory, filename)
      flo_filenames.append(path)

  return photo_filenames, flo_filenames


def _get_dataset_filename(dataset_dir, split_name, shard_id):
  output_filename = 'opticflow_%s_%05d-of-%05d.tfrecord' % (
      split_name, shard_id, _NUM_SHARDS)
  return os.path.join(dataset_dir, output_filename)


def _convert_dataset(split_name, images, labels, dataset_dir):
  """Converts the given images to a TFRecord dataset.

  Args:
    split_name: The name of the dataset, either 'train' or 'validation'.
    images: A list of absolute paths to png or jpg images.
    class_names_to_ids: A dictionary from class names (strings) to ids
      (integers).
    dataset_dir: The directory where the converted datasets are stored.
  """
  assert split_name in ['train', 'validation']

  num_per_shard = int(math.ceil(len(images) / float(_NUM_SHARDS)))
  with tf.Graph().as_default():
    image_reader = ImageReader()

    with tf.Session('') as sess:

      for shard_id in range(_NUM_SHARDS):
        output_filename = _get_dataset_filename(dataset_dir, split_name, shard_id)

        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
          start_ndx = shard_id * num_per_shard
          end_ndx = min((shard_id+1) * num_per_shard, len(images))
          for i in range(start_ndx, end_ndx):
            sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                i+1, len(images), shard_id))
            sys.stdout.flush()

            # Read the filename:
            image_data = tf.gfile.FastGFile(images[i], 'rb').read()
            height, width = image_reader.read_image_dims(sess, image_data)
            img_flo = read_flo_file(labels[i])
            print(img_flo.shape)

            # class_name = os.path.basename(os.path.dirname(images[i]))
            # class_id = class_names_to_ids[class_name]

            example = dataset_utils.image_to_tfexample(
                image_data, b'jpg', height, width, img_flo.tostring())
            tfrecord_writer.write(example.SerializeToString())
  sys.stdout.write('\n')
  sys.stdout.flush()


def read_flo_file(file_path):
  with open(file_path, 'rb') as f:

    magic = np.fromfile(f, np.float32, count=1)

    if 202021.25 != magic:
      print('Magic number incorrect. Invalid .flo file')
    else:
      w = np.fromfile(f, np.int32, count=1)[0]
      h = np.fromfile(f, np.int32, count=1)[0]

      data = np.fromfile(f, np.float32, count=2*w*h)

      # Reshape data into 3D array (columns, rows, bands)
      data2D = np.resize(data, (w, h, 2))
      return data2D


def _clean_up_temporary_files(dataset_dir):
  """Removes temporary files used to create the dataset.

  Args:
    dataset_dir: The directory where the temporary files are stored.
  """
  filename = _DATA_URL.split('/')[-1]
  filepath = os.path.join(dataset_dir, filename)
  tf.gfile.Remove(filepath)

  tmp_dir = os.path.join(dataset_dir, 'opticflow_photos')
  tf.gfile.DeleteRecursively(tmp_dir)


def _dataset_exists(dataset_dir):
  for split_name in ['train', 'validation']:
    for shard_id in range(_NUM_SHARDS):
      output_filename = _get_dataset_filename(
          dataset_dir, split_name, shard_id)
      if not tf.gfile.Exists(output_filename):
        return False
  return True


def run(data_dir,label_dir):
  """Runs the download and conversion operation.

  Args:
    dataset_dir: The dataset directory where the dataset is stored.
  """
  if not tf.gfile.Exists(data_dir):
    tf.gfile.MakeDirs(data_dir)

  if not tf.gfile.Exists(label_dir):
    tf.gfile.MakeDirs(label_dir)

  # if _dataset_exists(dataset_dir):
  #   print('Dataset files already exist. Exiting without re-creating them.')
  #   return

  # dataset_utils.download_and_uncompress_tarball(_DATA_URL, dataset_dir)


  photo_filenames, labels = _get_filenames_and_classes(data_dir,label_dir)
  # class_names_to_ids = dict(zip(class_names, range(len(class_names))))
  # # Divide into train and test:
  # random.seed(_RANDOM_SEED)
  # random.shuffle(photo_filenames)
  # training_filenames = photo_filenames[_NUM_VALIDATION]
  # validation_filenames = photo_filenames[:_NUM_VALIDATION]
  training_filenames = photo_filenames
  training_lbls_filenames = labels

  # training_lbls_filenames = labels[_NUM_VALIDATION]
  # validation_lbls_filenames = labels[:_NUM_VALIDATION]
  # # First, convert the training and validation sets.
  _convert_dataset('train', training_filenames, training_lbls_filenames, './tffiles')
  # _convert_dataset('validation', validation_filenames, class_names_to_ids,
  #                  dataset_dir)

  # # Finally, write the labels file:
  # labels_to_class_names = dict(zip(range(len(class_names)), class_names))
  # dataset_utils.write_label_file(labels_to_class_names, dataset_dir)

  # _clean_up_temporary_files(dataset_dir)
  # print('\nFinished converting the Flowers dataset!')