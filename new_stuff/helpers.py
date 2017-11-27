
import tensorflow as tf
from datasets import flowers
from tensorflow.contrib import slim
from preprocessing import inception_preprocessing
import network


def my_cnn(img_pair):  # is_training is not used...
        predict_flow5, predict_flow2 = network.train_network(img_pair)
        return predict_flow5, predict_flow2

def load_batch(dataset, batch_size=32, height=96, width=128, is_training=True):
    """Loads a single batch of data.

    Args:
      dataset: The dataset to load.
      batch_size: The number of images in the batch.
      height: The size of each image after preprocessing.
      width: The size of each image after preprocessing.
      is_training: Whether or not we're currently training or evaluating.
    
    Returns:
      images: A Tensor of size [batch_size, height, width, 3], image samples that have been preprocessed.
      images_raw: A Tensor of size [batch_size, height, width, 3], image samples that can be used for visualization.
      labels: A Tensor of size [batch_size], whose values range between 0 and dataset.num_classes.
    """
    data_provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset, common_queue_capacity=32,
        common_queue_min=8)
    image1, image2, flo = data_provider.get(['image1', 'image2','flo'])

    image1, image2, flo = fix_variables_randomness(image1,image2,flo)
    # image1 = inception_preprocessing.preprocess_image(image1, height, width, is_training=is_training)
    # image2 = inception_preprocessing.preprocess_image(image2, height, width, is_training=is_training)
    # flo = inception_preprocessing.preprocess_image(flo, height, width, is_training=is_training)
    
    # Preprocess the image for display purposes.

    # image_raw = tf.expand_dims(image_raw, 0)
    # image_raw = tf.image.resize_images(image_raw, [height, width])
    # image_raw = tf.squeeze(image_raw)

    # Batch it up.
    images, images_raw, labels = tf.train.batch(
          [image1, image2, flo],
          batch_size=batch_size,
          num_threads=1,
          capacity=2 * batch_size)
    return image1, image2, flo


def fix_variables_randomness(var1,var2,var3):

  name1 = "image/img1:0"
  name2 = "image/img2:0"
  name3 = "image/flo:0"

  image1 = None
  image2 = None
  flo = None

  if var1.name == name1:
    image1 = var1
  elif var1.name == name2:
    image2 = var1
  elif var1.name == name3:
    flo = var1

  if var2.name == name1:
    image1 = var2
  elif var2.name == name2:
    image2 = var2
  elif var2.name == name3:
    flo = var2

  if var3.name == name1:
    image1 = var3
  elif var3.name == name2:
    image2 = var3
  elif var3.name == name3:
    flo = var3

  return image1,image2,flo