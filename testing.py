import dataset_writer
import dataset_reader
import numpy as np
from PIL import Image
from tensorflow.examples.tutorials.mnist import mnist

print(mnist.IMAGE_PIXELS)
# writer = dataset_writer.DatasetWriter('SAME','./')

# writer.create_dataset_array()

# # flo_data = writer.read_flo_file('flow10.flo')
# # writer.convert_file('frame10.png','frame11.png',flo_data)
# writer.close_writer()
# reader = dataset_reader.DatasetReader()

# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)