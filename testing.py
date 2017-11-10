import dataset_writer
import dataset_reader
import numpy as np
from PIL import Image
import os
import tensorflow as tf
writer = dataset_writer.DatasetWriter('SAME','./')

writer.create_dataset_array()
# writer.close_writer()
# # flo_data = writer.read_flo_file('flow10.flo')
# # writer.convert_file('frame10.png','frame11.png',flo_data)
# writer.close_writer()
# reader = dataset_reader.DatasetReader()

# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# examples_dir = os.path.dirname(__file__)


# def prepare_input_data(img1, img2, data_format):
#     """Creates the arrays used as input from the two images."""
#     # scale images if necessary
#     if img1.size[0] != 256 or img1.size[1] != 192:
#         img1 = img1.resize((256,192))
#     if img2.size[0] != 256 or img2.size[1] != 192:
#         img2 = img2.resize((256,192))
#     img2_2 = img2.resize((64,48))
        
#     # transform range from [0,255] to [-0.5,0.5]
#     img1_arr = np.array(img1).astype(np.float32)/255 -0.5
#     img2_arr = np.array(img2).astype(np.float32)/255 -0.5
#     img2_2_arr = np.array(img2_2).astype(np.float32)/255 -0.5
#     if data_format == 'channels_first':
#         img1_arr = img1_arr.transpose([2,0,1])
#         img2_arr = img2_arr.transpose([2,0,1])
#         img2_2_arr = img2_2_arr.transpose([2,0,1])
#         image_pair = np.concatenate((img1_arr,img2_arr), axis=0)
#     else:
#         image_pair = np.concatenate((img1_arr,img2_arr),axis=-1)
    
#     print(image_pair.shape)
#     result = {
#         'image_pair': image_pair[np.newaxis,:],
#         'image1': img1_arr[np.newaxis,:], # first image
#         'image2_2': img2_2_arr[np.newaxis,:], # second image with (w=64,h=48)
#     }
#     return result


# if tf.test.is_gpu_available(True):
#     data_format='channels_first'
# else: # running on cpu requires channels_last data format
#     data_format='channels_last'
# img1 = Image.open(os.path.join(examples_dir,'sculpture1.png'))
# img2 = Image.open(os.path.join(examples_dir,'sculpture2.png'))


# input_data = prepare_input_data(img1,img2,data_format)
# print(input_data['image_pair'].shape)