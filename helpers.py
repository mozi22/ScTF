import re
import numpy as np
import tensorflow as tf 

def readPFM(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().decode('utf-8').rstrip())
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale


# warp the flow values to the image.
def warp(img,flow):
    x = list(range(0,self.input_size[0]))
    y = list(range(0,self.input_size[1] + 8))
    X, Y = tf.meshgrid(x, y)

    X = tf.cast(X,np.float32) + flow[:,:,0]
    Y = tf.cast(Y,np.float32) + flow[:,:,1]

    con = tf.stack([X,Y])
    result = tf.transpose(con,[1,2,0])
    result = tf.expand_dims(result,0)
    return tf.contrib.resampler.resampler(img[np.newaxis,:,:,:],result)

# resize the gt_flow to the size of predict_flow4 for minimizing loss also after encoder ( before decoder )
def downsample_label(gt_flow):

  gt_u = tf.slice(gt_flow,[0,0,0,0],[-1,-1,-1,1])
  gt_v = tf.slice(gt_flow,[0,0,0,1],[-1,-1,-1,1])
  gt_w = tf.slice(gt_flow,[0,0,0,2],[-1,-1,-1,1])

  gt_u = tf.image.resize_images(gt_u,[7,12],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  gt_v = tf.image.resize_images(gt_u,[7,12],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  gt_w = tf.image.resize_images(gt_u,[7,12],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  return tf.concat([gt_u,gt_v,gt_w],axis=-1)

# resize the gt_flow to the size of predict_flow4 for minimizing loss also after encoder ( before decoder )
def downsample_label(gt_flow):

  gt_u = tf.slice(gt_flow,[0,0,0,0],[-1,-1,-1,1])
  gt_v = tf.slice(gt_flow,[0,0,0,1],[-1,-1,-1,1])
  gt_w = tf.slice(gt_flow,[0,0,0,2],[-1,-1,-1,1])

  # since we're reducing the size, we need to reduce the flow values by the same factor.
  # decreasing width from 224 to 7 means we decreased the image by a factor of 0.031 ( 224 * 0.031 )
  gt_u = gt_u[:,:,:,0] * 0.031
  gt_v = gt_v[:,:,:,0] * 0.031
  gt_w = gt_w[:,:,:,0] * 0.031

  # decreasing width from 384 to 12 means we decreased the image by a factor of 0.031 ( 384 * 0.031 )
  gt_u = gt_u[:,:,:,1] * 0.031
  gt_v = gt_v[:,:,:,1] * 0.031
  gt_w = gt_w[:,:,:,1] * 0.031

  gt_u = tf.image.resize_images(gt_u,[7,12],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  gt_v = tf.image.resize_images(gt_v,[7,12],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  gt_w = tf.image.resize_images(gt_w,[7,12],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  return tf.concat([gt_u,gt_v,gt_w],axis=-1)

def swap_images_for_back_flow(images):

    '''
        Assuming batch_size = 32

        take half the batch_size and swap the images such that from image 1 to 16 
        batch contains I1,I2 and image 16 to 32 it contains I2,I1.
    '''

    first_rgbd = images[:,:,:,0:4]
    second_rgbd = images[:,:,:,4:8]

    return tf.concat([second_rgbd,first_rgbd],axis=3)