#
#  DeMoN - Depth Motion Network
#  Copyright (C) 2017  Benjamin Ummenhofer, Huizhong Zhou
#  
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
import tensorflow as tf
from .helpers import *
import lmbspecialops as sops


def _predict_flow_caffe_padding(inp, predict_confidence=False, **kwargs ):
    """Generates a tensor for optical flow prediction
    
    inp: Tensor

    predict_confidence: bool
        If True the output tensor has 4 channels instead of 2.
        The last two channels are the x and y flow confidence.
    """

    tmp = convrelu_caffe_padding(
        inputs=inp,
        num_outputs=24,
        kernel_size=3,
        strides=1,
        name="conv1",
        **kwargs,
    )
    
    output = conv2d_caffe_padding(
        inputs=tmp,
        num_outputs=4 if predict_confidence else 2,
        kernel_size=3,
        strides=1,
        name="conv2",
        **kwargs,
    )
    
    return output


def _upsample_prediction(inp, num_outputs, **kwargs ):
    """Upconvolution for upsampling predictions
    
    inp: Tensor
        Tensor with the prediction
        
    num_outputs: int
        Number of output channels.
        Usually this should match the number of channels in the predictions
    """
    output = tf.layers.conv2d_transpose(
        inputs=inp,
        filters=num_outputs,
        kernel_size=4,
        strides=2,
        padding='same',
        activation=None,
        kernel_initializer=default_weights_initializer(),
        name="upconv",
        **kwargs,
    )
    return output



def _refine_caffe_padding(inp, num_outputs, data_format, upsampled_prediction=None, features_direct=None, **kwargs):
    """ Generates the concatenation of 
         - the previous features used to compute the flow/depth
         - the upsampled previous flow/depth
         - the direct features that already have the correct resolution

    inp: Tensor
        The features that have been used before to compute flow/depth

    num_outputs: int 
        number of outputs for the upconvolution of 'features'

    upsampled_prediction: Tensor
        The upsampled flow/depth prediction

    features_direct: Tensor
        The direct features which already have the spatial output resolution
    """
    tmp = tf.layers.conv2d_transpose(
        inputs=inp,
        filters=num_outputs,
        kernel_size=4,
        strides=2,
        padding='VALID',
        activation=myLeakyRelu,
        kernel_initializer=default_weights_initializer(),
        data_format=data_format,
        name="upconv",
        **kwargs,
    )
    target_shape = features_direct.get_shape().as_list()
    upsampled_features = tf.slice(tmp, [0,0,1,1] if data_format=='channels_first' else [0,1,1,0], target_shape)
    inputs = [upsampled_features, features_direct, upsampled_prediction]
    concat_inputs = [ x for x in inputs if not x is None ]
    
    if data_format == 'channels_first':
        return tf.concat(concat_inputs, axis=1)
    else: # NHWC
        return tf.concat(concat_inputs, axis=3)



def flow_block_demon_original(image_pair, image2_2=None, intrinsics=None, prev_predictions=None, data_format='channels_first'):
    """Creates a flow network
    
    image_pair: Tensor
        Image pair concatenated along the channel axis.

    image2_2: Tensor
        Second image at resolution level 2 (downsampled two times)
        
    intrinsics: Tensor 
        The normalized intrinsic parameters

    prev_predictions: dict of Tensor
        Predictions from the previous depth block
    
    Returns a dict with the predictions
    """
    conv_params = {'data_format':data_format}

    # contracting part
    conv1 = convrelu2_caffe_padding(name='conv1', inputs=image_pair, num_outputs=32, kernel_size=9, stride=2, **conv_params)

    conv2 = convrelu2_caffe_padding(name='conv2', inputs=conv1, num_outputs=64, kernel_size=7, stride=2, **conv_params)
    conv2_1 = convrelu2_caffe_padding(name='conv2_1', inputs=conv2, num_outputs=64, kernel_size=3, stride=1, **conv_params)
    

    conv3 = convrelu2_caffe_padding(name='conv3', inputs=conv2_1, num_outputs=128, kernel_size=5, stride=2, **conv_params)
    conv3_1 = convrelu2_caffe_padding(name='conv3_1', inputs=conv3, num_outputs=128, kernel_size=3, stride=1, **conv_params)
    
    conv4 = convrelu2_caffe_padding(name='conv4', inputs=conv3_1, num_outputs=256, kernel_size=5, stride=2, **conv_params)
    conv4_1 = convrelu2_caffe_padding(name='conv4_1', inputs=conv4, num_outputs=256, kernel_size=3, stride=1, **conv_params)
    
    conv5 = convrelu2_caffe_padding(name='conv5', inputs=conv4_1, num_outputs=512, kernel_size=5, stride=2, **conv_params)
    conv5_1 = convrelu2_caffe_padding(name='conv5_1', inputs=conv5, num_outputs=512, kernel_size=3, stride=1, **conv_params)
    
    
    # expanding part
    with tf.variable_scope('predict_flow5'):
        predict_flowconf5 = _predict_flow_caffe_padding(conv5_1, predict_confidence=True, **conv_params)
    
    with tf.variable_scope('upsample_flow5to4'):
        predict_flowconf5to4 = _upsample_prediction(predict_flowconf5, 2, **conv_params)
   
    with tf.variable_scope('refine4'):
        concat4 = _refine_caffe_padding(
            inp=conv5_1, 
            num_outputs=256, 
            upsampled_prediction=predict_flowconf5to4, 
            features_direct=conv4_1,
            **conv_params,
        )

    with tf.variable_scope('refine3'):
        concat3 = _refine_caffe_padding(
            inp=concat4, 
            num_outputs=128, 
            features_direct=conv3_1,
            **conv_params,
        )

    with tf.variable_scope('refine2'):
        concat2 = _refine_caffe_padding(
            inp=concat3, 
            num_outputs=64, 
            features_direct=conv2_1,
            **conv_params,
        )

    with tf.variable_scope('predict_flow2'):
        predict_flowconf2 = _predict_flow_caffe_padding(concat2, predict_confidence=True, **conv_params)
 
    return { 'predict_flowconf5': predict_flowconf5, 'predict_flowconf2': predict_flowconf2 }

