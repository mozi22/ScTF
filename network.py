import tensorflow as tf

# def convrelu2(inputs, num_outputs, kernel_size, name, stride, data_format, **kwargs):
#     """Shortcut for two convolution+relu with 1D filter kernels 
    
#     num_outputs: int or (int,int)
#         If num_outputs is a tuple then the first element is the number of
#         outputs for the 1d filter in y direction and the second element is
#         the final number of outputs.
#     """
#     if isinstance(num_outputs,(tuple,list)):
#         num_outputs_y = num_outputs[0]
#         num_outputs_x = num_outputs[1]
#     else:
#         num_outputs_y = num_outputs
#         num_outputs_x = num_outputs

#     if isinstance(kernel_size,(tuple,list)):
#         kernel_size_y = kernel_size[0]
#         kernel_size_x = kernel_size[1]
#     else:
#         kernel_size_y = kernel_size
#         kernel_size_x = kernel_size

#     tmp_y = tf.layers.conv2d(
#         inputs=inputs,
#         filters=num_outputs_y,
#         kernel_size=[kernel_size_y,1],
#         strides=[stride,1],
#         padding='same',
#         activation=myLeakyRelu,
#         kernel_initializer=default_weights_initializer(),
#         data_format=data_format,
#         name=name+'y',
#         **kwargs,
#     )
#     return tf.layers.conv2d(
#         inputs=tmp_y,
#         filters=num_outputs_x,
#         kernel_size=[1,kernel_size_x],
#         strides=[1,stride],
#         padding='same',
#         activation=myLeakyRelu,
#         kernel_initializer=default_weights_initializer(),
#         data_format=data_format,
#         name=name+'x',
#         **kwargs,
#     )


# def flow_block(image_pair, image2_2=None, intrinsics=None, prev_predictions=None, data_format='channels_first', kernel_regularizer=None):
#     """Creates a flow network
    
#     image_pair: Tensor
#         Image pair concatenated along the channel axis.

#     image2_2: Tensor
#         Second image at resolution level 2 (downsampled two times)
        
#     intrinsics: Tensor 
#         The normalized intrinsic parameters

#     prev_predictions: dict of Tensor
#         Predictions from the previous depth block
    
#     Returns a dict with the predictions
#     """
#     conv_params = {'data_format':data_format, 'kernel_regularizer':kernel_regularizer}

#     # contracting part
#     conv1 = convrelu2(name='conv1', inputs=image_pair, num_outputs=(24,32), kernel_size=9, stride=2, **conv_params)

#     conv2 = convrelu2(name='conv2', inputs=conv1, num_outputs=(48,64), kernel_size=7, stride=2, **conv_params)
#     conv2_1 = convrelu2(name='conv2_1', inputs=conv2, num_outputs=64, kernel_size=3, stride=1, **conv_params)    
    
#     conv3 = convrelu2(name='conv3', inputs=conv2_1, num_outputs=(96,128), kernel_size=5, stride=2, **conv_params)
#     conv3_1 = convrelu2(name='conv3_1', inputs=conv3, num_outputs=128, kernel_size=3, stride=1, **conv_params)
    
#     conv4 = convrelu2(name='conv4', inputs=conv3_1, num_outputs=(192,256), kernel_size=5, stride=2, **conv_params)
#     conv4_1 = convrelu2(name='conv4_1', inputs=conv4, num_outputs=256, kernel_size=3, stride=1, **conv_params)
    
#     conv5 = convrelu2(name='conv5', inputs=conv4_1, num_outputs=384, kernel_size=5, stride=2, **conv_params)
#     conv5_1 = convrelu2(name='conv5_1', inputs=conv5, num_outputs=384, kernel_size=3, stride=1, **conv_params)

#     dense_slice_shape = conv5_1.get_shape().as_list()
#     if data_format == 'channels_first':
#         dense_slice_shape[1] = 96
#     else:
#         dense_slice_shape[-1] = 96
#     units = 1
#     for i in range(1,len(dense_slice_shape)):
#         units *= dense_slice_shape[i]
#     dense5 = tf.layers.dense(
#             tf.contrib.layers.flatten(tf.slice(conv5_1, [0,0,0,0], dense_slice_shape)),
#             units=units,
#             activation=myLeakyRelu,
#             kernel_initializer=default_weights_initializer(),
#             kernel_regularizer=kernel_regularizer,
#             name='dense5'
#             )
#     conv5_1_dense5 = tf.concat((conv5_1,tf.reshape(dense5, dense_slice_shape)),  axis=1 if data_format=='channels_first' else 3)

    
#     # expanding part
#     with tf.variable_scope('predict_flow5'):
#         predict_flowconf5 = _predict_flow(conv5_1_dense5, predict_confidence=True, **conv_params)
    
#     with tf.variable_scope('upsample_flow5to4'):
#         predict_flowconf5to4 = _upsample_prediction(predict_flowconf5, 2, **conv_params)
   
#     with tf.variable_scope('refine4'):
#         concat4 = _refine(
#             inp=conv5_1_dense5, 
#             num_outputs=256, 
#             upsampled_prediction=predict_flowconf5to4, 
#             features_direct=conv4_1,
#             **conv_params,
#         )

#     with tf.variable_scope('refine3'):
#         concat3 = _refine(
#             inp=concat4, 
#             num_outputs=128, 
#             features_direct=conv3_1,
#             **conv_params,
#         )

#     with tf.variable_scope('refine2'):
#         concat2 = _refine(
#             inp=concat3, 
#             num_outputs=64, 
#             features_direct=conv2_1,
#             **conv_params,
#         )

#     with tf.variable_scope('predict_flow2'):
#         predict_flowconf2 = _predict_flow(concat2, predict_confidence=True, **conv_params)
 
#     return { 'predict_flowconf5': predict_flowconf5, 'predict_flowconf2': predict_flowconf2 }

def convrelu2(name,inputs, filters, kernel_size, stride):

    # tmp_y = tf.layers.conv2d(
    #     inputs=inputs,
    #     filters=filters,
    #     kernel_size=[kernel_size,1],
    #     strides=[stride,1],
    #     padding='same',
    #     name=name+'y',
    #     activation=tf.nn.relu
    # )
    return tf.layers.conv2d(
        inputs=inputs,
        filters=filters,
        kernel_size=kernel_size,
        strides=stride,
        padding='same',
        activation=tf.nn.relu,
        name=name+'x'
    )

def train_network(image_pair):

    # contracting part
    conv1 = convrelu2(name='conv1', inputs=image_pair, filters=32, kernel_size=14, stride=2)
    conv2 = convrelu2(name='conv2', inputs=conv1, filters=64, kernel_size=12, stride=2)
    conv3 = convrelu2(name='conv3', inputs=conv2, filters=128, kernel_size=10, stride=2)
    conv4 = convrelu2(name='conv4', inputs=conv3, filters=256, kernel_size=10, stride=2)
    conv5 = convrelu2(name='conv5', inputs=conv4, filters=512, kernel_size=8, stride=2)


    # dense_slice_shape = conv5_1.get_shape().as_list()

    # dense5 = tf.layers.dense(
    #         tf.contrib.layers.flatten(tf.slice(conv5_1, [0,0,0,0], dense_slice_shape)),
    #         units=units,
    #         activation=myLeakyRelu,
    #         kernel_initializer=default_weights_initializer(),
    #         kernel_regularizer=kernel_regularizer,
    #         name='dense5'
    #         )
    # conv5_1_dense5 = tf.concat((conv5_1,tf.reshape(dense5, dense_slice_shape)),  axis=1 if data_format=='channels_first' else 3)
