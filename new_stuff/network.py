import tensorflow as tf

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


def _upsample_prediction(inp, num_outputs):
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
        activation=tf.nn.relu,
        name="upconv"
    )
    return output

def _predict_flow(inp):
    """Generates a tensor for optical flow prediction
    
    inp: Tensor

    predict_confidence: bool
        If True the output tensor has 4 channels instead of 2.
        The last two channels are the x and y flow confidence.
    """

    tmp = convrelu2(
        inputs=inp,
        filters=24,
        kernel_size=3,
        stride=1,
        name="conv1"
    )
    
    output = convrelu2(
        inputs=tmp,
        filters=2,
        kernel_size=3,
        stride=1,
        name="conv2"
    )
    
    return output

def _refine(inp, num_outputs, upsampled_prediction=None, features_direct=None):
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
    upsampled_features = tf.layers.conv2d_transpose(
        inputs=inp,
        filters=num_outputs,
        kernel_size=4,
        strides=2,
        padding='same',
        activation=tf.nn.relu,
        name="upconv"
    )
    inputs = [upsampled_features, features_direct, upsampled_prediction]
    concat_inputs = [ x for x in inputs if not x is None ]
    return tf.concat(concat_inputs, axis=3)


def train_network(image_pair):

    # contracting part
    conv1 = convrelu2(name='conv1', inputs=image_pair, filters=32, kernel_size=13, stride=2)
    conv2 = convrelu2(name='conv2', inputs=conv1, filters=64, kernel_size=12, stride=2)
    conv3 = convrelu2(name='conv3', inputs=conv2, filters=128, kernel_size=10, stride=2)
    conv4 = convrelu2(name='conv4', inputs=conv3, filters=256, kernel_size=10, stride=2)
    conv5 = convrelu2(name='conv5', inputs=conv4, filters=512, kernel_size=8, stride=2)

    conv5_shape = conv5.get_shape().as_list()

    sliced = tf.slice(conv5, [0,0,0,0], conv5_shape)
    result = tf.contrib.layers.flatten(sliced)

    units = 1
    for i in range(1,len(conv5_shape)):
        units *= conv5_shape[i]


    dense = tf.layers.dense(inputs=result, units=units, activation=tf.nn.relu)

    # reshaping back to convolution structure
    conv5_flow = tf.concat((conv5,tf.reshape(dense, conv5_shape)),axis=3)

    # predict flow
    with tf.variable_scope('predict_flow5'):
        predict_flow5 = _predict_flow(conv5_flow)


    with tf.variable_scope('upsample_flow5to4'):
        predict_flow5to4 = _upsample_prediction(predict_flow5, 2)

    with tf.variable_scope('refine4'):
        concat4 = _refine(
            inp=conv5_flow, 
            num_outputs=256, 
            upsampled_prediction=predict_flow5to4, 
            features_direct=conv4
        )


    with tf.variable_scope('refine3'):
        concat3 = _refine(
            inp=concat4, 
            num_outputs=128, 
            features_direct=conv3
        )

    with tf.variable_scope('refine2'):
        concat2 = _refine(
            inp=concat3, 
            num_outputs=64, 
            features_direct=conv2
        )
    # with tf.variable_scope('refine1'):
    #     concat1 = _refine(
    #         inp=concat2, 
    #         num_outputs=32, 
    #         features_direct=conv1
    #     )

    with tf.variable_scope('predict_flow2'):
        predict_flow2 = _predict_flow(concat2)

    return predict_flow5, predict_flow2 
