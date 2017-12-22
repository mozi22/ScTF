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



    # if name == "conv4":
    #     paddings = tf.constant([[0, 0],[2, 2], [4, 4],[0,0]])
    #     inputs = tf.pad(inputs,paddings,"CONSTANT",name=name)

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
        name="conv1_pred_flow"
    )
    
    output = convrelu2(
        inputs=tmp,
        filters=3,
        kernel_size=3,
        stride=1,
        name="conv2_pred_flow"
    )
    
    return output

def _refine(inp, num_outputs, upsampled_prediction=None, features_direct=None,name=None):
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



    print('inside')
    print(inp)
    print(features_direct)
    print(upsampled_features)

    inputs = [upsampled_features, features_direct, upsampled_prediction]
    concat_inputs = [ x for x in inputs if not x is None ]


    return tf.concat(concat_inputs, axis=3)


def train_network(image_pair):

    # contracting part
    with tf.variable_scope('down_convs'):
        conv1 = convrelu2(name='conv1', inputs=image_pair, filters=32, kernel_size=13, stride=2)
        conv2 = convrelu2(name='conv2', inputs=conv1, filters=64, kernel_size=12, stride=2)
        conv3 = convrelu2(name='conv3', inputs=conv2, filters=128, kernel_size=10, stride=2)
        conv4 = convrelu2(name='conv4', inputs=conv3, filters=256, kernel_size=10, stride=2)
        # conv5 = convrelu2(name='conv5', inputs=conv4, filters=512, kernel_size=8, stride=2)

    print('mozi')
    print(conv1)
    print(conv2)
    print(conv3)
    print(conv4)
    # print(conv5)

    conv4_shape = conv4.get_shape().as_list()

    sliced = tf.slice(conv4, [0,0,0,0], conv4_shape)
    result = tf.contrib.layers.flatten(sliced)

    units = 1
    for i in range(1,len(conv4_shape)):
        units *= conv4_shape[i]


    dense = tf.layers.dense(inputs=result, units=units, activation=tf.nn.relu)

    print("dense")
    print(dense)
    # reshaping back to convolution structure
    conv4_flow = tf.concat((conv4,tf.reshape(dense, conv4_shape)),axis=3)

    # predict flow
    with tf.variable_scope('predict_flow5'):
        predict_flow4 = _predict_flow(conv4_flow)


    with tf.variable_scope('upsample_flow4to3'):
        predict_flow4to3 = _upsample_prediction(predict_flow4, 3)

    with tf.variable_scope('refine3'):
        concat3 = _refine(
            inp=conv4_flow, 
            num_outputs=128,
            upsampled_prediction=predict_flow4to3, 
            features_direct=conv3,
            name='paddit'
        )


    # with tf.variable_scope('refine3'):
    #     concat3 = _refine(
    #         inp=concat4, 
    #         num_outputs=128, 
    #         features_direct=conv3
    #     )

    with tf.variable_scope('refine2'):
        concat2 = _refine(
            inp=concat3, 
            num_outputs=64,
            features_direct=conv2
        )

    with tf.variable_scope('refine1'):
        concat1 = _refine(
            inp=concat2, 
            num_outputs=32, 
            features_direct=conv1
        )


    with tf.variable_scope('predict_flow2'):
        predict_flow2 = _predict_flow(concat1)

    print(predict_flow2)
    return predict_flow4, predict_flow2 
