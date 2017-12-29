import tensorflow as tf
import re
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

def change_nans_to_zeros(x):
    
    return tf.where(tf.is_nan(x), tf.zeros_like(x), x)

def _activation_summary(x):
  """Helper to create summaries for activations.
  Creates a summary that provides a histogram of activations.
  Creates a summary that measure the sparsity of activations.
  Args:
    x: Tensor
  Returns:
    nothing
  """
  tensor_name = re.sub('%s_[0-9]*/' % 'SceneFlow', '', x.op.name)
  tf.summary.histogram(tensor_name + '/activations', x)
  tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def _summarize_bias_n_weights(bias_name,weights_name=None):



    if not weights_name == None:
        kernel = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, weights_name)
        tf.summary.histogram(weights_name,kernel)
    
    bias = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, bias_name)
    tf.summary.histogram(bias_name,bias)

def train_network(image_pair):

    # contracting part
    with tf.variable_scope('down_convs'):
        conv1 = convrelu2(name='conv1', inputs=image_pair, filters=32, kernel_size=3, stride=2)
        # conv1 = change_nans_to_zeros(conv1)

        conv2 = convrelu2(name='conv2', inputs=conv1, filters=64, kernel_size=2, stride=2)
        # conv2 = change_nans_to_zeros(conv2)

        conv3 = convrelu2(name='conv3', inputs=conv2, filters=128, kernel_size=2, stride=2)
        # conv3 = change_nans_to_zeros(conv3)

        conv4 = convrelu2(name='conv4', inputs=conv3, filters=256, kernel_size=2, stride=2)
        # conv4 = change_nans_to_zeros(conv4)
        # conv5 = convrelu2(name='conv5', inputs=conv4, filters=512, kernel_size=8, stride=2)

        _summarize_bias_n_weights('down_convs/conv1x/bias:0','down_convs/conv1x/kernel:0')
        _summarize_bias_n_weights('down_convs/conv2x/bias:0','down_convs/conv2x/kernel:0')
        _summarize_bias_n_weights('down_convs/conv3x/bias:0','down_convs/conv3x/kernel:0')
        _summarize_bias_n_weights('down_convs/conv4x/bias:0','down_convs/conv4x/kernel:0')


        _activation_summary(conv1)
        _activation_summary(conv2)
        _activation_summary(conv3)
        _activation_summary(conv4)

    conv4_shape = conv4.get_shape().as_list()

    sliced = tf.slice(conv4, [0,0,0,0], conv4_shape)
    result = tf.contrib.layers.flatten(sliced)

    units = 1
    for i in range(1,len(conv4_shape)):
        units *= conv4_shape[i]


    dense = tf.layers.dense(inputs=result, units=units, activation=tf.nn.sigmoid)
    dense = change_nans_to_zeros(dense)

    # reshaping back to convolution structure
    conv4_flow = tf.concat((conv4,tf.reshape(dense, conv4_shape)),axis=3)

    # predict flow
    with tf.variable_scope('predict_flow5'):


        predict_flow4 = _predict_flow(conv4_flow)
        # predict_flow4 = change_nans_to_zeros(predict_flow4)

        _summarize_bias_n_weights('predict_flow5/conv2_pred_flowx/bias:0','predict_flow5/conv2_pred_flowx/kernel:0')


    with tf.variable_scope('upsample_flow4to3'):
        predict_flow4to3 = _upsample_prediction(predict_flow4, 3)
        # predict_flow4to3 = change_nans_to_zeros(predict_flow4to3)

        _summarize_bias_n_weights('upsample_flow4to3/upconv/bias:0','upsample_flow4to3/upconv/kernel:0')


    with tf.variable_scope('refine3'):
        concat3 = _refine(
            inp=conv4_flow, 
            num_outputs=128,
            upsampled_prediction=predict_flow4to3, 
            features_direct=conv3,
            name='paddit'
        )
        # concat3 = change_nans_to_zeros(concat3)

        _summarize_bias_n_weights('refine3/upconv/bias:0','refine3/upconv/kernel:0')


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
        # concat2 = change_nans_to_zeros(concat2)
        _summarize_bias_n_weights('refine2/upconv/bias:0','refine2/upconv/kernel:0')

    with tf.variable_scope('refine1'):
        concat1 = _refine(
            inp=concat2, 
            num_outputs=32, 
            features_direct=conv1
        )
        # concat1 = change_nans_to_zeros(concat1)
        _summarize_bias_n_weights('refine1/upconv/bias:0','refine1/upconv/kernel:0')

    _activation_summary(predict_flow4to3)
    _activation_summary(concat3)
    _activation_summary(concat2)
    _activation_summary(concat1)




    with tf.variable_scope('predict_flow2'):

        predict_flow2 = _predict_flow(concat1)
        # predict_flow2 = change_nans_to_zeros(predict_flow2)
        # _summarize_bias_n_weights('predict_flow2/conv2_pred_flow/bias:0','predict_flow2/conv2_pred_flow/kernel:0')

    _activation_summary(predict_flow2)
    _activation_summary(predict_flow4)

    # for v in tf.trainable_variables():
    #     v = change_nans_to_zeros(v)
    #     _summarize_bias_n_weights(v.name)

    return predict_flow4, predict_flow2 
