import tensorflow as tf
import numpy as np
import lmbspecialops as sops
import math

# Polynomial Learning Rate

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_float('SIGL_START_LEARNING_RATE', 1,
                            """Where to start the learning.""")
tf.app.flags.DEFINE_float('SIGL_END_LEARNING_RATE', 500,
                            """Where to end the learning.""")
tf.app.flags.DEFINE_float('SIGL_POWER', 2,
                            """How fast the learning rate should go down.""")

# loss value ranges around 0.01 to 0.1
def photoconsistency_loss(img,predicted_flow, weight=10):

  with tf.variable_scope('photoconsistency_loss'):

    img1, img2 = get_separate_rgb_images(img)
    predicted_flow = denormalize_flow(predicted_flow)

    warped_img = flow_warp(img2,predicted_flow)

    warped_img = sops.replace_nonfinite(warped_img)
    img1 = get_occulation_aware_image(img1,warped_img)

    img1 = tf.stop_gradient(img1)

    pc_loss = endpoint_loss(img1, warped_img,weight,'pc_loss')
    # pc_loss = tf.Print(pc_loss,[pc_loss],'pcloss ye hai ')
    # tf.losses.compute_weighted_loss(pc_loss,weights=weight)
    # tf.summary.scalar('pc_loss',sops.replace_nonfinite(pc_loss))

  return pc_loss

def denormalize_flow(flow):

    u_factor = 0.414814815
    v_factor = 0.4

    input_size = math.ceil(960 * v_factor), math.floor(540 * u_factor)

    u = flow[:,:,:,0] * input_size[0]
    v = flow[:,:,:,1] * input_size[1]

    u = tf.expand_dims(u,axis=-1)
    v = tf.expand_dims(v,axis=-1)

    return tf.concat([u,v],axis=-1)

def forward_backward_loss(predicted_flow,weight=1):

  with tf.variable_scope('fb_loss'):

    # predicted_flow = sops.replace_nonfinite(predicted_flow)
    '''
      So, here we do the following steps. 

       1) Use meshgrid to generate pixel positions in X and Y direction
       2) Add forward flow values to the meshgrid X,Y positions, than we get the resulting flow field, we call it A.
       3) Warp A with the backward flow field (by warp I mean use resampler), where we pass in the backward_flow as
          arg1 and A as arg 2. This means now our flow field is pointing backwards, this gives us B.
       4) Now simply perform flow_forward + B to find  the difference between the forward and backward flow fields. 
       5) We've to minimize this difference to 0 ( or loss to 0 ). 
    '''

    # get batch size ( assuming batch_size will always be divisible by 2 )

    tensor_shape = predicted_flow.get_shape().as_list()

    batch_size = tensor_shape[0]
    # if the BS is 32, first 16 images represent forward flow prediction and the next 16 represent backward flow predictions.
    forward_part = batch_size // 2


    # 0 - 16 is the forward flow
    # 17 - 31 is the backward flow
    flow_forward = predicted_flow[0:forward_part,:,:,:]
    flow_backward = predicted_flow[forward_part:batch_size,:,:,:]

    flow_forward = sops.replace_nonfinite(flow_forward)
    flow_backward = sops.replace_nonfinite(flow_backward)
    # flow_forward = tf.Print(flow_forward,[flow_forward],'forward ye hai ')
    # flow_backward = tf.Print(flow_backward,[flow_backward],'backward ye hai ')
    flow_forward = tf.check_numerics(flow_forward,'flow_forward Nan Value found')
    flow_backward = tf.check_numerics(flow_backward,'flow_backward Nan Value found')

    # flow_forward_denormed = denormalize_flow(flow_forward)
    # flow_forward_denormed = sops.replace_nonfinite(flow_forward_denormed)
    flow_forward_denormed = flow_forward
    # tf.summary.image('flow_forward_u',tf.expand_dims(flow_forward[:,:,:,0],axis=-1))
    # tf.summary.image('flow_forward_v',tf.expand_dims(flow_forward[:,:,:,1],axis=-1))
    # tf.summary.image('flow_backward_u',tf.expand_dims(flow_backward[:,:,:,0],axis=-1))
    # tf.summary.image('flow_backward_v',tf.expand_dims(flow_backward[:,:,:,1],axis=-1))

    # step 1,2,3
    B = sops.replace_nonfinite(flow_warp(flow_backward,flow_forward_denormed))

    B = get_occulation_aware_image(flow_forward,B)



    # step 4
    fb_loss = sops.replace_nonfinite(endpoint_loss(-B,flow_forward,weight,'fb_loss',False))

    # tf.losses.compute_weighted_loss(fb_loss,weights=weight)

  return fb_loss

# loss value ranges around 0.01 to 2.0
# defined here :: https://arxiv.org/pdf/1702.02295.pdf
def endpoint_loss(gt_flow,predicted_flow,weight=500,scope='epe_loss',stop_grad=False):

  with tf.variable_scope(scope):

    if stop_grad == False:
      gt_flow = tf.stop_gradient(gt_flow)


    # get u & v value for gt
    gt_u = tf.slice(gt_flow,[0,0,0,0],[-1,-1,-1,1])
    gt_v = tf.slice(gt_flow,[0,0,0,1],[-1,-1,-1,1])

    # get u & v value for predicted_flow
    pred_u = tf.slice(predicted_flow,[0,0,0,0],[-1,-1,-1,1])
    pred_v = tf.slice(predicted_flow,[0,0,0,1],[-1,-1,-1,1])


    diff_u = sops.replace_nonfinite(gt_u - pred_u)
    diff_v = sops.replace_nonfinite(gt_v - pred_v)

    epe_loss = tf.sqrt((diff_u**2) + (diff_v**2) + 1e-6)

    # print('epe krdo')
    # epe_loss = tf.reduce_mean(epe_loss[:,])

    epe_loss = tf.reduce_mean(epe_loss)

    epe_loss = tf.check_numerics(epe_loss,'numeric checker')
    # epe_loss = tf.Print(epe_loss,[epe_loss],'epeloss ye hai ')

    tf.losses.compute_weighted_loss(epe_loss,weights=weight)
  
  return epe_loss



def depth_consistency_loss(img,predicted_optflow_uv,weight=10):

  with tf.variable_scope('depth_consistency_loss'):

    img1_depth, img2_depth = get_separate_depth_images(img)


    img2_depth = tf.expand_dims(img2_depth,axis=3)


    # will return a single channel depth image warped with uv optical flow
    warped_depth_img = flow_warp(img2_depth,predicted_optflow_uv[:,:,:,0:2])

    # loss = w - Z_1(x+u,y+v) + Z_0(x,y)
    dc_loss = predicted_optflow_uv[:,:,:,2] - warped_depth_img[:,:,:,0] + img1_depth
    dc_loss = tf.reduce_mean(dc_loss)


    tf.summary.scalar('dc_loss',sops.replace_nonfinite(dc_loss))
    # tf.losses.compute_weighted_loss(dc_loss,weights=weight)

    return dc_loss


# taken from DEMON Network
def scale_invariant_gradient( inp, deltas, weights, epsilon=0.001):
  """Computes the scale invariant gradient images
  
  inp: Tensor
      
  deltas: list of int
    The pixel delta for the difference. 
    This vector must be the same length as weight.
  weights: list of float
    The weight factor for each difference.
    This vector must be the same length as delta.
  epsilon: float
    epsilon value for avoiding division by zero
      
  """

  with tf.variable_scope('scale_inv_images'):
    inp = tf.transpose(inp,[0,3,1,2])


    assert len(deltas)==len(weights)

    sig_images = []
    for delta, weight in zip(deltas,weights):
        sig_images.append(sops.scale_invariant_gradient(inp, deltas=[delta], weights=[weight], epsilon=epsilon))
  return tf.concat(sig_images,axis=1)



# loss value ranges around 80 to 100
# taken from DEMON Network
def scale_invariant_gradient_loss(inp, gt, epsilon,decay_steps,global_step,weight=100):
  """Computes the scale invariant gradient loss
  inp: Tensor
      Tensor with the scale invariant gradient images computed on the prediction
  gt: Tensor
      Tensor with the scale invariant gradient images computed on the ground truth
  epsilon: float
    epsilon value for avoiding division by zero
  """

  with tf.variable_scope('scale_invariant_gradient_loss'):
    num_channels_inp = inp.get_shape().as_list()[1]
    num_channels_gt = gt.get_shape().as_list()[1]
    assert num_channels_inp%2==0
    assert num_channels_inp==num_channels_gt

    tmp = []
    for i in range(num_channels_inp//2):
        tmp.append(pointwise_l2_loss(inp[:,i*2:i*2+2,:,:], gt[:,i*2:i*2+2,:,:], epsilon))


    tmp = tf.add_n(tmp)



    # weight_increase_rate = tf.train.polynomial_decay(FLAGS.SIGL_START_LEARNING_RATE, global_step,
    #                                                 decay_steps, FLAGS.SIGL_END_LEARNING_RATE,
    #                                                 power=FLAGS.SIGL_POWER)

    # tf.summary.scalar('weight_increase_rate',weight)

    # tmp = tf.Print(tmp,[tmp],'sigl ye hai ')
    tf.losses.compute_weighted_loss(tmp,weights=weight)

    return tmp

# taken from DEMON Network
def pointwise_l2_loss(inp, gt, epsilon, data_format='NCHW'):
    """Computes the pointwise unsquared l2 loss.
    The input tensors must use the format NCHW. 
    This loss ignores nan values. 
    The loss is normalized by the number of pixels.
    
    inp: Tensor
        This is the prediction.
        
    gt: Tensor
        The ground truth with the same shape as 'inp'
        
    epsilon: float
        The epsilon value to avoid division by zero in the gradient computation
    """
    with tf.name_scope('pointwise_l2_loss'):
        gt_ = tf.stop_gradient(gt)
        diff = sops.replace_nonfinite(inp-gt_)
        if data_format == 'NCHW':
            return tf.reduce_mean(tf.sqrt(tf.reduce_sum(diff**2, axis=1)+epsilon))
        else: # NHWC
            return tf.reduce_mean(tf.sqrt(tf.reduce_sum(diff**2, axis=3)+epsilon))


# returns an image with all the occulded pixel values as 0
def get_occulation_aware_image(img,warped_img):
    masked_img = warped_img / warped_img
    masked_img = sops.replace_nonfinite(masked_img)
    return masked_img * img




# factorU = reduces the optical flow U component by the factor with which we reduce the size of image
# factorV = reduces the optical flow V component by the factor with which we reduce the size of image
# factorW = reduces the optical flow W component by the factor with which we reduce the size of image
# size = the size at which you want to resize your original image label.
# gt_flow = ground truth flow label.

# resize the gt_flow to the size of predict_flow4 for minimizing loss also after encoder ( before decoder )
def downsample_label(gt_flow,size=[224,384],factorU=0.5,factorV=0.5):

  gt_u = tf.slice(gt_flow,[0,0,0,0],[-1,-1,-1,1])
  gt_v = tf.slice(gt_flow,[0,0,0,1],[-1,-1,-1,1])
  # gt_w = tf.slice(gt_flow,[0,0,0,2],[-1,-1,-1,1])

  # since we're reducing the size, we need to reduce the flow values by the same factor.
  # decreasing width from 224 to 5 means we decreased the image by a factor ( 384 * 0.022 )
  gt_u = gt_u * factorU
  # decreasing width from 384 to 10 means we decreased the image by a factor ( 384 * 0.026 )
  gt_v = gt_v * factorV
  # decreasing depth, in this case we'll just take the avg of factors of width and height ( 0.026 + 0.024 / 2 )
  # gt_w = gt_w * factorW

  gt_u = tf.image.resize_images(gt_u,size,method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  gt_v = tf.image.resize_images(gt_v,size,method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  # gt_w = tf.image.resize_images(gt_w,size,method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  return tf.concat([gt_u,gt_v],axis=-1)
  # return tf.concat([gt_u,gt_v,gt_w],axis=-1)


def get_separate_rgb_images(img):
  return img[:,:,:,0:3],img[:,:,:,4:7]

def get_separate_depth_images(img):
  return img[:,:,:,3],img[:,:,:,7]

# warp the flow values to the image.
def flow_warp(img,flow):

  # returns [16,224,384,6]
  input_size = img.get_shape().as_list()

  x = list(range(0,input_size[2]))
  y = list(range(0,input_size[1]))

  X, Y = tf.meshgrid(x, y)

  # X = tf.expand_dims(X,0)
  # Y = tf.expand_dims(Y,0)

  X = tf.cast(X,np.float32)
  Y = tf.cast(Y,np.float32)

  X = X + flow[:,:,:,0]
  Y = Y + flow[:,:,:,1]

  con = tf.stack([X,Y])

  result = tf.transpose(con,[1,2,3,0])

  # result = tf.expand_dims(result,0)
  return tf.contrib.resampler.resampler(img,result)
