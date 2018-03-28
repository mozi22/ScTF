
import tensorflow as tf
import numpy as np
import lmbspecialops as sops


# warp the flow values to the image.
def flow_warp(img,flow):

  # returns [16,224,384,6]
  input_size = img.get_shape().as_list()

  x = list(range(0,input_size[1]))
  y = list(range(0,input_size[2]))
  X, Y = tf.meshgrid(x, y)

  # X = Y = 224,384 

  # converting shape of X = Y to (16,224,384), because flow size is (16,224,384,*)
  X = tf.expand_dims(X,0)
  Y = tf.expand_dims(Y,0)


  # X = tf.pad(X,tf.constant([[8,7],[0,0],[0,0]]),"CONSTANT")
  # Y = tf.pad(Y,tf.constant([[8,7],[0,0],[0,0]]),"CONSTANT")

  X = tf.transpose(X,[0,2,1])
  Y = tf.transpose(Y,[0,2,1])

  X = tf.cast(X,np.float32) + flow[:,:,:,0]
  Y = tf.cast(Y,np.float32) + flow[:,:,:,1]

  con = tf.stack([X,Y])

  result = tf.transpose(con,[1,2,3,0])

  return tf.contrib.resampler.resampler(img,result)

def photoconsistency_loss(img,predicted_flow, weight=100):

  with tf.variable_scope('photoconsistency_loss'):

    img1 = img[:,:,:,0:3]
    img2 = img[:,:,:,5:8]

    # warping using predicted flow
    warped_img = flow_warp(img1,predicted_flow)

    # img2 = get_occulation_aware_image(img2,warped_img)
    # img2 = tf.Print(img2,[img2],summarize=10000,message="msg_before")
    # img2 = tf.Print(img2,[img2],summarize=10000,message="msg_after")

    pc_loss = tf.losses.mean_squared_error(warped_img,img2)

    # tf.losses.compute_weighted_loss(pc_loss,weights=weight)

  return pc_loss


def forward_backward_loss(img,predicted_flow,weight=100):

  with tf.variable_scope('fb_loss'):

    img1 = img[:,:,:,0:4]
    img2 = img[:,:,:,4:8]

    # warping using predicted flow
    warped_img = flow_warp(img1,predicted_flow)

    fb_loss = sops.replace_nonfinite(img2 - warped_img)

    tf.losses.compute_weighted_loss(fb_loss,weights=weight)

  return pc_loss

# defined here :: https://arxiv.org/pdf/1702.02295.pdf
def endpoint_loss(gt_flow,predicted_flow,weight=1000):

  with tf.variable_scope('epe_loss'):

    gt_flow = tf.stop_gradient(gt_flow)

    # width * height
    total_num_of_pixels = gt_flow.get_shape().as_list()[1] * gt_flow.get_shape().as_list()[2]

    # get u & v value for gt
    gt_u = tf.slice(gt_flow,[0,0,0,0],[-1,-1,-1,1])
    gt_v = tf.slice(gt_flow,[0,0,0,1],[-1,-1,-1,1])

    # get u & v value for predicted_flow
    pred_u = tf.slice(predicted_flow,[0,0,0,0],[-1,-1,-1,1])
    pred_v = tf.slice(predicted_flow,[0,0,0,1],[-1,-1,-1,1])


    diff_u = sops.replace_nonfinite(gt_u - pred_u)
    diff_v = sops.replace_nonfinite(gt_v - pred_v)

    epe_loss = tf.sqrt((diff_u**2) + (diff_v**2))
    epe_loss = epe_loss / total_num_of_pixels 

    tf.losses.compute_weighted_loss(epe_loss,weights=weight)
  
  return epe_loss


def depth_loss(gt_flow,predicted_flow,weight=300):

  with tf.variable_scope('depth_loss'):

    gt_flow = tf.stop_gradient(gt_flow)

    # L1 loss on depth
    gt_w = tf.slice(gt_flow,[0,0,0,2],[-1,-1,-1,1])
    pred_w = tf.slice(predicted_flow,[0,0,0,2],[-1,-1,-1,1])

    depth_loss = tf.losses.absolute_difference(gt_w,pred_w,weights=weight)

    return depth_loss



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


# taken from DEMON Network
def scale_invariant_gradient_loss( inp, gt, epsilon ):
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

    return tf.add_n(tmp)

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
    masked_img = warped_img * tf.ones(img.get_shape())
    masked_img = masked_img / masked_img
    masked_img = sops.replace_nonfinite(masked_img)
    return sops.replace_nonfinite(masked_img * warped_img)
