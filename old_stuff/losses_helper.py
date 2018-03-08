
import helpers as hp
import tensorflow as tf

# # gt = ground truth flow matrix
# # flow = predicted flow matrix

# # apply forward flow and than backward flow. The pixel should end up at the same place
# # need to train the data in both direction for this to work


# # def forward_backward_loss(gt,flow)
# 	# function definiton



def photoconsistency_loss(img,gt_flow,predicted_flow, weight=0.2):

  """Calculate the photo consistency loss by warping with gt_flow 
     and predicted_flow and finding their mean difference."""

  # Args:
  #   img:  		The image to warp with. The outer list
  #   flow: 		Predicted flow
  #   gt_flow: 	Ground truth flow

  # warping using gt flow
  gt_warp = hp.warp(img,gt_flow)

  # warping using predicted flow
  warp = hp.warp(img,flow)

  pc_loss = gt_warp - warp

  tf.losses.compute_weighted_loss(pc_loss,weights=weight)


def endpoint_loss(gt_flow,predicted_flow,weight=0.8):

  # get u & v value for gt
  gt_u = tf.slice(gt_flow,[0,0,0,0],[-1,-1,-1,1])
  gt_v = tf.slice(gt_flow,[0,0,0,1],[-1,-1,-1,1])
  # gt_w = tf.slice(gt_flow,[0,0,2],[-1,-1,1])

  # get u & v value for predicted_flow
  pred_u = tf.slice(predicted_flow,[0,0,0,0],[-1,-1,-1,1])
  pred_v = tf.slice(predicted_flow,[0,0,0,1],[-1,-1,-1,1])
  # pred_w = tf.slice(predicted_flow,[0,0,2],[-1,-1,1])

  epe_loss = tf.sqrt(tf.square(tf.subtract(gt_u,pred_u)) + tf.square(tf.subtract(gt_v,pred_v)))
  tf.losses.compute_weighted_loss(epe_loss,weights=weight)
  
  return epe_loss


def mse_loss(label,prediction):
  # Calculate the average cross entropy loss across the batch.
  mse = tf.losses.mean_squared_error(label,prediction)
  return mse


def nan_to_zero(val):
  return tf.where(tf.is_nan(val), tf.add(tf.zeros_like(val),0.000003), val)
