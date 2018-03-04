
import helpers as hp
import tensorflow as tf

# # gt = ground truth flow matrix
# # flow = predicted flow matrix

# # apply forward flow and than backward flow. The pixel should end up at the same place
# # need to train the data in both direction for this to work


# # def forward_backward_loss(gt,flow)
# 	# function definiton



def photoconsistency_loss(img,gt_flow,flow, weight=0.3):
	
	"""Calculate the photo consistency loss by warping with gt_flow 
	   and predicted_flow and finding their mean difference."""

	# Args:
	#   img:  		The image to warp with. The outer list
	#   flow: 		Predicted flow
	#   gt_flow: 	Ground truth flow

	flow = hp.nan_to_zero(flow)
	
	# warping using gt flow
	gt_warp = hp.warp(img,gt_flow)

	# warping using predicted flow
	warp = hp.warp(img,flow)
	return find_mean_difference(gt_warp,warp)




# # defined as  sqrt[(u0 − u1)^2 + (v0 − v1)^2]

def endpoint_loss(gt_flow,flow, weight=0.7):

	# if the predicted flow has nan, just replace them with 0
	flow = hp.nan_to_zero(flow)
	return find_mean_difference(gt_flow,flow)


def find_mean_difference(result1,result2):
	answer = result2 - result1
	return tf.reduce_mean(tf.sqrt(tf.square(answer)))



def mse_loss(label,prediction):
  # Calculate the average cross entropy loss across the batch.
  mse = tf.losses.mean_squared_error(label,prediction)
  return mse


def _add_loss_summaries(total_loss):
  """Add summaries for losses in CIFAR-10 model.
  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.
  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  print('jambi')
  print(losses)
  print([total_loss])
  print(losses + [total_loss])

  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.summary.scalar(l.op.name + ' (raw)', l)
    tf.summary.scalar(l.op.name, loss_averages.average(l))

  return loss_averages_op
