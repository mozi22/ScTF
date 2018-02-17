
import helpers as hp

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

