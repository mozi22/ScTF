import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.python.client import device_lib
import os
import sys
import json
import glob

import network
import import_data
import helpers
import losses


# # read data
# img1 = Image.open(os.path.join(examples_dir,'sculpture1.png'))
# img2 = Image.open(os.path.join(examples_dir,'sculpture2.png'))

# input_data = prepare_input_data(img1,img2,data_format)

# print(get_available_gpus())



        

batch_size = 32
top_output = ('IMAGE_PAIR', 'MOTION', 'DEPTH', 'INTRINSICS')
reader_params = {
	'batch_size': batch_size,
	'test_phase': False,
	'motion_format': 'ANGLEAXIS6',
	'inverse_depth': True,
	'builder_threads': 1,
	'scaled_width': 256,
	'scaled_height': 192,
	'norm_trans_scale_depth': True,        
	'top_output': top_output,
	'scene_pool_size': 650,
	'builder_threads': 8,
}

_data_dir = './demon/datasets/traindata'
reader_params = datareader.add_sources(reader_params, glob.glob(os.path.join(_data_dir,'sun3d_train_0.01m_to_0.1m.h5')), 0.8)


with tf.name_scope("datareader"):


    reader_tensors = datareader.multi_vi_h5_data_reader(len(top_output), json.dumps(reader_params))
    data_tensors = reader_tensors[2]
    data_dict_all = dict(zip(top_output, data_tensors))
    num_test_iterations, current_batch_buffer, max_batch_buffer, current_read_buffer, max_read_buffer = tf.unstack(reader_tensors[0])
    tf.summary.scalar("datareader/batch_buffer",current_batch_buffer)
    tf.summary.scalar("datareader/read_buffer",current_read_buffer)

    # split the data for the individual towers
    data_dict_split = {}
    for k,v in data_dict_all.items():
        if k == 'INFO':
            continue # skip info vector
        if _num_gpus > 1:
            tmp = tf.split(v, num_or_size_splits=_num_gpus)
        else:
            tmp = [v]
        data_dict_split[k] = tmp


for gpu_id in range(_num_gpus):
	with tf.device('/cpu:{0}'.format(gpu_id)), tf.name_scope('tower_{0}'.format(gpu_id)) as tower:
		reuse = gpu_id != 0

		data_dict = {}
		for k,v in data_dict_split.items():
			data_dict[k] = v[gpu_id]

		# dict of the losses of the current tower
		loss_dict = {}

		# data preprocessing
		with tf.name_scope("data_preprocess"):
			rotation, translation = tf.split(value=data_dict['MOTION'], num_or_size_splits=2, axis=1)
			ground_truth = prepare_ground_truth_tensors(
				data_dict['DEPTH'],
				rotation,
				translation,
				data_dict['INTRINSICS'],
			)
			image1, image2 = tf.split(value=data_dict['IMAGE_PAIR'], num_or_size_splits=2, axis=1)
			image2_2 = tf.transpose(tf.image.resize_area(tf.transpose(image2,perm=[0,2,3,1]), (48,64)), perm=[0,3,1,2])
			if trainer.current_evo >= '5_refine':
				data_dict['image1'] = image1
			data_dict['image2_2'] = image2_2
			ground_truth['rotation'] = rotation
			ground_truth['translation'] = translation

		#
		# netFlow1
		#
		with tf.variable_scope('netFlow1',reuse=True):
			netFlow1_result = flow_block(image_pair=data_dict['IMAGE_PAIR'], kernel_regularizer=_kernel_regularizer)
			predict_flow5, predict_conf5 = tf.split(value=netFlow1_result['predict_flowconf5'], num_or_size_splits=2, axis=1)
			predict_flow2, predict_conf2 = tf.split(value=netFlow1_result['predict_flowconf2'], num_or_size_splits=2, axis=1)

		with tf.name_scope('netFlow1_losses'):
		    # slowly increase the weights for the scale invariant gradient losses
		    flow_sig_weight = ease_out_quad(global_stepf, 0, _flow_grad_loss_weight, float(max_iter//3))
		    conf_sig_weight = ease_out_quad(global_stepf, 0, _flow_conf_grad_loss_weight, float(max_iter//3))
		    # slowly decrase the importance of the losses on the smaller resolution
		    level5_factor = ease_in_quad(global_stepf, 1, -1, float(max_iter//3))


		    losses = flow_loss_block(
		        gt_flow2=ground_truth['flow2'], 
		        gt_flow5=ground_truth['flow5'], 
		        gt_flow2_sig=ground_truth['flow2_sig'], 
		        pr_flow2=predict_flow2, 
		        pr_flow5=predict_flow5, 
		        pr_conf2=predict_conf2, 
		        pr_conf5=predict_conf5, 
		        flow_weight=_flow_loss_weight, 
		        conf_weight=_flow_conf_loss_weight, 
		        flow_sig_weight=flow_sig_weight,
		        conf_sig_weight=conf_sig_weight,
		        conf_diff_scale=10,
		        level5_factor=level5_factor,
		        loss_prefix='netFlow1_'
		        )
		    loss_dict.update(losses) # add to the loss dict of the current tower

		    # add selected losses to the 'losses' collection. the remaining losses are only used for summaries.
		    selected_losses = ('loss_flow5', 'loss_flow2', 'loss_flow2_sig', 'loss_conf5', 'loss_conf2', 'loss_conf2_sig')
		    for l in selected_losses:
		        tf.losses.add_loss(losses['netFlow1_'+l])