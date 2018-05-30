import network
import synthetic_tf_converter as stc
import tensorflow as tf
import numpy as np
import math
from PIL import Image 

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('CKPT_FOLDER', 'ckpt/driving/epe/train/',
                           """The name of the tower """)




u_factor = 0.414814815
v_factor = 0.4

input_size = math.ceil(960 * v_factor), math.floor(540 * u_factor)


def get_depth_from_disp(disparity):
	focal_length = 1050.0
	disp_to_depth = disparity / focal_length
	return disp_to_depth

def combine_depth_values(img,depth):
	depth = np.expand_dims(depth,2)
	return np.concatenate((img,depth),axis=2)

def parse_input(img1,img2,disp1,disp2):
	img1 = Image.open(img1)
	img2 = Image.open(img2)

	disp1 = Image.open(disp1)
	disp2 = Image.open(disp2)

	img1 = img1.resize(input_size, Image.BILINEAR)
	img2 = img2.resize(input_size, Image.BILINEAR)

	disp1 = disp1.resize(input_size, Image.NEAREST)
	disp2 = disp2.resize(input_size, Image.NEAREST)

	disp1 = np.array(disp1)
	disp2 = np.array(disp2)

	depth1 = get_depth_from_disp(disp1)
	depth2 = get_depth_from_disp(disp2)

	# normalize
	depth1 = depth1 / np.max(depth1)
	depth2 = depth2 / np.max(depth1)

	img1 = np.array(img1) / 255
	img2 = np.array(img2) / 255

	rgbd1 = combine_depth_values(img1,depth1)
	rgbd2 = combine_depth_values(img2,depth2)

	img_pair = np.concatenate((rgbd1,rgbd2),axis=2)

	return img_pair

def load_model_ckpt(sess,filename):
	saver = tf.train.Saver()
	saver.restore(sess, tf.train.latest_checkpoint(filename))



def predict(img_pair):

	img_pair = np.expand_dims(img_pair,axis=0)

	feed_dict = {
		X: img_pair,
	}

	v = sess.run({'prediction': predict_flow2},feed_dict=feed_dict)

	return denormalize_flow(v['prediction'][0])

def denormalize_flow(flow):

	u = flow[:,:,0] * input_size[0]
	v = flow[:,:,1] * input_size[1]
	# w = flow[:,:,2] * self.max_depth_driving_chng
	
	flow = np.stack((u,v),axis=2)
	
	return flow, 'w'

def perform_testing():
	dataset_root = '../dataset_synthetic/middlebury/'
	dataset_type = ['middlebury2003','middlebury2005']

	for typee in dataset_type:

		ds_current_root = dataset_root + typee

		if typee == dataset_type[0]:
			print('parsing middlebury 2003 ... ')
			folders = ['conesF','teddyF']

			img1_path = 'im2.ppm'
			img2_path = 'im6.ppm'
			disp1_path = 'disp2.pgm'
			disp2_path = 'disp6.pgm'

		else:
			print('parsing middlebury 2005 ... ')
			folders = ['Art','Books','Dolls','Laundry','Moebius','Reindeer']

			img1_path = 'view1.png'
			img2_path = 'view5.png'
			disp1_path = 'disp1.png'
			disp2_path = 'disp5.png'


		for folder in folders:
			final_path = ''
			final_path = ds_current_root + '/' +folder + '/'

			img1_path_final = final_path + img1_path
			img2_path_final = final_path + img2_path
			disp1_path_final = final_path + disp1_path
			disp2_path_final = final_path + disp2_path

			print('')
			print('folder = '+ folder)
			print('')

			img_pair = parse_input(img1_path_final,img2_path_final,disp1_path_final,disp2_path_final)
			result = predict(img_pair)
			print(result)

sess = tf.InteractiveSession()
X = tf.placeholder(dtype=tf.float32, shape=(1, 224, 384, 8))
predict_flow2 = network.train_network(X)
load_model_ckpt(sess,FLAGS.CKPT_FOLDER)
predict_flow2 = predict_flow2[0]

perform_testing()