import numpy as np
import tensorflow as tf
from   PIL import Image
import helpers as hpl
import network
import math
import matplotlib as plt
import ijremote as ij
import synthetic_tf_converter as stc
FLAGS = tf.app.flags.FLAGS


tf.app.flags.DEFINE_boolean('SHOW_PREDICTED_DEPTH_CHANGE', False,
                            """Show Depth Images.""")

tf.app.flags.DEFINE_boolean('SHOW_GT_DEPTHS', False,
                            """Show Depth Images Ground Truth.""")

tf.app.flags.DEFINE_boolean('SHOW_GT_DEPTH_CHANGE', False,
                            """Show Depth Images Ground Truth.""")


tf.app.flags.DEFINE_boolean('SHOW_PREDICTED_FLOWS', True,
                            """Show both U and V Flow Values.""")

tf.app.flags.DEFINE_boolean('SHOW_GT_FLOWS', False,
                            """Show both U and V Fl ow Values Ground truths.""")

tf.app.flags.DEFINE_boolean('SHOW_PREDICTED_WARPED_RESULT', True,
                            """Perform warping with predicted flow values.""")

tf.app.flags.DEFINE_boolean('SHOW_GT_WARPED_RESULT', False,
                            """Perform warping with ground truth flow values.""")

tf.app.flags.DEFINE_boolean('SHOW_GT_IMGS', False,
                            """Show the ground truth images.""")

tf.app.flags.DEFINE_boolean('PARSING_PTB', False,
                            """Show the ground truth images.""")

tf.app.flags.DEFINE_boolean('PARSING_MID', True,
                            """Show the ground truth images.""")


tf.app.flags.DEFINE_string('PARENT_FOLDER_MID', '../dataset_synthetic/middlebury/',
                           """The root folder for the dataset """)

tf.app.flags.DEFINE_string('PARENT_FOLDER', '../dataset_synthetic/driving/',
                           """The root folder for the dataset """)

tf.app.flags.DEFINE_string('PARENT_FOLDER_PTB', '../dataset_ptb/ValidationSet/bear_front/',
                           """The root folder for the dataset """)

tf.app.flags.DEFINE_string('IMG1',  'frames_finalpass_webp/35mm_focallength/scene_backwards/fast/left/',
                           """The name of the tower """)

tf.app.flags.DEFINE_string('IMG2',  'frames_finalpass_webp/35mm_focallength/scene_backwards/fast/left/',
                           """The name of the tower """)

tf.app.flags.DEFINE_string('DISPARITY1', 'disparity/35mm_focallength/scene_backwards/fast/left/',
                           """The name of the tower """)

tf.app.flags.DEFINE_string('DISPARITY2', 'disparity/35mm_focallength/scene_backwards/fast/left/',
                           """The name of the tower """)

tf.app.flags.DEFINE_string('FLOW', 'optical_flow/35mm_focallength/scene_backwards/fast/into_future/left/',
                           """The name of the tower """)

tf.app.flags.DEFINE_string('DISPARITY_CHNG', 'disparity_change/35mm_focallength/scene_backwards/fast/into_future/left/',
                           """The name of the tower """)


tf.app.flags.DEFINE_string('CKPT_FOLDER', 'ckpt/driving/epe_all_ds/train/',
                           """The name of the tower """)

if FLAGS.PARSING_PTB == True:

	PATH_RGB = 'rgb/';
	PATH_DEPTH = 'depth/';

	IMG1_NUMBER = '01'
	IMG2_NUMBER = '02'

	FLAGS.IMG1 = FLAGS.PARENT_FOLDER_PTB + PATH_RGB + IMG1_NUMBER + '.png'
	FLAGS.IMG2 = FLAGS.PARENT_FOLDER_PTB + PATH_RGB + IMG2_NUMBER + '.png'
	FLAGS.DISPARITY1 = FLAGS.PARENT_FOLDER_PTB + PATH_DEPTH + IMG1_NUMBER + '.png'
	FLAGS.DISPARITY2 = FLAGS.PARENT_FOLDER_PTB + PATH_DEPTH + IMG2_NUMBER + '.png'

elif FLAGS.PARSING_MID == True:

	DATASETS = ['middlebury2003/','middlebury2005/']

		#          0        1			2_05	3		4		5			6			7
	folders = ['conesF/','teddyF/','Art/','Books/','Dolls/','Laundry/','Moebius/','Reindeer/']

	folder_num = 0

	if folder_num > 1:
		dataset_num = 1
		img1_path = 'view1.png'
		img2_path = 'view5.png'
		disp1_path = 'disp1.png'
		disp2_path = 'disp5.png'
	else:
		dataset_num = 0
		img1_path = 'im2.ppm'
		img2_path = 'im6.ppm'
		disp1_path = 'disp2.pgm'
		disp2_path = 'disp6.pgm'


	FLAGS.IMG1 = FLAGS.PARENT_FOLDER_MID + DATASETS[dataset_num] + folders[folder_num] +  img1_path
	FLAGS.IMG2 = FLAGS.PARENT_FOLDER_MID + DATASETS[dataset_num] + folders[folder_num] +  img2_path
	FLAGS.DISPARITY1 = FLAGS.PARENT_FOLDER_MID + DATASETS[dataset_num] + folders[folder_num] +  disp1_path
	FLAGS.DISPARITY2 = FLAGS.PARENT_FOLDER_MID + DATASETS[dataset_num] + folders[folder_num] +  disp2_path

else:
	IMG1_NUMBER = '0001'
	IMG2_NUMBER = '0002'


	FLAGS.IMG1 = FLAGS.PARENT_FOLDER + FLAGS.IMG1 + IMG1_NUMBER + '.webp'
	FLAGS.IMG2 = FLAGS.PARENT_FOLDER + FLAGS.IMG2 + IMG2_NUMBER + '.webp'
	FLAGS.DISPARITY1 = FLAGS.PARENT_FOLDER + FLAGS.DISPARITY1 + IMG1_NUMBER + '.pfm'
	FLAGS.DISPARITY2 = FLAGS.PARENT_FOLDER + FLAGS.DISPARITY2 + IMG2_NUMBER + '.pfm'
	FLAGS.DISPARITY_CHNG = FLAGS.PARENT_FOLDER + FLAGS.DISPARITY_CHNG + IMG1_NUMBER + '.pfm'
	FLAGS.FLOW = FLAGS.PARENT_FOLDER + FLAGS.FLOW + 'OpticalFlowIntoFuture_' + IMG1_NUMBER + '_L.pfm'




class FlowPredictor:

	def get_depth_from_disp(self,disparity):
		focal_length = 1050.0
		disp_to_depth = disparity / focal_length
		return disp_to_depth

	def normalizeOptFlow(self,flow,input_size):
		# remove the values bigger than the image size
		flow[:,:,0][flow[:,:,0] > input_size[0] ] = 0 # 384
		flow[:,:,1][flow[:,:,1] > input_size[1] ] = 0 # 224

		# separate the u and v values 
		flow_u = flow[:,:,0]
		flow_v = flow[:,:,1]
		# np.savetxt('non_normal.txt',flow_u)

		# Image.fromarray(flow[:,:,0]).show()
		# Image.fromarray(flow[:,:,1]).show()

		# opt_u = np.squeeze(flow_u).astype(np.uint8)
		# opt_v = np.squeeze(flow_v).astype(np.uint8)

		# result = np.dstack((flow_u,flow_v))
		# opt_u = Image.fromarray(result[:,:,0]).show() 
		# opt_v = Image.fromarray(result[:,:,1]).show()



		# normalize the values by the image dimensions
		flow_u = flow_u / input_size[0]
		flow_v = flow_v / input_size[1]



		# combine them back and return
		return np.dstack((flow_u,flow_v))

	def preprocess_mid(self):

		self.input_size = (256, 160)

		img1 = Image.open(FLAGS.IMG1)
		img2 = Image.open(FLAGS.IMG2)
		
		disp1 = Image.open(FLAGS.DISPARITY1)
		disp2 = Image.open(FLAGS.DISPARITY2)

		disp1 = disp1.resize(self.input_size, Image.NEAREST)
		disp2 = disp2.resize(self.input_size, Image.NEAREST)

		depth1 = self.get_depth_from_disp(np.array(disp1))
		depth2 = self.get_depth_from_disp(np.array(disp2))


		self.init_img1 = img1.resize(self.input_size, Image.BILINEAR)
		self.init_img2 = img2.resize(self.input_size, Image.BILINEAR)

		self.img1_arr = np.array(self.init_img1)
		self.img2_arr = np.array(self.init_img2)


		self.img1 = self.img1_arr / 255
		self.img2 = self.img2_arr / 255

		max_val = np.max(depth1)
		self.depth1 = depth1 / max_val
		self.depth2 = depth2 / max_val

		rgbd1 = self.combine_depth_values(self.img1,self.depth1)
		rgbd2 = self.combine_depth_values(self.img2,self.depth2)

		# # combine images to 8 channel rgbd-rgbd
		# img_pair = np.concatenate((self.img1,self.img2),axis=2)
		self.img_pair = np.concatenate((rgbd1,rgbd2),axis=2)


		disp1 = np.array(disp1,dtype=np.float32)
		flow_expanded_u = np.expand_dims(disp1,axis=2) 
		flow_expanded_v = np.expand_dims(np.zeros_like(disp1),axis=2)
		self.optical_floww = np.concatenate([flow_expanded_u,flow_expanded_v],axis=-1)
		# self.optical_floww = self.normalizeOptFlow(self.optical_floww,self.input_size)

		self.img_pair = np.expand_dims(self.img_pair,0)
		self.initialize_network()

		self.sess = tf.InteractiveSession()
		self.load_model_ckpt(self.sess,FLAGS.CKPT_FOLDER)

	# img1: path of img1
	# img2: path of img2
	# depth1: path of depth pfm 1
	# depth2: path of depth pfm 2
	def preprocess(self):

		self.input_size = 256,160


		self.u_factor_1 = 0.714285714
		self.v_factor_1 = 0.666666667

		self.u_factor_2 = 0.414814815
		self.v_factor_2 = 0.4


		result = stc.convert_for_testing().from_paths_to_data(FLAGS.DISPARITY1,
							   FLAGS.DISPARITY2,
							   FLAGS.DISPARITY_CHNG,
							   FLAGS.FLOW,
							   FLAGS.IMG1,
							   FLAGS.IMG2,
							   'L')

		img1 = Image.fromarray(result[0]['web_p'],'RGB')
		img2 = Image.fromarray(result[0]['web_p2'],'RGB')

		depth1 = Image.fromarray(result[0]['depth1'],mode='F')
		depth2 = Image.fromarray(result[0]['depth2'],mode='F')

		self.init_img1 = img1.resize(self.input_size,Image.NEAREST)
		self.init_img2 = img2.resize(self.input_size,Image.NEAREST)

		depth1 = depth1.resize(self.input_size,Image.NEAREST)
		depth2 = depth2.resize(self.input_size,Image.NEAREST)


		depth1 = np.array(depth1)
		depth2 = np.array(depth2)

		self.img1_arr = np.array(self.init_img1,dtype=np.float32)[:,:,0:3]
		self.img2_arr = np.array(self.init_img2,dtype=np.float32)[:,:,0:3]

		self.depth1 = depth1 / np.max(depth1)
		self.depth2 = depth2 / np.max(depth1)

		# normalize images
		self.img1 = self.img1_arr / 255
		self.img2 = self.img2_arr / 255


		rgbd1 = self.combine_depth_values(self.img1,self.depth1)
		rgbd2 = self.combine_depth_values(self.img2,self.depth2)

		# # combine images to 8 channel rgbd-rgbd
		# img_pair = np.concatenate((self.img1,self.img2),axis=2)
		self.img_pair = np.concatenate((rgbd1,rgbd2),axis=2)

		# # add padding to axis=0 to make the input image (224,384,8)
		# self.img_pair = np.pad(self.img_pair,((4,4),(0,0),(0,0)),'constant')

		# # change dimension from (224,384,8) to (1,224,384,8)
		self.img_pair = np.expand_dims(self.img_pair,0)
		self.initialize_network()

		self.sess = tf.InteractiveSession()
		self.load_model_ckpt(self.sess,FLAGS.CKPT_FOLDER)


	def preprocess_ptb(self):

		self.input_size = 256, 160

		img1 = Image.open(FLAGS.IMG1)
		img2 = Image.open(FLAGS.IMG2)
		
		depth1 = Image.open(FLAGS.DISPARITY1)
		depth2 = Image.open(FLAGS.DISPARITY2)

		self.init_img1 = img1.resize(self.input_size, Image.BILINEAR)
		self.init_img2 = img2.resize(self.input_size, Image.BILINEAR)

		depth1 = depth1.resize(self.input_size, Image.NEAREST)
		depth2 = depth2.resize(self.input_size, Image.NEAREST)


		self.img1_arr = np.array(self.init_img1)
		self.img2_arr = np.array(self.init_img2)

		depth1 = np.array(depth1)
		depth2 = np.array(depth2)

		img1 = self.img1_arr / 255
		img2 = self.img2_arr / 255

		depth1 = depth1 / np.max(depth1)
		depth2 = depth2 / np.max(depth1)

		rgbd1 = self.combine_depth_values(img1,depth1)
		rgbd2 = self.combine_depth_values(img2,depth2)


		self.img_pair = np.concatenate((rgbd1,rgbd2),axis=2)

		# # add padding to axis=0 to make the input image (224,384,8)
		# self.img_pair = np.pad(self.img_pair,((4,4),(0,0),(0,0)),'constant')

		# # change dimension from (224,384,8) to (1,224,384,8)
		self.img_pair = np.expand_dims(self.img_pair,0)
		self.initialize_network()

		self.sess = tf.InteractiveSession()
		self.load_model_ckpt(self.sess,FLAGS.CKPT_FOLDER)

	def read_gt(self,opt_flow,disp_chng):
		opt_flow = hpl.readPFM(opt_flow)[0]
		disp_chng = hpl.readPFM(disp_chng)[0]

		disp_chng = Image.fromarray(disp_chng)

		_ ,_ , resized_inv_depth = self.get_resized_inverse_depth(None,None,disp_chng,self.input_size)


		opt_flow = self.downsample_opt_flow(opt_flow,self.input_size)

		opt_flow_u = opt_flow[:,:,0] * self.u_factor_1
		opt_flow_v = opt_flow[:,:,1] * self.v_factor_1

		opt_flow_u = opt_flow[:,:,0] * self.u_factor_2
		opt_flow_v = opt_flow[:,:,1] * self.v_factor_2

		return np.stack((opt_flow_u,opt_flow_v),axis=2), resized_inv_depth

	# send in the disparity values, this will return the normalized inverse depth values.
	def get_resized_inverse_depth(self,disparity,disparity2,disparity_change,input_size):


		depth1 = None
		depth2 = None
		depth_change = None

		if disparity != None:
		
			disparity = disparity.resize(input_size,Image.NEAREST)
			disparity2 = disparity2.resize(input_size,Image.NEAREST)
	
			disparity = np.array(disparity)
			disparity2 = np.array(disparity2)

			disparity = disparity * 0.4
			disparity2 = disparity2 * 0.4

			# convert disparities to depth
			depth1 = self.get_depth_from_disp(disparity)
			depth2 = self.get_depth_from_disp(disparity2)

			# get inverse depth
			depth1 = 1 / depth1
			depth2 = 1 / depth2

		elif disparity_change != None:


			disparity_change = disparity_change.resize(input_size,Image.NEAREST)

			disparity_change = np.array(disparity_change)
	
			disparity_change = disparity_change * 0.4

			# there are 0 values in disparity_change. We can add an epsilon to to shift the matrix.
			# disparity_change = disparity_change + 1e-6

			depth_change = self.get_depth_from_disp(disparity_change)

			depth_change = 1 / depth_change


		return depth1,depth2,depth_change


	def get_depth_chng_from_disp_chng(self,disparity,disparity_change):
		disp2 = disparity + disparity_change

		depth1 = self.get_depth_from_disp(disparity)
		depth2 = self.get_depth_from_disp(disp2)

		return depth1 - depth2

	def warp(self,img,flow):

		img = img.astype(np.float32)

		x = list(range(0,self.input_size[0]))
		y = list(range(0,self.input_size[1]))
		X, Y = tf.meshgrid(x, y)

		X = tf.cast(X,np.float32) + flow[:,:,0]
		Y = tf.cast(Y,np.float32) + flow[:,:,1]

		con = tf.stack([X,Y])
		result = tf.transpose(con,[1,2,0])
		result = tf.expand_dims(result,0)
		return tf.contrib.resampler.resampler(img[np.newaxis,:,:,:],result)

	def show_image(self,array,img_title):
		# shaper = array.shape
		a = Image.fromarray(array)
		# a = a.resize((math.ceil(shaper[1] * 2),math.ceil(shaper[0] * 2)), Image.BILINEAR)
		a.show(title=img_title)
		# a.save('prediction_without_pc_loss.jpg')

	def denormalize_flow(self,flow):

		flow = np.squeeze(flow)
		u = flow[:,:,0] * self.input_size[0]
		v = flow[:,:,1] * self.input_size[1]
		# w = flow[:,:,2] * self.max_depth_driving_chng


		flow = np.stack((u,v),axis=2)
		
		return flow


	def predict(self):

		if self.lossee is None:
			feed_dict = {
				self.X: self.img_pair
			}
			v = self.sess.run([self.predict_flow2],feed_dict=feed_dict)
		else:
			feed_dict = {
				self.X: self.img_pair,
				self.Y: np.expand_dims(self.optical_floww,axis=0)
			}
			v, self.lossee = self.sess.run([self.predict_flow2,self.lossee],feed_dict=feed_dict)

		return self.denormalize_flow(v), self.lossee

	def get_depth_from_disp(self,disparity):
		disparity = disparity + 1e-6

		focal_length = 1050
		disp_to_depth = focal_length / disparity
		return disp_to_depth



	def downsample_opt_flow(self,data,size):
		data = np.delete(data,2,axis=2)

		u = data[:,:,0]
		v = data[:,:,1]
		
		dt = Image.fromarray(u,mode='F')
		dt = dt.resize(size, Image.NEAREST)

		dt2 = Image.fromarray(v,mode='F')
		dt2 = dt2.resize(size, Image.NEAREST)
		u = np.array(dt)
		v = np.array(dt2)

		return np.stack((u,v),axis=2)

	def combine_depth_values(self,img,depth):
		depth = np.expand_dims(depth,2)
		return np.concatenate((img,depth),axis=2)

	def initialize_network(self,lbl=False):

		self.batch_size = 1

		self.X = tf.placeholder(dtype=tf.float32, shape=(self.batch_size, 160, 256, 8))
		self.Y = tf.placeholder(dtype=tf.float32, shape=(self.batch_size, 160, 256, 2))

		self.predict_flow2 = network.train_network(self.X)

		self.predict_flow2 = self.predict_flow2[0]

		if FLAGS.PARSING_MID == True:
			self.lossee = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.Y, self.predict_flow2))))
		else:
			self.lossee = None

		# self.predict_flow2 = self.predict_flow2

	def load_model_ckpt(self,sess,filename):
		saver = tf.train.Saver()
		saver.restore(sess, tf.train.latest_checkpoint(filename))

####### work part ######
predictor = FlowPredictor()

if FLAGS.PARSING_PTB == True:
	predictor.preprocess_ptb()
elif FLAGS.PARSING_MID == True:
	predictor.preprocess_mid()
	gt_flow = predictor.optical_floww
else:
	predictor.preprocess()
	gt_flow, gt_depth_change = predictor.read_gt(FLAGS.FLOW,FLAGS.DISPARITY_CHNG)


pr_flow, loss = predictor.predict()
print(pr_flow.shape)

if not loss is None:
	print(loss)

if FLAGS.PARSING_PTB == False:
	# show gt flows
	if FLAGS.SHOW_GT_FLOWS == True:
		ij.setImage('gt_flow_uv',np.transpose(gt_flow,[2,0,1]))

	# warp with gt predited flow values
	if FLAGS.SHOW_GT_WARPED_RESULT == True:
		# gt_flow = np.pad(gt_flow,((4,4),(0,0),(0,0)),'constant')

		flow = predictor.warp(predictor.img2_arr,gt_flow)
		result = flow.eval()[0].astype(np.uint8)
		predictor.show_image(result,'warped_img_gt')

	# show inv depth values for both images
	if FLAGS.SHOW_GT_DEPTHS == True:
		ij.setImage('gt_inv_depth1',predictor.inv_depth1)
		ij.setImage('gt_inv_depth2',predictor.inv_depth2)


	# show inv depth values for both images
	if FLAGS.SHOW_GT_DEPTH_CHANGE == True:
		ij.setImage('gt_depth_change',gt_depth_change)


# show gt images
if FLAGS.SHOW_GT_IMGS == True:
	predictor.init_img1.show()
	predictor.init_img2.show()

# show predicted depth change
if FLAGS.SHOW_PREDICTED_DEPTH_CHANGE == True:
	ij.setImage('predicted_depth_change',pr_depth_change)

# show predicted flow values
if FLAGS.SHOW_PREDICTED_FLOWS == True:
	ij.setImage('predicted_uv',np.transpose(pr_flow,[2,0,1]))

# show warped result with predicted flow values
if FLAGS.SHOW_PREDICTED_WARPED_RESULT == True:
	flow = predictor.warp(predictor.img2_arr,pr_flow)
	result = flow.eval()[0].astype(np.uint8)
	predictor.show_image(result,'warped_img_pr')