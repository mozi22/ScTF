import numpy as np
import tensorflow as tf
from   PIL import Image
import helpers as hpl
import network
import matplotlib.pyplot as plt
import ijremote as ij

class FlowPredictor:

	# img1: path of img1
	# img2: path of img2
	# depth1: path of depth pfm 1
	# depth2: path of depth pfm 2
	def preprocess(self,img1,img2,disparity1,disparity2):


		factor = 0.4
		self.input_size = int(960 * factor), int(540 * factor)
		self.driving_disp_chng_max = 236.467
		self.driving_disp_max = 349.347 

		# read resized images to network standards
		self.init_img1, self.init_img2 = self.read_image(img1,img2)

		self.img1_arr = np.array(self.init_img1,dtype=np.float32)
		self.img2_arr = np.array(self.init_img2,dtype=np.float32)

		# normalize images
		self.img1 = self.img1_arr / 255
		self.img2 = self.img2_arr / 255

		# read disparity values from matrices
		disp1 = hpl.readPFM(disparity1)[0]
		disp2 = hpl.readPFM(disparity2)[0]

		disp1 = Image.fromarray(disp1)
		disp2 = Image.fromarray(disp2)
		# resize disparity values
		disp1 = disp1.resize(self.input_size,Image.NEAREST)
		disp2 = disp2.resize(self.input_size,Image.NEAREST)

		# resize disp values

		disp1 = np.array(disp1)
		disp2 = np.array(disp2)

		# normalize disp values
		self.disp1 = disp1 / self.driving_disp_max
		self.disp2 = disp2 / self.driving_disp_max

		# combine depth values with images
		rgbd1 = self.combine_depth_values(self.img1,self.disp1)
		rgbd2 = self.combine_depth_values(self.img2,self.disp2)

		# combine images to 8 channel rgbd-rgbd
		# img_pair = np.concatenate((rgbd1,rgbd2),axis=2)
		img_pair = np.concatenate((self.img1,self.img2),axis=2)

		# add padding to axis=0 to make the input image (224,384,8)
		img_pair = np.pad(img_pair,((4,4),(0,0),(0,0)),'constant')

		# change dimension from (224,384,8) to (1,224,384,8)
		self.img_pair = np.expand_dims(img_pair,0)
		self.initialize_network()

		self.sess = tf.InteractiveSession()
		# # self.load_model_ckpt(self.sess,'ckpt/driving/depth/train/model_ckpt_15000.ckpt')
		# self.load_model_ckpt(self.sess,'ckpt/driving/conv10/train/model_ckpt_24300.ckpt')
		self.load_model_ckpt(self.sess,'ckpt/driving/multi_gpu_epe_loss/')


	def read_gt(self,opt_flow,disp_chng):
		opt_flow = hpl.readPFM(opt_flow)[0]
		disp_chng = hpl.readPFM(disp_chng)[0]

		disp_chng = Image.fromarray(disp_chng)
		disp_chng = disp_chng.resize(self.input_size,Image.NEAREST)

		opt_flow = self.downsample_opt_flow(opt_flow,self.input_size)

		# z = np.zeros((opt_flow.shape[0],opt_flow.shape[1]))

		# opt_flow = self.combine_depth_values(opt_flow,z)
		# Image.fromarray(opt_flow,'RGB').show()

		# final_label = self.combine_depth_values(opt_flow,disp_chng)
		return opt_flow * 0.4


	def get_depth_chng_from_disp_chng(self,disparity,disparity_change):
		disp2 = disparity + disparity_change

		depth1 = self.get_depth_from_disp(disparity)
		depth2 = self.get_depth_from_disp(disp2)

		return depth1 - depth2


	def warp(self,img,flow):
		x = list(range(0,self.input_size[0]))
		y = list(range(0,self.input_size[1] + 8))
		X, Y = tf.meshgrid(x, y)

		X = tf.cast(X,np.float32) + flow[:,:,0]
		Y = tf.cast(Y,np.float32) + flow[:,:,1]



		con = tf.stack([X,Y])
		result = tf.transpose(con,[1,2,0])
		result = tf.expand_dims(result,0)
		return tf.contrib.resampler.resampler(img[np.newaxis,:,:,:],result)


	def show_image(self,array,img_title):
		a = Image.fromarray(array)
		a.show(title=img_title)



	def denormalize_flow(self,flow,show_flow):

		opt_u = flow[:,:,0]
		opt_v = flow[:,:,1]
		# spacing = np.linspace(0, 1, num=100)

		# plt.hist(opt_u.flatten(),bins=spacing)  # arguments are passed to np.histogram
		# plt.hist(opt_v.flatten(),bins=spacing)  # arguments are passed to np.histogram
		# plt.title("Flow predict")
		# plt.show()


		u = flow[:,:,0] * self.input_size[0]
		v = flow[:,:,1] * self.input_size[1]
		# w = flow[:,:,2] * self.driving_disp_chng_max

		# w = self.get_depth_from_disp(w)

		# if show_flow:
		# 	self.show_image(u,'Flow_u')
		# 	self.show_image(v,'Flow_v')
			# self.show_image(w,'Flow_w')

		Image.fromarray(u).save('predictflow_u.tiff')
		Image.fromarray(v).save('predictflow_v.tiff')
		
		flow = np.stack((u,v),axis=2)
		print('JAZZY MA')
		print(flow.shape)
		
		# not being used currently.
		# flow_with_depth = np.stack((u,v,w),axis=2)
		


		return flow


	def postprocess(self,flow,show_flow=True,gt=False):

		if gt==True:
			print('working')
			# self.show_image(flow[:,:,0],'Flow_u')
			# self.show_image(flow[:,:,1],'Flow_v')
			# Image.fromarray(flow[:,:,0]).save('originalflow_u.tiff')
			# Image.fromarray(flow[:,:,1]).save('originalflow_v.tiff')
			# self.show_image(flow[:,:,2],'Flow_w')
		else:
			flow = self.denormalize_flow(flow,show_flow)

		# ij.setImage('PredictedFlow_u',flow[:,:,0])
		# ij.setImage('PredictedFlow_v',flow[:,:,1])
		self.img2_arr = np.pad(self.img2_arr,((4,4),(0,0),(0,0)),'constant')
		flow = self.warp(self.img2_arr,flow)

		result = flow.eval()[0].astype(np.uint8)
		self.show_image(result,'warped_img')

		# plt.hist(result, bins='auto')  # arguments are passed to np.histogram
		# plt.title("Histogram with 'auto' bins")
		# plt.show()
		# self.init_img1.show(title='img1')
		# self.init_img2.show(title='img2')



	def predict(self):
		feed_dict = {
			self.X: self.img_pair,
		}

		v = self.sess.run({'prediction': self.predict_flow2},feed_dict=feed_dict)
		self.postprocess(v['prediction'][0])

	def get_depth_from_disp(self,disparity):
		focal_length = 35
		disp_to_depth = focal_length / disparity
		return disp_to_depth


	def read_image(self,img1,img2):

		img1 = Image.open(img1)
		img2 = Image.open(img2)

		img1 = img1.resize(self.input_size, Image.BILINEAR)
		img2 = img2.resize(self.input_size, Image.BILINEAR)

		return img1, img2

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

	def initialize_network(self):

		self.batch_size = 1

		self.X = tf.placeholder(dtype=tf.float32, shape=(self.batch_size, 224, 384, 6))
		self.Y = tf.placeholder(dtype=tf.float32, shape=(self.batch_size, 224, 384, 2))
		self.predict_flow5, self.predict_flow2 = network.train_network(self.X)

	def load_model_ckpt(self,sess,filename):
		saver = tf.train.Saver()
		saver.restore(sess, tf.train.latest_checkpoint(filename))

