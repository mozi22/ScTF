import numpy as np
import tensorflow as tf
from   PIL import Image
import helpers as hpl
import network
import math
import matplotlib as plt
# import ijremote as ij

class FlowPredictor:

	# img1: path of img1
	# img2: path of img2
	# depth1: path of depth pfm 1
	# depth2: path of depth pfm 2
	def preprocess(self,img1,img2,disparity1,disparity2):

		factor = 0.4
		self.input_size = int(960 * factor), int(540 * factor)
		# self.driving_disp_chng_max = 7.5552e+08
		# self.driving_disp_max = 30.7278
		self.max_depth_driving = 9.98134
		self.max_depth_driving_chng = 6.75619

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

		self.depth1 = self.get_depth_from_disp(disp1)
		self.depth2 = self.get_depth_from_disp(disp2)

		self.depth1 = 1 / self.depth1
		self.depth2 = 1 / self.depth2

		# normalize disp values
		self.depth1 = self.depth1 / self.max_depth_driving
		self.depth2 = self.depth2 / self.max_depth_driving

		# combine depth values with images
		rgbd1 = self.combine_depth_values(self.img1,self.depth1)
		rgbd2 = self.combine_depth_values(self.img2,self.depth2)

		# ij.setImage('imag1',self.depth1)
		# ij.setImage('imag2',self.depth2)

		# d1 = np.expand_dims(self.depth1,axis=2)
		# d2 = np.expand_dims(self.depth2,axis=2)
		# img_pair = np.concatenate((d1,d2),axis=2)
		# # combine images to 8 channel rgbd-rgbd
		img_pair = np.concatenate((rgbd1,rgbd2),axis=2)
		# img_pair = np.concatenate((self.img1,self.img2),axis=2)

		# # add padding to axis=0 to make the input image (224,384,8)
		self.img_pair = np.pad(img_pair,((4,4),(0,0),(0,0)),'constant')

		# # change dimension from (224,384,8) to (1,224,384,8)
		self.img_pair = np.expand_dims(self.img_pair,0)
		self.initialize_network()

		self.sess = tf.InteractiveSession()
		# # self.load_model_ckpt(self.sess,'ckpt/driving/depth/train/model_ckpt_15000.ckpt')
		# self.load_model_ckpt(self.sess,'ckpt/driving/conv10/train/model_ckpt_24300.ckpt')
		self.load_model_ckpt(self.sess,'ckpt/driving/corr_net/')


	def read_gt(self,opt_flow,disp_chng):
		opt_flow = hpl.readPFM(opt_flow)[0]
		disp_chng = hpl.readPFM(disp_chng)[0]

		disp_chng = Image.fromarray(disp_chng)
		disp_chng = disp_chng.resize(self.input_size,Image.NEAREST)

		opt_flow = self.downsample_opt_flow(opt_flow,(160,80))


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
		x = list(range(0,160))
		y = list(range(0,80))
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

	def denormalize_flow(self,flow,show_flow):

		u = flow[:,:,0] * self.input_size[0]
		v = flow[:,:,1] * self.input_size[1]
		w = flow[:,:,2] * self.max_depth_driving_chng
		# w = 1 / w

		# if show_flow:
		self.show_image(u,'Flow_u')
		self.show_image(v,'Flow_v')
			# ij.setImage('PredictedFlow_w',w)

		Image.fromarray(u).save('predictflow_u.tiff')
		Image.fromarray(v).save('predictflow_v.tiff')
		
		flow = np.stack((u,v),axis=2)
		
		# not being used currently.
		# flow_with_depth = np.stack((u,v,w),axis=2)
		return flow


	def postprocess(self,flow,show_flow=True,gt=False):

		print(flow.shape)

		if gt==True:
			self.show_image(flow[:,:,0],'Flow_u')
			self.show_image(flow[:,:,1],'Flow_v')
			self.show_image(flow[:,:,2],'Flow_w')
		else:
			flow = self.denormalize_flow(flow,show_flow)




		# ij.setImage('PredictedFlow_u',flow[:,:,0])
		# ij.setImage('PredictedFlow_v',flow[:,:,1])

		self.img2_arr = np.pad(self.img2_arr,((4,4),(0,0),(0,0)),'constant')
		self.img2_arr = self.img2_arr.astype(np.uint8)
		self.img2_arr = Image.fromarray(self.img2_arr[:,:,0:3],'RGB')

		self.img2_arr = self.img2_arr.resize((160,80), Image.BILINEAR)


		self.img2_arr = np.array(self.img2_arr,dtype=np.float32)


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
			self.X1: self.img_pair[:,:,:,0:4],
			self.X2: self.img_pair[:,:,:,4:8]
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

		self.X1 = tf.placeholder(dtype=tf.float32, shape=(self.batch_size, 224, 384, 4))
		self.X2 = tf.placeholder(dtype=tf.float32, shape=(self.batch_size, 224, 384, 4))
		self.Y = tf.placeholder(dtype=tf.float32, shape=(self.batch_size, 224, 384, 3))

		self.tunnel1 = network.network_tunnel(self.X1,'tunnel_layer1')
		self.tunnel2 = network.network_tunnel(self.X2,'tunnel_layer2')

		self.predict_flow5, self.predict_flow2 = network.network_core(self.tunnel1,self.tunnel2)

	def load_model_ckpt(self,sess,filename):
		saver = tf.train.Saver()
		saver.restore(sess, tf.train.latest_checkpoint(filename))