import numpy as np
import tensorflow as tf
from   PIL import Image
import helpers as hpl
import network


class FlowPredictor:

	# img1: path of img1
	# img2: path of img2
	# depth1: path of depth pfm 1
	# depth2: path of depth pfm 2
	def preprocess(self,img1,img2,disparity1,disparity2):


		factor = 0.4
		self.input_size = int(960 * factor), int(540 * factor)
		self.driving_max_depth = 30.7278
		self.ckpt = './ckpt/ckpt'


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

		# resize depth values
		depth1 = self.get_depth_from_disp(np.array(disp1))
		depth2 = self.get_depth_from_disp(np.array(disp2))

		# normalize depth values
		self.depth1 = depth1 / self.driving_max_depth
		self.depth2 = depth2 / self.driving_max_depth

		# combine depth values with images
		rgbd1 = self.combine_depth_values(self.img1,self.depth1)
		rgbd2 = self.combine_depth_values(self.img2,self.depth2)

		# combine images to 8 channel rgbd-rgbd
		img_pair = np.concatenate((rgbd1,rgbd2),axis=2)

		img_pair = np.pad(img_pair,((4,4),(0,0),(0,0)),'constant')

		self.img_pair = np.expand_dims(img_pair,0)
		self.initialize_network()

		self.sess = tf.InteractiveSession()
		self.load_model_ckpt(self.sess,'ckpt/driving/model_ckpt_2999.ckpt')


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

	def denormalize_flow(self,flow):
		u = flow[:,:,0] * self.input_size[0]
		v = flow[:,:,1] * self.input_size[1]
		w = flow[:,:,2] * self.driving_max_depth

		return np.stack((u,v,w),axis=2)


	def postprocess(self,flow):

		flow = self.denormalize_flow(flow)

		self.img2_arr = np.pad(self.img2_arr,((4,4),(0,0),(0,0)),'constant')

		flow = self.warp(self.img2_arr,flow)
		print(flow.eval().shape)
		a = Image.fromarray(flow.eval()[0].astype(np.uint8))
		a.save('./abc.png')
		a.show()
		self.init_img1.show()
		self.init_img2.show()



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


	def combine_depth_values(self,img,depth):
		depth = np.expand_dims(depth,2)
		return np.concatenate((img,depth),axis=2)

	def initialize_network(self):

		self.batch_size = 1

		self.X = tf.placeholder(dtype=tf.float32, shape=(self.batch_size, 224, 384, 8))
		self.Y = tf.placeholder(dtype=tf.float32, shape=(self.batch_size, 224, 384, 3))
		self.predict_flow5, self.predict_flow2 = network.train_network(self.X)

	def load_model_ckpt(self,sess,filename):
		saver = tf.train.Saver()
		saver.restore(sess, filename)

