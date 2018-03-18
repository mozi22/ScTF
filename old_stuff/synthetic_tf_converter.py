



''' 
	A single example in tfrecord will have the following information 

	Important: Seprate tfrecords files will be created for fast and slow versions of the readings ( ask this from the supervisor also )
	
	file_id -> {{ id of the file represented by the frames indexes in camera_data or file names in other folders for e.g 0000.pfm means the id is 0000 }}
	camera_data -> {{ only the frame value }}
	scene_type -> forward | backward
	disparity_stereo_l -> {{ matrix representing the left stereo pair disparity values }}
	disparity_stereo_r -> {{ matrix representing the right stereo pair disparity values }}

	disparity_change_into_future_l -> {{ matrix representing the left stereo pair disparity_change values }}
	disparity_change_into_future_r -> {{ matrix representing the right stereo pair disparity_change values }}

	disparity_change_into_past_l -> {{ matrix representing the left stereo pair disparity_change values }}
	disparity_change_into_past_r -> {{ matrix representing the right stereo pair disparity_change values }}

	optical_flow_into_past_l 	-> {{ matrix representing the optical flow for the left stereo pair into past }}
	optical_flow_into_past_r 	-> {{ matrix representing the optical flow for the right stereo pair into past }}
	optical_flow_into_future_l 	-> {{ matrix representing the optical flow for the left stereo pair into future }}
	optical_flow_into_future_r 	-> {{ matrix representing the optical flow for the right stereo pair into future }}

	image_webp -> {{ the image encoded as webp for which we have all the above values }}
'''

import re
import numpy as np
from PIL import Image 
import tensorflow as tf
import os, os.path
import math
from multiprocessing import Process
import ijremote as ij

class SyntheticTFRecordsWriter:


	def __init__(self):

		self.max_depth_driving = 30.7278
		self.max_depth_driving_chng = 7.5552e+08

		self.max_depth2 = 0
		self.max_depth = 0
		self.max_depth_chng = 0


		# these params need to be updated when running on the bigger dataset

		# here 2 means 0000 and 0001 i.e 2 records

		self.flyingdata_TEST_FOLDERS_IDS = 150
		self.flyingdata_TRAIN_FOLDERS_IDS = 750
		self.flyingdata_FILES_IDS = [6,16]

		# self.flyingdata_TEST_FOLDERS_IDS = 2
		# self.flyingdata_TRAIN_FOLDERS_IDS = 2
		# self.flyingdata_FILES_IDS = [6,8]

		# self.dataset_root = '../dataset_synthetic_sm50/'
		self.dataset_save = '../dataset_synthetic/tfrecords2/driving'
		self.dataset_root = '../dataset_synthetic/'

		self.datasets = ['driving','flyingthings3d','monkaa']
		
		self.data_types = [ 
			'disparity',
			'disparity_change',
			'frames_finalpass_webp',
			'optical_flow'
		]

		self.camera_focal_lengths = ['35mm_focallength']

		self.scene_types = [
			'scene_backwards',
			# 'scene_forwards'
		]


		self.camera_speeds = ['fast']


		self.times = [
			'into_future'
			# 'into_past'
		]


		# for monkaa ( need to add the others here too for large dataset)
		self.monkaa_scenes = ['a_rain_of_stones_x2','eating_camera2_x2','treeflight_x2','flower_storm_augmented1_x2',
		'eating_x2','lonetree_augmented1_x2','funnyworld_x2','family_x2',
		'treeflight_augmented1_x2','funnyworld_augmented0_x2','eating_naked_camera2_x2','funnyworld_camera2_x2',
		'lonetree_winter_x2','funnyworld_camera2_augmented1_x2','flower_storm_x2','treeflight_augmented0_x2',
		'lonetree_x2','top_view_x2','funnyworld_augmented1_x2','funnyworld_camera2_augmented0_x2',
		'lonetree_difftex_x2','lonetree_augmented0_x2','flower_storm_augmented0_x2','lonetree_difftex2_x2'] 
		# self.monkaa_scenes = ['a_rain_of_stones_x2','eating_camera2_x2']

		# for flyingthings3d
		self.tnts = ['TRAIN','TEST']
		self.letters = ['A','B','C']

		# 7 will be 15 here where both values are inclusive. Named from 0006 too 0015.
		# in each folder.
		self.flying_data_folder_train_limit = [0,self.flyingdata_TRAIN_FOLDERS_IDS]
		self.flying_data_folder_test_limit = [0,self.flyingdata_TEST_FOLDERS_IDS]
		self.flying_data_file_imit = self.flyingdata_FILES_IDS


		self.directions = ['left']





	# we load the content of the file in memory and refer the lines required
	# in the for loop w.r.t each id (item).
	def load_camera_file(self,path):
		path = path + '/camera_data.txt'
		with open(path, "r") as myfile:
		    data=myfile.readlines()
		    return data

	# gets the L,R frame values from camera_data based on the id passed.
	def get_frame_by_id(self,idd, dataset_flying = False):

		if dataset_flying:
			if len(self.camera_data) < 37:
				idd = 8

		frame_line = idd * 4

		left = self.camera_data[frame_line + 1]
		right = self.camera_data[frame_line + 2]

		left_values = map(lambda x: float(x),left.split(' ')[1:])
		right_values = map(lambda x: float(x),right.split(' ')[1:])

		left_values_as_np = np.fromiter(left_values, dtype=np.float32)
		right_values_as_np = np.fromiter(right_values, dtype=np.float32)

		return [left_values_as_np,right_values_as_np]


	def get_optical_flow_file_name(self,direction,time,file_id):
		if direction == self.directions[0] and time == self.times[0]:
			return 'OpticalFlowIntoFuture_'+str(file_id)+'_L'
		# elif direction == self.directions[1] and time == self.times[0]:
			# return 'OpticalFlowIntoFuture_'+str(file_id)+'_R'
		elif direction == self.directions[0] and time == self.times[1]:
			return 'OpticalFlowIntoPast_'+str(file_id)+'_L'
		# elif direction == self.directions[1] and time == self.times[1]:
		# 	return 'OpticalFlowIntoPast_'+str(file_id)+'_R'

	def parse_driving_dataset(self,dataset):

		# first we create the test file. And than the rest as training.

		print('Converting '+ dataset + '...')

		test_writer = self.init_tfrecord_writer(self.dataset_save+'_TEST.tfrecords')
		train_writer = self.init_tfrecord_writer(self.dataset_save+'_TRAIN.tfrecords')

		for camera_focal_length in self.camera_focal_lengths:
			for scene_type in self.scene_types:
				for camera_speed in self.camera_speeds:

					# read the camera frames file
					path = self.dataset_root + '/'.join([dataset,'camera_data',camera_focal_length,scene_type,camera_speed])
					self.camera_data = self.load_camera_file(path)
					for time in self.times:
						for direction in self.directions:
							disparity_path = (path + '/' + direction).replace('camera_data','disparity')
							disparity_change_path = (path + '/' + time + '/' + direction).replace('camera_data','disparity_change')
							optical_flow_path = (path + '/' + time + '/' + direction).replace('camera_data','optical_flow')
							frames_finalpass_webp_path = (path + '/' + direction).replace('camera_data','frames_finalpass_webp')

							files_count_in_this_scene = len([name for name in os.listdir(disparity_path + '/')])
							test_files = 100

							print(disparity_path)
							for file_id in range(1,files_count_in_this_scene + 1):

								# break out since we don't want to keep the last file.
								# suppose the last file is 300. We don't have 301 which will
								# be its pair to be inputted to the network.
								if file_id == files_count_in_this_scene:
									break

								disparity = disparity_path + '/' + "%04d" % (file_id,) + '.pfm'
								disparity2 = disparity_path + '/' + "%04d" % (file_id+1,) + '.pfm'
								disparity_change = disparity_change_path + '/' + "%04d" % (file_id,) + '.pfm'


								optical_flow = optical_flow_path + '/' + self.get_optical_flow_file_name(direction,time,"%04d" % (file_id,)) + '.pfm'
								frames_finalpass_webp = frames_finalpass_webp_path + '/' + "%04d" % (file_id,) + '.webp'
								frames_finalpass_webp2 = frames_finalpass_webp_path + '/' + "%04d" % (file_id+1,) + '.webp'



								patches = self.from_paths_to_data(
									disparity,
									disparity2,
									disparity_change,
									optical_flow,
									frames_finalpass_webp,
									frames_finalpass_webp2,
									camera_focal_length)

								camera_L_R = self.get_frame_by_id(file_id - 1)

								if file_id > test_files:
									print('train written')
									self.create_tf_example(patches,
										camera_L_R,
										train_writer,
										time)
								else:
									print('test written')
									self.create_tf_example(patches,
										camera_L_R,
										test_writer,
										time)

			# 					break
			# 				break
			# 			break
			# 		break
			# 	break
			# break


		self.close_writer(train_writer)
		self.close_writer(test_writer)

	def create_tf_example(self,patches,camera_L_R,writer,scene_direction):

		for item in patches:

			# downsampled_opt_flow = self.downsample_labels(np.array(item['opt_fl']),2)
			# downsampled_disp_chng = self.downsample_labels(np.array(item['disp_chng']),0)

			width , height = item['depth'].shape[0] , item['depth'].shape[1]
			depth = item['depth'].tostring()
			depth2 = item['depth2'].tostring()

			opt_flow = item['optical_flow'].tostring()
			cam_frame_L = camera_L_R[0].tostring()
			cam_frame_R = camera_L_R[1].tostring()
			depth_chng = item['disp_change'].tostring()
			frames_finalpass_webp = item['web_p'].tostring()
			frames_finalpass_webp2 = item['web_p2'].tostring()


			if scene_direction == self.times[0]:
				direction = b'f';
			else:
				direction = b'b';


			example = tf.train.Example(features=tf.train.Features(
				feature={
					'width': self._int64_feature(width),
					'height': self._int64_feature(height),
					'depth1': self._bytes_feature(depth),
					'depth2': self._bytes_feature(depth2),
					'disp_chng': self._bytes_feature(depth_chng),
					'opt_flow': self._bytes_feature(opt_flow),
					'cam_frame_L': self._bytes_feature(cam_frame_L),
					'cam_frame_R': self._bytes_feature(cam_frame_R),
					'image1': self._bytes_feature(frames_finalpass_webp),
					'image2': self._bytes_feature(frames_finalpass_webp2),


					# into_future or into_past for optic flow and disp change 
					'direction':self._bytes_feature(direction)
			    }),
			)

			writer.write(example.SerializeToString())



	# 	return patches
	def downsample_opt_flow(self,data,size):
		data = np.delete(data,2,axis=2)

		u = data[:,:,0]
		v = data[:,:,1]
		
		dt = Image.fromarray(u)
		dt = dt.resize(size, Image.NEAREST)

		dt2 = Image.fromarray(v)
		dt2 = dt2.resize(size, Image.NEAREST)

		u = np.array(dt)
		v = np.array(dt2)

		return np.stack((u,v),axis=2)

	def warp_image(self,f1,f2,data):
		sess = tf.InteractiveSession()
		npf1 = np.array(f1,dtype=np.float32)
		npf2 = np.array(f2,dtype=np.float32)


		x = list(range(0,256))
		y = list(range(0,160))
		X, Y = tf.meshgrid(x, y)

		X = tf.cast(X,np.float32) + data[:,:,0]
		Y = tf.cast(Y,np.float32) + data[:,:,1]

		con = tf.stack([X,Y])
		result = tf.transpose(con,[1,2,0])
		result = tf.expand_dims(result,0)

		# print(sess.run(result).shape)
		# print(npf2[np.newaxis,:,:,:].shape)


		mmm = tf.contrib.resampler.resampler(npf2[np.newaxis,:,:,:],result)
		# print(mmm.eval())

		a = Image.fromarray(mmm.eval()[0].astype(np.uint8))
		a.save('./dbc.png')
		a.show()
		f1.show()


	def get_depth_from_disp(self,disparity):
		focal_length = 35
		disp_to_depth = focal_length / disparity
		return disp_to_depth

	def get_depth_chng_from_disp_chng(self,disparity,disparity_change):
		disp2 = disparity + disparity_change

		depth1 = self.get_depth_from_disp(disparity)
		depth2 = self.get_depth_from_disp(disp2)

		return depth1 - depth2

	def normalizeOptFlow(self,flow,input_size):

		# remove the values bigger than the image size
		flow[:,:,0][flow[:,:,0] > input_size[0] ] = 0
		flow[:,:,1][flow[:,:,1] > input_size[1] ] = 0

		# separate the u and v values 
		flow_u = flow[:,:,0]
		flow_v = flow[:,:,1]

		# Image.fromarray(flow[:,:,0]).show()
		# Image.fromarray(flow[:,:,1]).show()

		# opt_u = np.squeeze(flow_u).astype(np.uint8)
		# opt_v = np.squeeze(flow_v).astype(np.uint8)

		# result = np.dstack((flow_u,flow_v))
		# opt_u = Image.fromarray(result[:,:,0]).show() 
		# opt_v = Image.fromarray(result[:,:,1]).show()

		Image.fromarray(flow[:,:,0]).save('opf_1.tiff')

		# normalize the values by the image dimensions
		flow_u = flow_u / input_size[0]
		flow_v = flow_v / input_size[1]



		# combine them back and return
		return np.dstack((flow_u,flow_v))



	def from_paths_to_data(self,disparity,
								disparity2,
								disparity_change,
								optical_flow,
								frames_finalpass_webp_path,
								frames_finalpass_webp_path2,
								focal_length):


		# print('disp2 = '+ disparity2)
		# print('optical_flow = '+ optical_flow)
		# print('frames_finalpass_webp_path1 = '+ frames_finalpass_webp_path)
		# print('frames_finalpass_webp_path2 = '+ frames_finalpass_webp_path2)
		# print('')


		# reduce the sizes of images and flow values by this factor
		self.factor = 0.4
		input_size = int(960 * self.factor), int(540 * self.factor)

		# parse pfm files for disparities
		disparity  = self.readPFM(disparity)[0]
		disparity2  = self.readPFM(disparity2)[0]

		disparity_change = self.readPFM(disparity_change)[0]


		opt_flow = self.readPFM(optical_flow)[0]

		# reduce optical flow size
		opt_flow = self.downsample_opt_flow(opt_flow,input_size)

		# reduce the flow values by same factor as image sizes are reduced by
		opt_flow = opt_flow * self.factor


		# normalize flow values between 0 - 1

		ij.setImage('before_u',opt_flow[:,:,0])
		ij.setImage('before_v',opt_flow[:,:,1])
		opt_flow = self.normalizeOptFlow(opt_flow,input_size)
		Image.fromarray(opt_flow[:,:,0]).save('opf_1n.tiff')


		web_p_file = Image.open(frames_finalpass_webp_path)
		web_p_file2 = Image.open(frames_finalpass_webp_path2)

		web_p_file = web_p_file.resize(input_size, Image.BILINEAR)
		web_p_file2 = web_p_file2.resize(input_size, Image.BILINEAR)

		disparity = Image.fromarray(disparity)
		disparity2 = Image.fromarray(disparity2)
		disparity_change = Image.fromarray(disparity_change)

		disparity = disparity.resize(input_size,Image.NEAREST)
		disparity2 = disparity2.resize(input_size,Image.NEAREST)
		disparity_change = disparity_change.resize(input_size,Image.NEAREST)

		disparity = np.array(disparity)
		disparity2 = np.array(disparity2)
		disparity_change = np.array(disparity_change)

		# disparity = disparity / self.max_disparity_driving
		# disparity2 = disparity2 / self.max_disparity_driving
		# disparity_change = disparity_change / self.max_disparity_chng_driving


		# reduce the disparity values by same factor as image sizes are reduced by
		# disparity = disparity * self.factor
		# disparity2 = disparity2 * self.factor
		# disparity_change = disparity_change * self.factor


		# there are 0 values in disparity_change. We can add an epsilon to to shift the matrix.
		disparity_change = disparity_change + 1e-6

		# convert disparities to depth
		depth = self.get_depth_from_disp(disparity)
		depth2 = self.get_depth_from_disp(disparity2)
		depth_change = self.get_depth_chng_from_disp_chng(disparity,disparity_change)



		# depth_mx = np.amax(depth) 
		# depth2_mx = np.amax(depth2) 
		# depth_change_mx = np.amax(depth_change)


		# # print max depth value
		# if float(self.max_depth) < depth_mx:
		# 	self.max_depth = depth_mx
		# 	print('Max value = '+ str(self.max_depth))

		# if float(self.max_depth2) < depth2_mx:
		# 	self.max_depth2 = depth2_mx
		# 	print('Max value2 = '+ str(self.max_depth2))

		# if float(self.max_depth_chng) < depth_change_mx:
		# 	self.max_depth_chng = depth_change_mx
		# 	print('Max value_chng = '+ str(self.max_depth_chng))


		# normalize depth values
		depth = depth / self.max_depth_driving
		depth2 = depth2 / self.max_depth_driving
		depth_change = depth_change / self.max_depth_driving_chng

		depth_mx = np.amax(depth) 
		depth2_mx = np.amax(depth2) 
		depth_change_mx = np.amax(depth_change)

		# print if depth > 1 ( shouldn't be the case after normalization)
		if depth_mx > 1:
			print('> depth1 = '+ str(depth_mx))

		if depth2_mx > 1:
			print('> depth2 = '+ str(depth2_mx))

		if depth_change_mx > 1:
			print('> depth_Change = '+ str(depth_change_mx))

		# if max_disparity2  > self.max_disparity2:
		# 	self.max_disparity2 = max_disparity2
		# 	print('max disparity2 updated = '+str(self.max_disparity2))

		# if max_disparity  > self.max_disparity:
		# 	self.max_disparity = max_disparity
		# 	print('max disparity updated = '+str(self.max_disparity))

		# if max_disparity_change > self.max_disparity_chng:
		# 	self.max_disparity_chng = max_disparity_change
		# 	print('max disparity change updated = '+str(self.max_disparity_chng))


		# disparity = disparity / self.max_disparity_driving
		# disparity2 = disparity2 / self.max_disparity_driving
		# disparity_change = disparity_change / self.max_disparity_chng_driving


		# if optical_flow == '../dataset_synthetic_sm/driving/optical_flow/35mm_focallength/scene_backwards/fast/into_future/left/OpticalFlowIntoFuture_0001_L.pfm':
		# 	print(opt_flow)
			# self.warp_image(web_p_file,web_p_file2,opt_flow)

		# print('web_p_file.shape')
		# print(np.array(web_p_file).shape)
		# print(disparity)
		# print(disparity2.shape)
		# print(disparity_change.shape)
		# print(opt_flow.shape)
		# print('')


		web_p_file = np.array(web_p_file)
		web_p_file2 = np.array(web_p_file2)

		# web_p_file = web_p_file / 255
		# web_p_file2 = web_p_file2 / 255


		return [{
			'web_p': web_p_file,
			'web_p2': web_p_file2,
			'depth': depth,
			'depth2': depth2,
			'disp_change': depth_change,
			'optical_flow': opt_flow,
			'path': optical_flow
		}]




	def convert(self):

	    self.runInParallel(
	    				self.parse_driving_dataset(self.datasets[0])
	    				)
	    # self.runInParallel(self.parse_flyingthings3d_dataset(self.datasets[1]))
	
	def runInParallel(self,*fns):
	  proc = []
	  for fn in fns:
	    p = Process(target=fn)
	    p.start()
	    proc.append(p)
	  for p in proc:
	    p.join()



	# reads the PFM file and returns an np matrix.
	def readPFM(self,file):
		file = open(file, 'rb')

		color = None
		width = None
		height = None
		scale = None
		endian = None

		header = file.readline().decode('utf-8').rstrip()
		if header == 'PF':
			color = True
		elif header == 'Pf':
			color = False
		else:
			raise Exception('Not a PFM file.')

		dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
		if dim_match:
			width, height = map(int, dim_match.groups())
		else:
			raise Exception('Malformed PFM header.')

		scale = float(file.readline().decode('utf-8').rstrip())
		if scale < 0: # little-endian
			endian = '<'
			scale = -scale
		else:
			endian = '>' # big-endian

		data = np.fromfile(file, endian + 'f')
		shape = (height, width, 3) if color else (height, width)

		data = np.reshape(data, shape)
		data = np.flipud(data)
		return data, scale

	# close the file writer
	def close_writer(self,writer):
		writer.close()

	# value: the value needed to be converted to feature
	def _bytes_feature(self,value):
	    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

	def _int64_feature(self,value):
	    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
	  
	def init_tfrecord_writer(self,filename):
		return tf.python_io.TFRecordWriter(self.dataset_root + '/' + filename)

# a = SyntheticTFRecordsWriter()
# a.convert()
