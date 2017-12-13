



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

class SyntheticTFRecordsWriter:


	def __init__(self):


		# these params need to be updated when running on the bigger dataset

		# here 2 means 0000 and 0001 i.e 2 records
		self.flyingdata_TEST_FOLDERS_IDS = 2
		self.flyingdata_TRAIN_FOLDERS_IDS = 2
		self.flyingdata_FILES_IDS = [6,8]
		self.driving_FILES = 3
		self.monkaa_FILES = 2

		# for driving 50 files will be kept as testing and the rest for training.
		self.driving_TEST_FILES_COUNT = 50

		# for monkaa 20 files will be kept as testing and the rest for training.
		self.monkaa_TEST_FILES_COUNT = 20


		self.dataset_root = '/home/muazzam/mywork/python/thesis/danda/dataset/'

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
			'scene_forwards'
		]

		self.camera_speeds = ['fast']


		self.times = [
			'into_future',
			'into_past'
		]


		# for monkaa ( need to add the others here too for large dataset)
		self.monkaa_scenes = ['a_rain_of_stones_x2','eating_camera2_x2'] 


		# for flyingthings3d
		self.tnts = ['TRAIN','TEST']
		self.letters = ['A','B','C']

		# 7 will be 15 here where both values are inclusive. Named from 0006 too 0015.
		# in each folder.
		self.flying_data_folder_train_limit = [0,self.flyingdata_TRAIN_FOLDERS_IDS]
		self.flying_data_folder_test_limit = [0,self.flyingdata_TEST_FOLDERS_IDS]
		self.flying_data_file_limit = self.flyingdata_FILES_IDS


		self.directions = ['left','right']





	# we load the content of the file in memory and refer the lines required
	# in the for loop w.r.t each id (item).
	def load_camera_file(self,path):
		path = path + '/camera_data.txt'
		with open(path, "r") as myfile:
		    data=myfile.readlines()
		    return data

	# gets the L,R frame values from camera_data based on the id passed.
	def get_frame_by_id(self,idd):

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
		elif direction == self.directions[1] and time == self.times[0]:
			return 'OpticalFlowIntoFuture_'+str(file_id)+'_R'
		elif direction == self.directions[0] and time == self.times[1]:
			return 'OpticalFlowIntoPast_'+str(file_id)+'_L'
		elif direction == self.directions[1] and time == self.times[1]:
			return 'OpticalFlowIntoPast_'+str(file_id)+'_R'

	def parse_driving_dataset(self,dataset):
		test_examples_counter = 1

		self.init_tfrecord_writer(dataset+'_TEST.tfrecords')

		# first we create the test file. And than the rest as training.

		print('Converting '+ dataset + '...')

		for data_type in self.data_types:
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

								for file_id in range(1,self.driving_FILES):

									disparity = disparity_path + '/' + "%04d" % (file_id,) + '.pfm'
									disparity_change = disparity_change_path + '/' + "%04d" % (file_id,) + '.pfm'
									optical_flow = optical_flow_path + '/' + self.get_optical_flow_file_name(direction,time,"%04d" % (file_id,)) + '.pfm'
									frames_finalpass_webp = frames_finalpass_webp_path + '/' + "%04d" % (file_id,) + '.webp'


									disparity,disparity_change,optical_flow,frames_finalpass_webp = self.from_paths_to_data(
										disparity,
										disparity_change,
										optical_flow,
										frames_finalpass_webp)

									camera_L_R = self.get_frame_by_id(file_id)

									self.create_tf_example(disparity,
										disparity_change,
										optical_flow,
										frames_finalpass_webp,
										camera_L_R)

									if test_examples_counter == self.driving_TEST_FILES_COUNT:
										test_examples_counter = 0
										self.close_writer()
										self.init_tfrecord_writer(dataset+'_TRAIN.tfrecords')
									else:										
										test_examples_counter = test_examples_counter + 1





	def create_tf_example(self,disparity, disparity_change, optical_flow,frames_finalpass_webp,camera_L_R):

		disp = disparity.tostring()
		disp_chng = disparity_change.tostring()
		opt_flow = optical_flow.tostring()
		cam_frame_L = camera_L_R[0].tostring()
		cam_frame_R = camera_L_R[1].tostring()


		example = tf.train.Example(features=tf.train.Features(
			feature={
			    'disparity_width': self._int64_feature(disparity.shape[1]),
			    'disparity_height': self._int64_feature(disparity.shape[0]),
			    'disparity_change_width': self._int64_feature(disparity_change.shape[1]),
			    'disparity_change_height': self._int64_feature(disparity_change.shape[0]),
			    'optical_flow_width': self._int64_feature(optical_flow.shape[1]),
			    'optical_flow_height': self._int64_feature(optical_flow.shape[0]),
			    'disp': self._bytes_feature(disp),
			    'disp_chng': self._bytes_feature(disp_chng),
			    'opt_flow': self._bytes_feature(opt_flow),
			    'cam_frame_L': self._bytes_feature(cam_frame_L),
			    'cam_frame_R': self._bytes_feature(cam_frame_R),
			    'image': self._bytes_feature(frames_finalpass_webp)
		    }),
		)

		self.writer.write(example.SerializeToString())


	def from_paths_to_data(self,disparity,disparity_change,optical_flow,frames_finalpass_webp_path):
		web_p_file = open(frames_finalpass_webp_path, 'rb').read()
		disparity, _ = self.readPFM(disparity)
		disparity_change, _ = self.readPFM(disparity_change)
		optical_flow, _ = self.readPFM(optical_flow)
		return disparity, disparity_change, optical_flow, web_p_file


	def parse_flyingthings3d_dataset(self,dataset):

		print('Converting '+ dataset + '...')
		path = ''
		for tnt in self.tnts:

			if tnt == self.tnts[0]:
				self.init_tfrecord_writer(dataset+'_TRAIN.tfrecords')
				folders_range = self.flying_data_folder_train_limit[1]
			else:
				self.close_writer()
				self.init_tfrecord_writer(dataset+'_TEST.tfrecords')
				folders_range = self.flying_data_folder_test_limit[1]

			for data_type in self.data_types:
				for let in self.letters:


					for folder_id in range(0,folders_range):

						path = self.dataset_root + '/'.join([dataset,'camera_data',tnt,let,"%04d" % (folder_id,)])
						self.camera_data = self.load_camera_file(path)

						for direction in self.directions:
							for time in self.times:
								for file_id in range(self.flying_data_file_limit[0],self.flying_data_file_limit[1]):

									disparity_path = (path + '/' + direction).replace('camera_data','disparity') + '/' + str("%04d" % (file_id,)) + '.pfm'
									disparity_change_path = (path + '/' + time + '/' + direction).replace('camera_data','disparity_change') + '/' + str("%04d" % (file_id,)) + '.pfm'
									optical_flow_path = (path.replace('camera_data','optical_flow') + '/' + time + '/' + direction + '/' + self.get_optical_flow_file_name(direction,time,"%04d" % (file_id,))) + '.pfm'

									frames_finalpass_webp_path = (path + '/' + direction).replace('camera_data','frames_finalpass_webp') + '/' + str("%04d" % (file_id,)) + '.webp'


									disparity,disparity_change,optical_flow,frames_finalpass_webp = self.from_paths_to_data(
										disparity_path,
										disparity_change_path,
										optical_flow_path,
										frames_finalpass_webp_path)

									camera_L_R = self.get_frame_by_id(file_id)

									self.create_tf_example(disparity,
										disparity_change,
										optical_flow,
										frames_finalpass_webp,
										camera_L_R)



	def parse_monkaa_dataset(self,dataset):

		print('Converting '+ dataset + '...')
		self.init_tfrecord_writer(dataset+'_TEST.tfrecords')
		test_examples_counter = 1
		for data_type in self.data_types:
			for scene in self.monkaa_scenes:
				path = self.dataset_root + '/'.join([dataset,'camera_data',scene])
				self.camera_data = self.load_camera_file(path)


				for direction in self.directions:
					for time in self.times:
						for file_id in range(0,self.monkaa_FILES):
							disparity_path = (path + '/' + direction).replace('camera_data','disparity') + '/' + str("%04d" % (file_id,)) + '.pfm'
							disparity_change_path = (path + '/' + time + '/' + direction).replace('camera_data','disparity_change') + '/' + str("%04d" % (file_id,)) + '.pfm'
							optical_flow_path = (path.replace('camera_data','optical_flow') + '/' + time + '/' + direction + '/' + self.get_optical_flow_file_name(direction,time,"%04d" % (file_id,))) + '.pfm'
							frames_finalpass_webp_path = (path + '/' + direction).replace('camera_data','frames_finalpass_webp') + '/' + str("%04d" % (file_id,)) + '.webp'


							disparity,disparity_change,optical_flow,frames_finalpass_webp = self.from_paths_to_data(
								disparity_path,
								disparity_change_path,
								optical_flow_path,
								frames_finalpass_webp_path)

							camera_L_R = self.get_frame_by_id(file_id)

							self.create_tf_example(disparity,
								disparity_change,
								optical_flow,
								frames_finalpass_webp,
								camera_L_R)

							if test_examples_counter == self.monkaa_TEST_FILES_COUNT:
								test_examples_counter = 0
								self.close_writer()
								self.init_tfrecord_writer(dataset+'_TRAIN.tfrecords')
							else:										
								test_examples_counter = test_examples_counter + 1




	def convert(self):

		path = ''

		for dataset in self.datasets:
			if dataset == self.datasets[0]:
				self.parse_driving_dataset(dataset)
			elif dataset == self.datasets[1]:
				self.parse_flyingthings3d_dataset('flyingthings3d')
			else:
				self.parse_monkaa_dataset('monkaa')

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
	def close_writer(self):
		self.writer.close()

	# value: the value needed to be converted to feature
	def _bytes_feature(self,value):
	    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

	def _int64_feature(self,value):
	    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

	def init_tfrecord_writer(self,filename):
		self.writer = tf.python_io.TFRecordWriter(filename)



a = SyntheticTFRecordsWriter()
a.convert()