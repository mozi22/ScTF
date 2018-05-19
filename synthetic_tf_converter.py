



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
import glob
import csv
from multiprocessing import Process
# import ijremote as ij
from os.path import isfile, join
class SyntheticTFRecordsWriter:


	def __init__(self):


		# 1 = driving
		# 2 = flying
		# 3 = monkaa
		# 4 = ptb

		# this param decides which dataset to parse.
		self.dataset_number = 4

		# these are inverse depths
		self.max_depth_driving = 0.232809
		# self.max_depth_driving_chng = 2.70248
		self.max_depth_driving_chng = 0.157644


		self.max_depth_monkaa = 120.009 
		self.max_depth_monkaa_chng = 7.5552e+08


		self.values_between_highest_and_lowest = 0

		self.max_depth_flying = 119.986
		self.max_depth_flying_chng = 1.20306e+06

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
		self.dataset_save = '../dataset_synthetic/'
		self.dataset_root = '../dataset_synthetic/'

		self.dataset_ptb_root = '../dataset_ptb/'
		self.ptb_folders = ['ValidationSet']


		self.datasets = ['driving','flyingthings3d','monkaa','ptb']
		
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
		self.monkaa_scenes = ['a_rain_of_stones_x2','eating_camera2_x2','treeflight_x2',
		# 'flower_storm_augmented1_x2',
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
		self.flying_data_file_limit = self.flyingdata_FILES_IDS


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



	def parse_ptb_dataset(self,dataset):

		print('Converting '+ dataset + '...')

		self.u_factor = 0.414814815
		self.v_factor = 0.4

		input_size = math.ceil(960 * self.v_factor), math.floor(540 * self.u_factor)

		test_writer = self.init_tfrecord_writer(self.dataset_save+'ktp_TEST.tfrecords')
		# train_writer = self.init_tfrecord_writer(self.dataset_save+'ktp_TRAIN.tfrecords')

		# dataset_max_values = [36277,31452,29610,65168,34026,65168,65168,65168,39217,65168,
		# 					  65168,65168,65168,65168,65168,65168,65168,18042,65168,65168,
		# 					  65168,65168,35176,65168,30839,65168,17389,65168,65168,65168,
		# 					  32025,32947,65168,17216,65168,65168,47879,65168,55008,38168,
		# 					  65168,64405,65168,53785,65168,65168,65168,65168,65168,54029,
		# 					  65133,65168,56201,55684,65168,65168,65168,65168,65168,65168,
		# 					  18111,65168,65168,17323,54060,65168,28986,65168,65168,31325,
		# 					  65168,65168,42126,65168,65168,65168,59395,44633,65168,65036,
		# 					  65168,37144,65168,63516,17847,65168,22517,65168,65168,65168,
		# 					  65168,49099,65168,65168,65168]


		for set_type,dirss in enumerate(self.ptb_folders):
			path = self.dataset_ptb_root + self.ptb_folders[set_type] + '/'


			# print('#######################################################')
			# print(' module = '+ self.ptb_folders[set_type])
			# print('#######################################################')

			evaluation_dirs = os.listdir(path)

			for i,item in enumerate(evaluation_dirs):

				# max_value = dataset_max_values[i]

				new_path = path + item

				depth_path = new_path + '/depth'
				rgb_path = new_path + '/rgb'

				# self.rename_ptb_files(depth_path,rgb_path)

				files_depth = [f for f in os.listdir(depth_path) if isfile(join(depth_path, f))]
				files_rgb = [f for f in os.listdir(rgb_path) if isfile(join(rgb_path, f))]

				files_depth = sorted(files_depth)
				files_rgb = sorted(files_rgb)

				self.max_depth = 0
				self.max_depth2 = 0

				# print('#######################################################')
				# print(' module = '+ item)
				# print('#######################################################')

				for i in range(0,len(files_depth)):

					if i == len(files_depth) - 1:
						break

					img1 = rgb_path + '/' + files_rgb[i]
					img2 = rgb_path + '/' + files_rgb[i+1]

					depth1 = depth_path + '/' + files_depth[i]
					depth2 = depth_path + '/' + files_depth[i+1]

					img1 = Image.open(img1)
					img2 = Image.open(img2)
					
					depth1 = Image.open(depth1)
					depth2 = Image.open(depth2)

					img1 = img1.resize(input_size, Image.BILINEAR)
					img2 = img2.resize(input_size, Image.BILINEAR)

					depth1 = depth1.resize(input_size, Image.NEAREST)
					depth2 = depth2.resize(input_size, Image.NEAREST)

					# depth1 = self.visualize_ptb_image(depth1,True)
					# depth2 = self.visualize_ptb_image(depth2)

					depth1 = np.array(depth1)
					depth2 = np.array(depth2)


					# normalize depth values
					# depth1 = depth1 / max_value
					# depth2 = depth2 / max_value

					# self.check_max_depth(depth1,depth2,0)

					img1 = np.array(img1) / 255
					img2 = np.array(img2) / 255

					optical_flow = np.zeros_like(img1)
					depth_change = np.zeros_like(img1)

					patches = [{
						'web_p': img1,
						'web_p2': img2,
						'depth1': np.array(depth1),
						'depth2': np.array(depth2),
						'depth_change': depth_change,
						'optical_flow': optical_flow,
						'path': ''
					}]


					# if set_type == 0:
					# 	self.create_tf_example(patches,
					# 		'',
					# 		train_writer,
					# 		'')
					# else:
					# print('train finished')
					self.create_tf_example(patches,
						'',
						test_writer,
						'')

		# self.close_writer(train_writer)
		self.close_writer(test_writer)


	def max_ptb_depth_values(depth1,depth2):
		
		depth1 = np.max(depth1)
		depth2 = np.max(depth2)

		if depth1 > self.max_depth:
			self.max_depth = depth1
			print(self.max_depth)

		if depth2 > self.max_depth2:
			self.max_depth2 = depth2
			print(self.max_depth2)


	def visualize_ptb_image(self,depth,show_as_img=False):

		# depth = '../dataset_ptb/EvaluationSet/bag1/depth/10.png'
		# depth = Image.open(depth)

		depth1 = np.array(depth,dtype=np.uint16)
		depth1_right = depth1 >> 3
		depth1_left = depth1 << 13
		depth1 = depth1_left | depth1_right
		
		if show_as_img == True:
			ij.setImage('depth8',depth1.astype(np.uint8))
			# ij.setImage('depth16',depth1.astype(np.float32))
			# Image.fromarray(depth1.astype(np.uint8)).show()
			# Image.fromarray(depth1).show()

		return depth1.astype(np.uint8)

	# renames the original dataset filenames from d-21313-0,d-2139213-1 to d0,d1
	# to read the files in the same order from depth and rgb folder.
	def rename_ptb_files(self,depth_path,rgb_path):

		for filename in glob.glob(depth_path+'/*'):
			new_name = re.sub('d-([0-9]|[a-z])+-','',filename)
			os.rename(filename, new_name)

		for filename in glob.glob(rgb_path+'/*'):
			new_name = re.sub('r-([0-9]|[a-z])+-','',filename)
			os.rename(filename, new_name)



	def parse_driving_dataset(self,dataset):

		# first we create the test file. And than the rest as training.

		print('Converting '+ dataset + '...')

		# with open('driving_test.csv','w') as f1, open('driving_train.csv','w') as f2:

			# test_writer=csv.writer(f1, delimiter=',',lineterminator='\n',)
			# train_writer=csv.writer(f2, delimiter=',',lineterminator='\n',)

		test_writer = self.init_tfrecord_writer(self.dataset_save+'driving_TEST.tfrecords')
		train_writer = self.init_tfrecord_writer(self.dataset_save+'driving_TRAIN.tfrecords')
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

								# row = [frames_finalpass_webp,
								# 	   frames_finalpass_webp2,
								# 	   optical_flow,
								# 	   disparity,
								# 	   disparity2,
								# 	   disparity_change]

								# if file_id > test_files:
								# 	train_writer.writerow(row)
								# else:
								# 	test_writer.writerow(row)

								patches = self.from_paths_to_data(
									disparity,
									disparity2,
									disparity_change,
									optical_flow,
									frames_finalpass_webp,
									frames_finalpass_webp2,
									camera_focal_length)




								# camera_L_R = self.get_frame_by_id(file_id - 1)
								camera_L_R ='1'

								if file_id > test_files:
									self.create_tf_example(patches,
										camera_L_R,
										train_writer,
										time)
								else:
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

	def parse_flyingthings3d_dataset(self,dataset):

		print('Converting '+ dataset + '...')
		path = ''

		test_writer = self.init_tfrecord_writer(self.dataset_save+'flying_TEST.tfrecords')
		train_writer = self.init_tfrecord_writer(self.dataset_save+'flying_TRAIN.tfrecords')

		for tnt in self.tnts:

			if tnt == self.tnts[0]:
				folders_range = self.flying_data_folder_train_limit[1]
			else:
				folders_range = self.flying_data_folder_test_limit[1]

			for data_type in self.data_types:
				for let in self.letters:


					for folder_id in range(0,folders_range):



						path = self.dataset_root + '/'.join([dataset,'camera_data',tnt,let,"%04d" % (folder_id,)])

						if os.path.isdir(path) == False:
							continue

						self.camera_data = self.load_camera_file(path)

						for direction in self.directions:
							for time in self.times:
								for file_id in range(self.flying_data_file_limit[0],self.flying_data_file_limit[1]):

									if file_id == self.flying_data_file_limit[1] - 1:
										break


									disparity_path = (path + '/' + direction).replace('camera_data','disparity') + '/' + str("%04d" % (file_id,)) + '.pfm'
									disparity_path2 = (path + '/' + direction).replace('camera_data','disparity') + '/' + str("%04d" % (file_id+1,)) + '.pfm'
									disparity_change_path = (path + '/' + time + '/' + direction).replace('camera_data','disparity_change') + '/' + str("%04d" % (file_id,)) + '.pfm'
									optical_flow_path = (path.replace('camera_data','optical_flow') + '/' + time + '/' + direction + '/' + self.get_optical_flow_file_name(direction,time,"%04d" % (file_id,))) + '.pfm'

									frames_finalpass_webp_path = (path + '/' + direction).replace('camera_data','frames_finalpass_webp') + '/' + str("%04d" % (file_id,)) + '.webp'
									frames_finalpass_webp_path2 = (path + '/' + direction).replace('camera_data','frames_finalpass_webp') + '/' + str("%04d" % (file_id+1,)) + '.webp'


									patches = self.from_paths_to_data(
										disparity_path,
										disparity_path2,
										disparity_change_path,
										optical_flow_path,
										frames_finalpass_webp_path,
										frames_finalpass_webp_path2,
										self.camera_focal_lengths[0])


									camera_L_R = self.get_frame_by_id(file_id - 6,True)

									if tnt == self.tnts[0]:
										self.create_tf_example(patches,
											camera_L_R,
											train_writer,
											time)
									else:
										self.create_tf_example(patches,
											camera_L_R,
											test_writer,
											time)

				# 					if self.global_check == self.break_mode:
				# 						break
				# 				if self.global_check == self.break_mode:
				# 					break

				# 			if self.global_check == self.break_mode:
				# 					break
				# 		if self.global_check == self.break_mode:
				# 				break
				# 	if self.global_check == self.break_mode:
				# 			break
				# if self.global_check == self.break_mode:
				# 		break


		self.close_writer(train_writer)
		self.close_writer(test_writer)

	def parse_monkaa_dataset(self,dataset):

		print('Converting '+ dataset + '...')

		test_writer = self.init_tfrecord_writer(self.dataset_save+'monkaa_TEST.tfrecords')
		train_writer = self.init_tfrecord_writer(self.dataset_save+'monkaa_TRAIN.tfrecords')

		test_examples_counter = 1
		for data_type in self.data_types:
			for scene in self.monkaa_scenes:
				path = self.dataset_root + '/'.join([dataset,'camera_data',scene])
				self.camera_data = self.load_camera_file(path)


				for direction in self.directions:
					for time in self.times:

						disparity_folder = (path + '/' + direction).replace('camera_data','disparity') + '/'
						disparity_change_folder = (path + '/' + time + '/' + direction).replace('camera_data','disparity_change') + '/'
						optical_flow_folder = path.replace('camera_data','optical_flow') + '/' + time + '/' + direction + '/'
						frames_finalpass_webp_path_folder = (path + '/' + direction).replace('camera_data','frames_finalpass_webp') + '/'

						files_count_in_this_scene = len([name for name in os.listdir(disparity_folder + '/')])


						# keep 30% of files as test files.
						test_files = math.ceil((files_count_in_this_scene * 30) / 100)

						for file_id in range(0,files_count_in_this_scene):

							if file_id == files_count_in_this_scene - 1:
								break

							disparity_path = disparity_folder + str("%04d" % (file_id,)) + '.pfm'
							disparity_path2 = disparity_folder + str("%04d" % (file_id+1,)) + '.pfm'
							disparity_change_path = disparity_change_folder + str("%04d" % (file_id,)) + '.pfm'
							optical_flow_path = optical_flow_folder + self.get_optical_flow_file_name(direction,time,"%04d" % (file_id,)) + '.pfm'
							frames_finalpass_webp_path = frames_finalpass_webp_path_folder + str("%04d" % (file_id,)) + '.webp'
							frames_finalpass_webp_path2 = frames_finalpass_webp_path_folder + str("%04d" % (file_id+1,)) + '.webp'


							patches = self.from_paths_to_data(
								disparity_path,
								disparity_path2,
								disparity_change_path,
								optical_flow_path,
								frames_finalpass_webp_path,
								frames_finalpass_webp_path2,
								self.camera_focal_lengths[0])



							camera_L_R = self.get_frame_by_id(file_id)

							if file_id > test_files:
								self.create_tf_example(patches,
									camera_L_R,
									train_writer,
									time)
							else:
								self.create_tf_example(patches,
									camera_L_R,
									test_writer,
									time)

 
		self.close_writer(train_writer)
		self.close_writer(test_writer)



	def create_tf_example(self,patches,camera_L_R,writer,scene_direction):

		# self.factor = 0.4
		# input_size = int(960 * self.factor), int(540 * self.factor) 
		for item in patches:




			# downsampled_opt_flow = self.downsample_labels(np.array(item['opt_fl']),2)
			# downsampled_disp_chng = self.downsample_labels(np.array(item['disp_chng']),0)

			width , height = item['depth1'].shape[0] , item['depth1'].shape[1]
			depth = item['depth1'].tostring()
			depth2 = item['depth2'].tostring()

			# np.savetxt('moja.txt',(item['optical_flow'][:,:,0] * input_size[0]))
			# Image.fromarray(item['optical_flow'][:,:,0] * input_size[0]).show()
			# Image.fromarray(item['optical_flow'][:,:,1] * input_size[1]).show()

			opt_flow = item['optical_flow'].tostring()
			# cam_frame_L = camera_L_R[0].tostring()
			# cam_frame_R = camera_L_R[1].tostring()
			depth_chng = item['depth_change'].tostring()
			frames_finalpass_webp = item['web_p'].tostring()
			frames_finalpass_webp2 = item['web_p2'].tostring()


			# if scene_direction == self.times[0]:
			# 	direction = b'f';
			# else:
			# 	direction = b'b';


			example = tf.train.Example(features=tf.train.Features(
				feature={
					'depth1': self._bytes_feature(depth),
					'depth2': self._bytes_feature(depth2),
					'depth_change': self._bytes_feature(depth_chng),
					'opt_flow': self._bytes_feature(opt_flow),
					'image1': self._bytes_feature(frames_finalpass_webp),
					'image2': self._bytes_feature(frames_finalpass_webp2)
			    }),
			)

			writer.write(example.SerializeToString())

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
		focal_length = 1050.0
		disp_to_depth = disparity / focal_length
		return disp_to_depth

	def get_depth_chng_from_disp_chng(self,disparity,disparity_change):
		disp2 = disparity + disparity_change

		depth1 = self.get_depth_from_disp(disparity)
		depth2 = self.get_depth_from_disp(disp2)

		return depth1 - depth2

	def normalizeOptFlow(self,flow,input_size):

		# remove the values bigger than the image size
		flow[:,:,0][flow[:,:,0] > input_size[0] ] = np.nan
		flow[:,:,1][flow[:,:,1] > input_size[1] ] = np.nan

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


	def check_max_depth(self,depth11,depth22,depth_changee):

		depth11 = np.max(depth11)
		depth22 = np.max(depth22)
		depth_changee = np.max(depth_changee)

		if depth11 > self.max_depth:
			self.max_depth = depth11
			print('max depth = '+str(depth11))

		if depth22 > self.max_depth2:
			self.max_depth2 = depth22
			print('max depth2 = '+str(depth22))

		if depth_changee > self.max_depth_chng:
			self.max_depth_chng = depth_changee
			print('max depth_chng = '+str(depth_changee))



	def check_normalized_depth(self,depth1,depth2,depth_change):
		depth_mx = np.amax(depth1) 
		depth2_mx = np.amax(depth2) 
		depth_change_mx = np.amax(depth_change)

		if depth_mx > 1:
			print('depth is bigger = ' + str(depth_mx))

		if depth2_mx > 1:
			print('depth2 is bigger = ' + str(depth2_mx))

		if depth_change_mx > 1:
			print('depth_change is bigger = ' + str(depth_change_mx))

	def check_nan_or_inf(self,depth1,depth2,depth_change):
		if np.isnan(depth1).any() or np.isinf(depth1).any():
			print('nan or inf in depth1')
		if np.isnan(depth2).any() or np.isinf(depth2).any():
			print('nan or inf in depth2')
		if np.isnan(depth_change).any() or np.isinf(depth_change).any():
			print('nan or inf in depth_change')
		self.check_normalized_depth(depth1,depth2,depth_change)


	# send in the disparity values, this will return the normalized inverse depth values.
	def get_resized_inverse_depth(self,disparity,disparity2,disparity_change,input_size):
		disparity = disparity.resize(input_size,Image.NEAREST)
		disparity2 = disparity2.resize(input_size,Image.NEAREST)
		disparity_change = disparity_change.resize(input_size,Image.NEAREST)

		disparity = np.array(disparity)
		disparity2 = np.array(disparity2)
		disparity_change = np.array(disparity_change)

		# since we resized, we need to change the values by the same factor as resize
		# disparity = disparity * self.factor
		# disparity2 = disparity2 * self.factor
		# disparity_change = disparity_change * self.factor

		# there are 0 values in disparity_change. We can add an epsilon to to shift the matrix.
		# disparity_change = disparity_change + 1e-6

		# convert disparities to depth
		depth1 = self.get_depth_from_disp(disparity)
		depth2 = self.get_depth_from_disp(disparity2)
		depth_change = self.get_depth_from_disp(disparity_change)
		# depth_change = self.get_depth_chng_from_disp_chng(disparity,disparity_change)


		# get inverse depth
		# depth1 = 1 / depth1
		# depth2 = 1 / depth2
		# depth_change = 1 / depth_change

		# write the max depth values and depth change values ( to normalzie the data with it )
		# self.check_max_depth(depth1,depth2,depth_change) 

		# normalize depth values

		# if self.dataset_number == 1:
		# 	depth1,depth2,depth_change = self.get_normalized_driving_depths(depth1,depth2,depth_change)
		# elif self.dataset_number == 2:
		# 	depth1,depth2,depth_change = self.get_normalized_flying_depths(depth1,depth2,depth_change)
		# else:
		# 	depth1,depth2,depth_change = self.get_normalized_monkaa_depths(depth1,depth2,depth_change)

		return depth1,depth2,depth_change


	def get_normalized_flying_depths(self,depth1,depth2,depth_change):
		depth1 = depth1 / self.max_depth_flying
		depth2 = depth2 / self.max_depth_flying
		depth_change = depth_change / self.max_depth_flying_chng

		return depth1, depth2, depth_change

	def get_normalized_driving_depths(self,depth1,depth2,depth_change):
		depth1 = depth1 / self.max_depth_driving
		depth2 = depth2 / self.max_depth_driving

		depth_change = depth_change / self.max_depth_driving_chng

		return depth1, depth2, depth_change


	def get_normalized_monkaa_depths(self,depth1,depth2,depth_change):
		depth1 = depth1 / self.max_depth_monkaa
		depth2 = depth2 / self.max_depth_monkaa
		depth_change = depth_change / self.max_depth_monkaa_chng

		return depth1, depth2, depth_change


	# 	return patches
	def downsample_opt_flow(self,data,size):
		data = np.delete(data,2,axis=2)

		u = data[:,:,0]
		v = data[:,:,1]
		
		dt = Image.fromarray(u,mode='F')
		dt = dt.resize(size, Image.NEAREST)

		dt2 = Image.fromarray(v,mode='F')
		dt2 = dt2.resize(size, Image.NEAREST)

		u = np.array(dt) * self.u_factor
		v = np.array(dt2) * self.v_factor

		opt_flow = np.stack((u,v),axis=2)

		# reduce the flow values by same factor as image sizes are reduced by
		# opt_flow = opt_flow * self.factor

		# normalize flow values between 0 - 1
		# opt_flow = self.normalizeOptFlow(opt_flow,size)

		return opt_flow

	def from_paths_to_data(self,
						   disparity,
						   disparity2,
						   disparity_change,
						   optical_flow,
						   frames_finalpass_webp_path,
						   frames_finalpass_webp_path2,
						   focal_length):

		# reduce the sizes of images and flow values by this factor

		# for size 128x128
		# self.u_factor = 0.237037037
		# self.v_factor = 0.133333333



		self.u_factor = 0.414814815
		self.v_factor = 0.4

		input_size = math.ceil(960 * self.v_factor), math.floor(540 * self.u_factor)

		# parse pfm files for disparities
		disparity  = self.readPFM(disparity)[0]
		disparity2  = self.readPFM(disparity2)[0]

		disparity_change = self.readPFM(disparity_change)[0]

		opt_flow = self.readPFM(optical_flow)[0]

		# reduce optical flow size
		opt_flow_full = self.downsample_opt_flow(opt_flow,input_size)

		web_p_file = Image.open(frames_finalpass_webp_path)
		web_p_file2 = Image.open(frames_finalpass_webp_path2)

		web_p_file = web_p_file.resize(input_size, Image.BILINEAR)
		web_p_file2 = web_p_file2.resize(input_size, Image.BILINEAR)


		disparity = Image.fromarray(disparity)
		disparity2 = Image.fromarray(disparity2)
		disparity_change = Image.fromarray(disparity_change)

		depth1,depth2,depth_change = self.get_resized_inverse_depth(disparity,disparity2,disparity_change,input_size)

		# self.check_nan_or_inf(depth1,depth2,depth_change)
		# self.check_normalized_depth(depth1,depth2,depth_change)

		web_p_file = np.array(web_p_file)[:,:,0:3]
		web_p_file2 = np.array(web_p_file2)[:,:,0:3]

		return [{
			'web_p': web_p_file,
			'web_p2': web_p_file2,
			'depth1': depth1,
			'depth2': depth2,
			'depth_change': depth_change,
			'optical_flow': opt_flow_full,
			'path': optical_flow
		}]




	def convert(self):
		if self.dataset_number == 1:
			self.parse_driving_dataset(self.datasets[0])
		elif self.dataset_number == 2:
			self.parse_flyingthings3d_dataset(self.datasets[1])
		elif self.dataset_number == 3:
			self.parse_monkaa_dataset(self.datasets[2])
		else:
			self.parse_ptb_dataset(self.datasets[3])
	
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

		if self.dataset_number == 1:
			print('driving finished')

		elif self.dataset_number == 2:
			print('flying finished')
		else:
			print('monkaa finished')

		print('values in between = '+str(self.values_between_highest_and_lowest))

	# value: the value needed to be converted to feature
	def _bytes_feature(self,value):
	    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

	def _int64_feature(self,value):
	    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
	  
	def init_tfrecord_writer(self,filename):
		return tf.python_io.TFRecordWriter(filename)


def convert_whole_dataset():
	a = SyntheticTFRecordsWriter()
	a.convert()

def convert_for_testing():
	return SyntheticTFRecordsWriter()

convert_whole_dataset()