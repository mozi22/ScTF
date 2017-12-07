



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





class SyntheticTFRecordsWriter:


	def __init__(self):

		# these params need to be updated when running on the bigger dataset

		# here 2 means 0000 and 0001 i.e 2 records
		self.flyingdata_TEST_FOLDERS_IDS = 2
		self.flyingdata_TRAIN_FOLDERS_IDS = 2
		self.flyingdata_FILES_IDS = [6,8]
		self.driving_FILES = 3
		self.monkaa_FILES = 2


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

		return [left,right]


	def get_path_based_on_module(self,path_so_far,module,time,direction):

		if module == self.data_types[0]:
			disparity_path = path_so_far + '/' + direction



	def parse_driving_dataset(self,dataset):
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

								print(disparity_path)
								print(disparity_change_path)
								print(optical_flow_path)
								print(frames_finalpass_webp_path)

								# since we've 300 files for fast folders
								# for file_id in range(1,self.driving_FILES):

								# this is the path from where we read all the files. it could belong to one of the
								# other 4 modules leaving out camera_data since we already read it above.
								# final_folder_path = self.get_path_based_on_module(path,dataset,time,direction)


	def parse_flyingthings3d_dataset(self,dataset):

		path = ''
		for data_type in self.data_types:
			for tnt in self.tnts:
				for let in self.letters:

					if tnt == self.tnts[0]:
						folders_range = self.flying_data_folder_train_limit[1]
					else:
						folders_range = self.flying_data_folder_test_limit[1]

					for folder_id in range(0,folders_range):

						path = self.dataset_root + '/'.join([dataset,'camera_data',tnt,let,"%04d" % (folder_id,)])
						self.camera_data = self.load_camera_file(path)

						for direction in self.directions:						
							for time in self.times:
								for file_id in range(self.flying_data_file_limit[0],self.flying_data_file_limit[1]):

									disparity_path = (path + '/' + direction).replace('camera_data','disparity')
									disparity_change_path = (path + '/' + time + '/' + direction).replace('camera_data','disparity_change')
									optical_flow_path = (path + '/' + time + '/' + direction).replace('camera_data','optical_flow')
									frames_finalpass_webp_path = (path + '/' + direction).replace('camera_data','frames_finalpass_webp')

									print(disparity_path)
									print(disparity_change_path)
									print(optical_flow_path)
									print(frames_finalpass_webp_path)



	def parse_monkaa_dataset(self,dataset):
		path = ''

		for data_type in self.data_types:
			for scene in self.monkaa_scenes:
				path = self.dataset_root + '/'.join([dataset,'camera_data',scene])
				self.camera_data = self.load_camera_file(path)


				for direction in self.directions:
					for time in self.times:
						for file in range(0,self.monkaa_FILES):
							disparity_path = (path + '/' + direction).replace('camera_data','disparity')
							disparity_change_path = (path + '/' + time + '/' + direction).replace('camera_data','disparity_change')
							optical_flow_path = (path + '/' + time + '/' + direction).replace('camera_data','optical_flow')
							frames_finalpass_webp_path = (path + '/' + direction).replace('camera_data','frames_finalpass_webp')

							print(disparity_path)
							print(disparity_change_path)
							print(optical_flow_path)
							print(frames_finalpass_webp_path)





	def convert(self):

		path = ''

		# for dataset in self.datasets:
		# 	if dataset == self.datasets[0]:
		# 		self.parse_driving_dataset(dataset)
		# 	elif dataset == self.datasets[1]:
		# self.parse_flyingthings3d_dataset('flyingthings3d')
			# else:
				# monkaa
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



a = SyntheticTFRecordsWriter()
a.convert()