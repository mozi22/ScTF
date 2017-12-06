



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

		self.datasets = ['driving','flyingthings3d','monkaa']
		
		self.directions = ['left','right']

		self.times = [
			'into_future',
			'into_past'
		]

		self.data_types = [ 
			'disparity_change',
			'disparity',
			'optical_flow',
			'frames_finalpass_webp',
			'camera_data'
		]

		self.scene_types = [
			'scene_backwards',
			'scene_forwards'
		]

		self.camera_speeds = ['fast']
		self.camera_focal_lengths = ['35mm_focallength']


		self.total_files = 300

		self.dataset_root = '/home/muazzam/mywork/python/thesis/danda/'


	def generate_file_paths(self,id):

		# generate camera file path
		camera_file_path = self.dataset_root + ''


	def convert(self):


		for file_id in range(self.total_files):
			for dataset in self.datasets:
				for data_type in self.data_types:
					for camera_focal_length in self.self.camera_focal_lengths:
						for scene_type in self.scene_types:
							for camera_speed in self.camera_speeds:
								for time in self.times:
									for direction in self.directions:
										



			self.generate_file_paths(file_id)




	# reads the PFM file and returns an np matrix.
	def readPFM(file):
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