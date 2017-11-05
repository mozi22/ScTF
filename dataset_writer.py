
import tensorflow as tf
import numpy as np
from PIL import Image

class DatasetWriter:


	# import_option_type: specifies if the labels  will be present in the same folder or in a different folder
	#					  can have only 2 values: SAME or DIFF

	# root_folder: root folder in which the dataset is present
	# labels_folder: if import_option_type == DIFF than this will specify the folder path

	def __init__(self,import_option_type, root_folder , labels_folder = None):

		self.TFRecordFileName = "optical_flow.tfrecords"

		self.import_option_type = import_option_type

		if self.import_option_type == 'DIFF':
			self.labels_folder = labels_folder


		self.root_folder = root_folder
		self.writer = tf.python_io.TFRecordWriter(self.TFRecordFileName)


	def traverse_subfolders(self,root_folder):
		print('muazzam')

	# image1: path to first image of the pair
	# image2: path to second image of the pair
	# gt_flow: the .flo file np array, representing the ground truth
	def convert_file(self,image1,image2,gt_flow):
		img1 = np.array(Image.open(image1))
		img2 = np.array(Image.open(image2))


		# image sizes for both are expected to be the same
		flattened_pixels = img1.shape[0]*img1.shape[1]


		# reshape the image to nx3 where 3 are the rgb values for pixel at n
		flat_img1 = img1.reshape((flattened_pixels,3))
		flat_img2 = img2.reshape((flattened_pixels,3))

		print(flat_img1[2])

		img_raw_1 = flat_img1.tostring()
		img_raw_2 = flat_img2.tostring()
		gt_flow = gt_flow.tostring()

		example = tf.train.Example(features=tf.train.Features(
			feature={

				# we already know that there will be 3 columns (R-G-B). Hence we just keep track of the 
				# number of rows, which is width * height of the image.
			    'rows': self._int64_feature(flattened_pixels),
			    'img1': self._bytes_feature(img_raw_1),
			    'img2': self._bytes_feature(img_raw_2),
			    'flow': self._bytes_feature(gt_flow)
		    }),
		)

		self.writer.write(example.SerializeToString())

	# file_path: path to the flo file
	def read_flo_file(self,file_path):
		with open(file_path, 'rb') as f:

			magic = np.fromfile(f, np.float32, count=1)

			if 202021.25 != magic:
				print('Magic number incorrect. Invalid .flo file')
			else:
				w = np.fromfile(f, np.int32, count=1)[0]
				h = np.fromfile(f, np.int32, count=1)[0]

				data = np.fromfile(f, np.float32, count=2*w*h)

				# Reshape data into 3D array (columns, rows, bands)
				data2D = np.resize(data, (w, h, 2))
				return data2D

	# close the file writer
	def close_writer(self):
		self.writer.close()


	# value: the value needed to be converted to feature
	def _bytes_feature(self,value):
	    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

	def _int64_feature(self,value):
	    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

	def _float_feature(self,value):
	    return tf.train.Feature(int64_list=tf.train.FloatList(value=[value]))
