
import tensorflow as tf
import numpy as np
from PIL import Image

class TFRecordsConverter:


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
	# gt_flow: the .flo file representing the ground truth
	def convert_file(self,image1,image2,gt_flow):
		img1 = np.array(Image.open(image1))
		img2 = np.array(Image.open(image2))

		height = img1.shape[0]
		width = img1.shape[1]


		img_raw_1 = img1.tostring()
		img_raw_2 = img2.tostring()

		example = tf.train.Example(features=tf.train.Features(
			feature={
			    'height': self._int64_feature(height),
			    'width': self._int64_feature(width),
			    'image_1': self._bytes_feature(img_raw_1),
			    'image_2': self._bytes_feature(img_raw_2),
			    'flow': self._float_feature(gt_flow)
		    }),
		)

		self.writer.write(example.SerializeToString())


	def close_writer(self):
		self.writer.close()


	def _bytes_feature(self,value):
	    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

	def _int64_feature(self,value):
	    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

	def _float_feature(self,value):
	    return tf.train.Feature(int64_list=tf.train.FloatList(value=[value]))
