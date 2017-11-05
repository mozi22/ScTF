import tensorflow as tf
import numpy as np


class DatasetReader:

	def __init__(self,tf_file_path = 'optical_flow.tfrecords'):
		record_iterator = tf.python_io.tf_record_iterator(path=tf_file_path)
		self.iterate(record_iterator)



	def iterate(self,iterator):

			for string_record in iterator:

				example = tf.train.Example()
				example.ParseFromString(string_record)

				rows = int(example.features.feature['rows'].int64_list.value[0])
				img1 = (example.features.feature['img1'].bytes_list.value[0])
				img2 = (example.features.feature['img2'].bytes_list.value[0])
				flow = (example.features.feature['flow'].bytes_list.value[0])
			
			# original image
			img_orig_1 = np.fromstring(img1, dtype=np.uint8)
			img_orig_2 = np.fromstring(img2, dtype=np.uint8)

			# reshape images to original form
			img_reshaped_1 = img_orig_1.reshape((rows,3))
			img_reshaped_2 = img_orig_2.reshape((rows,3))
			
			print(img_reshaped_1[2])