import tensorflow as tf
import numpy as np


class DatasetReader:

	def __init__(self,tf_file_path = 'optical_flow.tfrecords'):
		record_iterator = tf.python_io.tf_record_iterator(path=tf_file_path)
		self.iterate(record_iterator)



	def iterate(self,iterator):

			index = 0
			for string_record in iterator:
				print('iteration '+str(index))
				index = index + 1
				example = tf.train.Example()
				example.ParseFromString(string_record)

				rows = int(example.features.feature['rows'].int64_list.value[0])
				img_pair = (example.features.feature['img_pair'].bytes_list.value[0])
				flow = (example.features.feature['flow'].bytes_list.value[0])
			
			# original image
			img_pair_orig = np.fromstring(img_pair, dtype=np.uint8)

			# reshape images to original form
			img_reshaped_1 = img_pair_orig.reshape((rows,6))

			print(img_reshaped_1)
			