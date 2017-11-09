import tensorflow as tf
import numpy as np
import network

class DatasetReader:


	# starts the queue runner to enqueue values inside our window
	def start_queue_runner(self,sess):
		self.coord = tf.train.Coordinator()
		self.threads = tf.train.start_queue_runners(sess=sess,coord=self.coord)

	# stop the queue runner to stop training process
	def stop_queue_runner(self):
		self.coord.request_stop()
		self.coord.join(self.threads)


	def open_session(self):
		return tf.Session()




	def build_input_pipeline(self,filenames):

			session = self.open_session()
			self.start_queue_runner(session)

			# Create a list of filenames and pass it to a queue
			filename_queue = tf.train.string_input_producer(filenames, num_epochs=1)

			# Define a reader and read the next record
			recordReader = tf.TFRecordReader()

			# read a file from the filename_queue
			key, fullExample = recordReader.read(filename_queue)

			# Decode an example record we wrote inside the file.
			features = tf.parse_single_example(fullExample,{
				'rows': tf.FixedLenFeature([], tf.int64),
				'img_pair': tf.FixedLenFeature([], tf.string),
				'flow': tf.FixedLenFeature([], tf.string)
			})

			# Convert the image_pair data in nx6 form, from string back to the matrix
			image = tf.decode_raw(features['img_pair'],tf.uint8)
			label = tf.decode_raw(features['flow'], tf.uint8)
			rows = tf.cast(features['rows'], tf.int32)

			# get the total number of rows we of the file we saved data into.
			value = rows.eval()


			image.set_shape([value])

			print(image.eval())

			self.stop_queue_runner()
		# rows = int(fullExample.features.feature['rows'].int64_list.value[0])
		# image = (fullExample.features.feature['img_pair'].bytes_list.value[0])
		# flow = (fullExample.features.feature['flow'].bytes_list.value[0])


		# image = tf.reshape(image, [rows,6])

		# init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
		# sess.run(init_op)

		# print(value)

		# images, labels = tf.train.shuffle_batch([image, label], batch_size=10, capacity=30, num_threads=1, min_after_dequeue=10)

		# init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
		# sess.run(init_op)
		# coord = tf.train.Coordinator()
		# threads = tf.train.start_queue_runners(coord=coord)
		# try:
		# 	while not coord.should_stop():
		# 		X, y = sess.run([images, label])
		# except Exception as e:
		# 	coord.request_stop(e)
		# finally:
		# 	coord.request_stop()
		# 	coord.join(threads)
	    # sess.run(image)

	    # # Any preprocessing here ...
	    
	    # # Creates batches by randomly shuffling tensors
	    # images, labels = tf.train.shuffle_batch([image, label], batch_size=10, capacity=30, num_threads=1, min_after_dequeue=10)
		# index = 0
		# for string_record in iterator:
		# 	index = index + 1
		# 	example = tf.train.Example()
		# 	example.ParseFromString(string_record)

		# 	rows = int(example.features.feature['rows'].int64_list.value[0])
		# 	img_pair = (example.features.feature['img_pair'].bytes_list.value[0])
		# 	flow = (example.features.feature['flow'].bytes_list.value[0])
			
		# 	# original image
		# 	img_pair_orig = np.fromstring(img_pair, dtype=np.uint8)

		# 	# reshape images to original form
		# 	img_reshaped_1 = img_pair_orig.reshape((rows,6))

		# 	network.train_network(img_pair_orig)