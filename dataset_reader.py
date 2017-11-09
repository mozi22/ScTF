import tensorflow as tf
import numpy as np
import network

class DatasetReader:


	def iterate(self,filenames):



		with tf.Session() as sess:

			# Create a list of filenames and pass it to a queue
			filename_queue = tf.train.string_input_producer(filenames, num_epochs=1)
			# Define a reader and read the next record
			recordReader = tf.TFRecordReader()

			key, fullExample = recordReader.read(filename_queue)

			# Decode the record read by the reader
			features = tf.parse_single_example(fullExample,{
				'rows': tf.FixedLenFeature([], tf.int64),
				'img_pair': tf.FixedLenFeature([], tf.string),
				'flow': tf.FixedLenFeature([], tf.string)
			})

			# Convert the image data from string back to the numbers
			image = tf.decode_raw(features['img_pair'],tf.uint8)
			label = tf.decode_raw(features['flow'], tf.uint8)
			rows = tf.cast(features['rows'], tf.int32)

			# print(image.eval())
			# rows = int(fullExample.features.feature['rows'].int64_list.value[0])
			# image = (fullExample.features.feature['img_pair'].bytes_list.value[0])
			# flow = (fullExample.features.feature['flow'].bytes_list.value[0])



			init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
			sess.run(init_op)
			coord = tf.train.Coordinator()
			threads = tf.train.start_queue_runners(coord=coord)


			value = rows.eval()
			image = tf.reshape(image,[value,6])
			label = tf.reshape(label, [value,2])

			images, labels = tf.train.shuffle_batch([image, label], batch_size=4, capacity=20, num_threads=1, min_after_dequeue=10)

			# print(images.eval())

			# # print(value)
			# image.set_shape(5)
			# print(label.eval())

			coord.request_stop()
			coord.join(threads)

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