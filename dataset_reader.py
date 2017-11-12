import tensorflow as tf
import numpy as np
import network

class DatasetReader:


	def iterate(self,filenames):



		with tf.Session() as sess:

			# Create a list of filenames and pass it to a queue
			filename_queue = tf.train.string_input_producer(filenames)
			# Define a reader and read the next record
			recordReader = tf.TFRecordReader()

			key, fullExample = recordReader.read(filename_queue)

			# Decode the record read by the reader
			features = tf.parse_single_example(fullExample,{
				'width': tf.FixedLenFeature([], tf.int64),
				'height': tf.FixedLenFeature([], tf.int64),
				'img_pair': tf.FixedLenFeature([], tf.string),
				'flow': tf.FixedLenFeature([], tf.string)
			})

			# Convert the image data from binary back to arrays(Tensors)
			image = tf.decode_raw(features['img_pair'],tf.uint8)
			label = tf.decode_raw(features['flow'], tf.float32)
			width = tf.cast(features['width'], tf.int32)
			height = tf.cast(features['height'], tf.int32)

			# reshape data to its original form
			image = tf.reshape(image,[480,640,6])
			label = tf.reshape(label, [480,640,2])

			# shuffle the data and get them as batches
			images, labels = tf.train.shuffle_batch([image, label],
													batch_size=2, 
													capacity=20, 
													num_threads=1, 
													min_after_dequeue=10)



			# our image is 480x640 and each pixel has values RGBRGB i.e 6 channels.
			placeholder_image_pair = tf.placeholder(dtype=tf.float32, shape=(1,480,640,6))

			# build the network here
			network.train_network(placeholder_image_pair)

			# initialize the threads coordinator
			coord = tf.train.Coordinator()

			# start enqueing the data to be dequeued for batch training
			threads = tf.train.start_queue_runners(sess, coord=coord)


			# start passing the data to the feed_dict here and running everything after initializing the variables			

			# finalise 
			coord.request_stop()
			coord.join(threads)