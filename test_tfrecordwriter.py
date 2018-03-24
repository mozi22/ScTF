import tensorflow as tf
import numpy as np
# def create_tf_example(writer,item):

# 	# downsampled_opt_flow = self.downsample_labels(np.array(item['opt_fl']),2)
# 	# downsampled_disp_chng = self.downsample_labels(np.array(item['disp_chng']),0)
# 	item = item.tostring()

# 	example = tf.train.Example(features=tf.train.Features(
# 		feature={
# 			'item': _bytes_feature(item),
# 	    }),
# 	)

# 	writer.write(example.SerializeToString())


# # value: the value needed to be converted to feature
# def _bytes_feature(value):
#     return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

 
# def init_tfrecord_writer(filename):
# 	return tf.python_io.TFRecordWriter(filename)





# file_handle = init_tfrecord_writer('tester.tfrecords')

# img_1 = np.array([[[1,2],[5,6]],[[9,10],[13,14]]],dtype=np.float32)
# img_2 = np.array([[[3,4],[7,8]],[[11,12],[15,16]]],dtype=np.float32)
# img_3 = np.array([[[100,200],[500,600]],[[900,1000],[1300,1400]]],dtype=np.float32)
# img_4 = np.array([[[300,400],[700,800]],[[1100,1200],[1500,1600]]],dtype=np.float32)

# create_tf_example(file_handle,img_1)
# create_tf_example(file_handle,img_2)
# create_tf_example(file_handle,img_3)
# create_tf_example(file_handle,img_4)
# file_handle.close()

def read():
	filename_queue = tf.train.string_input_producer(['tester.tfrecords'])
	# Define a reader and read the next record
	recordReader = tf.TFRecordReader(name="TfReaderV"+'1')

	key, fullExample = recordReader.read(filename_queue)



	# Decode the record read by the reader
	features = tf.parse_single_example(fullExample, {
		'item': tf.FixedLenFeature([], tf.string)
	},
	name="ExampleParserV"+'1')

	item = tf.decode_raw(features['item'], tf.float32)
	item = tf.reshape(item, [ 2,2,2 ],name="reshape_img1")

	img_2 = np.array([[[3,4],[7,8]],[[11,12],[15,16]]],dtype=np.float32)

	item1 = tf.concat([item,img_2],axis=-1)
	item2 = tf.concat([img_2,item],axis=-1)

	print(item1)
	print(item2)
	result = tf.stack([item1,item2])

	test(result)


def test(item):
	sess = tf.InteractiveSession()
	label = tf.zeros(item.get_shape())

	images, labels = tf.train.batch(
								[ item , label ],
								batch_size=1,
								capacity=100,
								num_threads=48,
								# min_after_dequeue=1,
								enqueue_many=False)

	sess.run(tf.global_variables_initializer())
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(sess=sess,coord=coord)

	print('ranaola')
	print(images)
	print(sess.run(images))



	coord.request_stop()
	coord.join(threads)


read()