import tensorflow as tf
import numpy as np
import network
import tensorflow.contrib.slim as slim


class DatasetReader:
    def iterate(self, filenames):
        with tf.Session() as sess:
            # Create a list of filenames and pass it to a queue
            filename_queue = tf.train.string_input_producer(filenames)
            # Define a reader and read the next record
            recordReader = tf.TFRecordReader()

            key, fullExample = recordReader.read(filename_queue)

            # Decode the record read by the reader
            features = tf.parse_single_example(fullExample, {
                'width': tf.FixedLenFeature([], tf.int64),
                'height': tf.FixedLenFeature([], tf.int64),
                'img_pair': tf.FixedLenFeature([], tf.string),
                'flow': tf.FixedLenFeature([], tf.string)
            })

            # Convert the image data from binary back to arrays(Tensors)
            image = tf.decode_raw(features['img_pair'], tf.uint8)
            label = tf.decode_raw(features['flow'], tf.float32)
            width = tf.cast(features['width'], tf.int32)
            height = tf.cast(features['height'], tf.int32)

            # reshape data to its original form
            image = tf.reshape(image, [480, 640, 6])
            label = tf.reshape(label, [480, 640, 2])

            # shuffle the data and get them as batches
            images, labels = tf.train.shuffle_batch([image, label],
                                                    batch_size=32,
                                                    capacity=100,
                                                    num_threads=1,
                                                    min_after_dequeue=10)

            # our image is 480x640 and each pixel has values RGBRGB i.e 6 channels.
            placeholder_image_pair = tf.placeholder(dtype=tf.float32, shape=(1, 480, 640, 6))

            Y = tf.placeholder(dtype=tf.float32, shape=(1, 120, 160, 2))

            print('ala11')
            # build the network here
            predict_flow5, predict_flow2 = network.train_network(placeholder_image_pair)
            print(predict_flow2.get_shape())
            # predict_flow2 = np.random.rand(120,160,2).astype(np.float32)
            print('ala12')
            # # measure of error of our model
            # # this needs to be minimised by adjusting W and b
            mse = tf.reduce_mean(tf.squared_difference(predict_flow2, Y))

            print('ala13')
            # # define training step which minimizes cross entropy
            optimizer = tf.train.AdamOptimizer(1e-4).minimize(mse)

            print('ala14')
            # initialize the variables
            sess.run(tf.global_variables_initializer())

            print('ala15')
            # initialize the threads coordinator
            coord = tf.train.Coordinator()
            print('ala16')

            # start enqueing the data to be dequeued for batch training
            threads = tf.train.start_queue_runners(sess, coord=coord)
            print('ala')
            # start passing the data to the feed_dict here and running everything after initializing the variables
            # Retrieve a single instance:
            img, lbl = sess.run([images, labels])
            print('ala2')
            sess.run(optimizer, feed_dict={X: img, Y: lbl})
            print('ala3')

            # finalise
            coord.request_stop()
            coord.join(threads)
