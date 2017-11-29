import tensorflow as tf
import numpy as np
import network
import tensorflow.contrib.slim as slim
from PIL import Image


class DatasetReader:



    def decode_features(self,fullExample):    
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
        image = tf.reshape(image, [96, 128, 6])
        label = tf.reshape(label, [24, 32, 2])

        return image, label, width, height

    def get_image_for_testing(self):
        img1 = np.array(Image.open('./dataset/training/Grove2/frame10.png'))
        img2 = np.array(Image.open('./dataset/training/Grove2/frame11.png'))

        img_pair = np.concatenate((img1,img2),axis=-1)

        return img_pair[np.newaxis,:]



    def iterate(self, filenames,feed_for_train = True):
    

            if feed_for_train == False:
                img_pair = self.get_image_for_testing()



            with tf.name_scope('input_pipeline'):
                # Create a list of filenames and pass it to a queue
                filename_queue = tf.train.string_input_producer(filenames)
                # Define a reader and read the next record
                recordReader = tf.TFRecordReader()

                key, fullExample = recordReader.read(filename_queue)

                image, label, width, height = self.decode_features(fullExample)

                # shuffle the data and get them as batches
                self.imageBatch, self.labelBatch = tf.train.shuffle_batch([image, label],
                                                        batch_size=4,
                                                        capacity=100,
                                                        num_threads=1,
                                                        min_after_dequeue=6)


            with tf.name_scope('create_graph'):

                if feed_for_train == False:
                    # our image is 480x640 and each pixel has values RGBRGB i.e 6 channels.
                    self.X = tf.placeholder(dtype=tf.float32, shape=(1, 96, 128, 6))
                    self.Y = tf.placeholder(dtype=tf.float32, shape=(1, 24, 32, 2))
                else:
                    self.X = tf.placeholder(dtype=tf.float32, shape=(4, 96, 128, 6))
                    self.Y = tf.placeholder(dtype=tf.float32, shape=(4, 24, 32, 2))

                # build the network here
                predict_flow5, predict_flow2 = network.train_network(self.X)
                # # measure of error of our model
                # # this needs to be minimised by adjusting W and b
                mse = tf.reduce_mean(tf.squared_difference(predict_flow2, self.Y))

                # # define training step which minimizes cross entropy
                self.optimizer = tf.train.AdamOptimizer(1e-4).minimize(mse)


            sess = tf.InteractiveSession()

            if feed_for_train:
                self.train_network(sess)
                self.save_model(sess)
            else:
                self.load_model_ckpt(sess)

                # feed the image pair and make prediction.
                feed_dict = {
                    self.X: img_pair,
                }

                v = sess.run({'prediction': predict_flow2},feed_dict=feed_dict)
                print(v['prediction'].shape)




    def train_network(self,sess):
        merged_summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter('./tb',graph=sess.graph)

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        # initialize the threads coordinator
        coord = tf.train.Coordinator()
        # start enqueing the data to be dequeued for batch training
        threads = tf.train.start_queue_runners(sess, coord=coord)
        # start passing the data to the feed_dict here and running everything after initializing the variables
        # Retrieve a single instance:
        batch_xs, batch_ys = sess.run([self.imageBatch, self.labelBatch])

        sess.run(self.optimizer,feed_dict={self.X: batch_xs, self.Y: batch_ys})


        summary_str = sess.run(merged_summary_op, feed_dict={self.X: batch_xs, self.Y: batch_ys})
        summary_writer.add_summary(summary_str, 1)


        # finalise
        coord.request_stop()
        coord.join(threads)

 
    def save_model(self,sess):
        saver = tf.train.Saver()
        saver.save(sess, 'muazzam/my_test_model.ckpt')


    def load_model_ckpt(self,sess):
        saver = tf.train.Saver()
        saver.restore(sess,'muazzam/my_test_model.ckpt')



        