import io
import network
import numpy as np
from   PIL import Image
import tensorflow as tf
import tensorflow.contrib.slim as slim

class DatasetReader:


    def decode_features(self,fullExample):    
        # Decode the record read by the reader
        features = tf.parse_single_example(fullExample, {
            'width': tf.FixedLenFeature([], tf.int64),
            'height': tf.FixedLenFeature([], tf.int64),
            'disp': tf.FixedLenFeature([], tf.string),
            'opt_flow': tf.FixedLenFeature([], tf.string),
            'cam_frame_L': tf.FixedLenFeature([], tf.string),
            'cam_frame_R': tf.FixedLenFeature([], tf.string),
            'image': tf.FixedLenFeature([], tf.string),
            'disp_chng': tf.FixedLenFeature([], tf.string),
            'direction': tf.FixedLenFeature([], tf.string)
        })

        # Convert the image data from binary back to arrays(Tensors)
        # disp_width = tf.cast(features['width'], tf.int32)
        # disp_height = tf.cast(features['height'], tf.int32)



        direction = features['direction']
        disp = tf.decode_raw(features['disp'], tf.float32)
        image = tf.decode_raw(features['image'], tf.uint8)
        opt_flow = tf.decode_raw(features['opt_flow'], tf.float32)
        disp_chng = tf.decode_raw(features['disp_chng'], tf.float32)
        cam_frame_L = tf.decode_raw(features['cam_frame_L'], tf.float32)
        cam_frame_R = tf.decode_raw(features['cam_frame_R'], tf.float32)

        input_dimensions = [54,96]
        lbl_dimensions = [27,48]


        image = tf.to_float(image)
        # reshape data to its original form
        image = tf.reshape(image, [input_dimensions[0],input_dimensions[1], 3])
        disp = tf.reshape(disp, [input_dimensions[0],input_dimensions[1]])
        opt_flow = tf.reshape(opt_flow, [lbl_dimensions[0],lbl_dimensions[1],2])
        disp_chng = tf.reshape(disp_chng,[lbl_dimensions[0],lbl_dimensions[1]])
        depth,depth_change = self.get_depth_from_disparity(disp,disp_chng)

        # return opt_flow

        inputt = self.combine_depth_values(image,depth,2)
        label = self.combine_depth_values(opt_flow,depth_change,2)

        padding_input = tf.constant([[5, 5],[0, 0],[0, 0]])
        padding_lbl = tf.constant([[2, 3],[0, 0],[0, 0]])

        inputt = tf.pad(inputt,padding_input,'CONSTANT')
        label = tf.pad(label,padding_lbl,'CONSTANT')

        # return inputt,label

        return {
            'input': inputt,
            'label': label
        }

    # combines the depth value in the image RGB values to make it an RGBD tensor.
    # where the resulting tensor will have depth values in the 4th element of 3rd dimension i.e [0][0][3].
    # where [x][x][0] = R, [x][x][1] = G, [x][x][2] = B, [x][x][3] = D
    def combine_depth_values(self,image,depth,rank):
        depth = tf.expand_dims(depth,rank)
        return tf.concat([image,depth],rank)


    def get_depth_from_disparity(self,disparity,disparity_change):
        # focal_length = 35
        # baseline = 1
        # focal_length * baseline = 35

        focal_length = tf.constant([35],dtype=tf.float32)

        depth = tf.divide(focal_length,disparity)
        depth_change = tf.divide(focal_length,disparity_change)

        return depth,depth_change


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

                features = self.decode_features(fullExample)

                # shuffle the data and get them as batches
                self.imageBatch, self.labelBatch = tf.train.shuffle_batch([ features['input'], 
                                                                            features['label']],
                                                        batch_size=4,
                                                        capacity=100,
                                                        num_threads=1,
                                                        min_after_dequeue=6)


            with tf.name_scope('create_graph'):

                if feed_for_train == False:
                    # our image is 480x640 and each pixel has values RGBRGB i.e 6 channels.
                    self.X = tf.placeholder(dtype=tf.float32, shape=(1, 80, 128, 4))
                    self.Y = tf.placeholder(dtype=tf.float32, shape=(1, 40, 64, 4))
                else:
                    self.X = tf.placeholder(dtype=tf.float32, shape=(1, 64, 96, 4))
                    self.Y = tf.placeholder(dtype=tf.float32, shape=(1, 32, 48, 3))

                # build the network here
                predict_flow5, predict_flow2 = network.train_network(self.X)

                # measure of error of our model
                # this needs to be minimised by adjusting W and b
                mse = tf.reduce_mean(tf.squared_difference(predict_flow2, self.Y))

                # # # define training step which minimizes cross entropy
                self.optimizer = tf.train.AdamOptimizer(1e-4).minimize(mse)


            sess = tf.InteractiveSession()
            self.train_network(sess,features)

            # if feed_for_train:
            #     self.train_network(sess)
            #     self.save_model(sess)
            # else:
            #     self.load_model_ckpt(sess)

            #     # feed the image pair and make prediction.
            #     feed_dict = {
            #         self.X: img_pair,
            #     }

            #     v = sess.run({'prediction': predict_flow2},feed_dict=feed_dict)
            #     print(v['prediction'].shape)




    def train_network(self,sess,a):
        # merged_summary_op = tf.summary.merge_all()
        # summary_writer = tf.summary.FileWriter('./tb',graph=sess.graph)

        tf.global_variables_initializer()
        tf.local_variables_initializer()

        # initialize the threads coordinator
        coord = tf.train.Coordinator()
        # start enqueing the data to be dequeued for batch training
        threads = tf.train.start_queue_runners(sess, coord=coord)
        # start passing the data to the feed_dict here and running everything after initializing the variables
        # Retrieve a single instance:
        # batch_xs, batch_ys = sess.run([self.imageBatch, self.labelBatch])

        # sess.run(self.optimizer,feed_dict={self.X: batch_xs, self.Y: batch_ys})


        # print(a.eval().shape)
        # summary_str = sess.run(merged_summary_op, feed_dict={self.X: batch_xs, self.Y: batch_ys})
        # summary_writer.add_summary(summary_str, 1)

        print('done')
        # finalise
        coord.request_stop()
        coord.join(threads)

 
    def save_model(self,sess):
        saver = tf.train.Saver()
        saver.save(sess, 'ckpt/my_test_model.ckpt')


    def load_model_ckpt(self,sess):
        saver = tf.train.Saver()
        saver.restore(sess,'ckpt/my_test_model.ckpt')