import io
import network
import numpy as np
from   PIL import Image
import tensorflow as tf
import helpers as hpl
import tensorflow.contrib.slim as slim
from tensorflow.python import debug as tf_debug
class DatasetReader:


    def tf_record_input_pipeline(self,fullExample):
        # Decode the record read by the reader
        features = tf.parse_single_example(fullExample, {
            'width': tf.FixedLenFeature([], tf.int64),
            'height': tf.FixedLenFeature([], tf.int64),
            'disp1': tf.FixedLenFeature([], tf.string),
            'disp2': tf.FixedLenFeature([], tf.string),
            'opt_flow': tf.FixedLenFeature([], tf.string),
            'cam_frame_L': tf.FixedLenFeature([], tf.string),
            'cam_frame_R': tf.FixedLenFeature([], tf.string),
            'image1': tf.FixedLenFeature([], tf.string),
            'image2': tf.FixedLenFeature([], tf.string),
            'disp_chng': tf.FixedLenFeature([], tf.string),
            'direction': tf.FixedLenFeature([], tf.string)
        })

        # Convert the image data from binary back to arrays(Tensors)
        # disp_width = tf.cast(features['width'], tf.int32)
        # disp_height = tf.cast(features['height'], tf.int32)

        direction = features['direction']
        disp1 = tf.decode_raw(features['disp1'], tf.float32)
        disp2 = tf.decode_raw(features['disp2'], tf.float32)
        image1 = tf.decode_raw(features['image1'], tf.uint8)
        image2 = tf.decode_raw(features['image2'], tf.uint8)
        opt_flow = tf.decode_raw(features['opt_flow'], tf.float32)
        disp_chng = tf.decode_raw(features['disp_chng'], tf.float32)
        cam_frame_L = tf.decode_raw(features['cam_frame_L'], tf.float32)
        cam_frame_R = tf.decode_raw(features['cam_frame_R'], tf.float32)

        self.input_pipeline_dimensions = [160, 256]
        image1 = tf.to_float(image1)
        image2 = tf.to_float(image2)
        # reshape data to its original form
        image1 = tf.reshape(image1, [self.input_pipeline_dimensions[0],self.input_pipeline_dimensions[1], 3],name="reshape_img1")
        image2 = tf.reshape(image2, [self.input_pipeline_dimensions[0],self.input_pipeline_dimensions[1], 3],name="reshape_img2")
    
        disp1 = tf.reshape(disp1, [self.input_pipeline_dimensions[0],self.input_pipeline_dimensions[1]],name="reshape_disp1")
        disp2 = tf.reshape(disp2, [self.input_pipeline_dimensions[0],self.input_pipeline_dimensions[1]],name="reshape_disp2")

        label_pair = tf.reshape(opt_flow, [self.input_pipeline_dimensions[0],self.input_pipeline_dimensions[1],2],name="reshape_img_pair")
        disp_chng = tf.reshape(disp_chng,[self.input_pipeline_dimensions[0],self.input_pipeline_dimensions[1]],name="reshape_disp_change")

        depth1 = self.get_depth_from_disparity(disp1)
        depth2 = self.get_depth_from_disparity(disp2)
        depth_chng = self.get_depth_chng_from_disparity_chng(disp1,disp_chng)

        # normalize image RGB values b/w -0.5 to 0.5
        image1 = tf.divide(image1,[255]) -0.5
        image2 = tf.divide(image2,[255]) -0.5

        # normalize depth values b/w -0.5 to 0.5
        depth1 = tf.divide(depth1,[tf.reduce_max(depth1)]) -0.5
        depth2 = tf.divide(depth2,[tf.reduce_max(depth2)]) -0.5

        # inverse depth
        # depth1 = tf.divide(1,depth1)
        # depth2 = tf.divide(1,depth2)

        image1 = self.combine_depth_values(image1,depth1,2)
        image2 = self.combine_depth_values(image2,depth2,2)

        # depth should be added to both images before this line 
        img_pair = tf.concat([image1,image2],axis=-1)


        label_pair = self.combine_depth_values(label_pair,depth_chng,2)

        # inputt = self.divide_inputs_to_patches(img_pair,8)
        # label = self.divide_inputs_to_patches(label_pair,3)

        # padding_input = tf.constant([[0, 0],[5, 4],[0, 0]])
        # padding_lbl = tf.constant([[0, 0],[5, 5],[0, 0],[0,0]])

        # inputt = tf.pad(inputt,padding_lbl,'SYMMETRIC')
        # label = tf.pad(label,padding_lbl,'SYMMETRIC')

        label = tf.image.resize_images(label_pair,[80,128],tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        return {
            'input': img_pair,
            'label': label
        }



    def divide_inputs_to_patches(self,image,last_dimension):

        
        image = tf.expand_dims(image,0)
        ksize = [1, 54, 96, 1]

        image_patches = tf.extract_image_patches(
            image, ksize, ksize, [1, 1, 1, 1], 'VALID')
        image_patches_reshaped = tf.reshape(image_patches, [-1, 54, 96, last_dimension])

        return image_patches_reshaped


    # combines the depth value in the image RGB values to make it an RGBD tensor.
    # where the resulting tensor will have depth values in the 4th element of 3rd dimension i.e [0][0][3].
    # where [x][x][0] = R, [x][x][1] = G, [x][x][2] = B, [x][x][3] = D
    def combine_depth_values(self,image,depth,rank):
        depth = tf.expand_dims(depth,rank)
        return tf.concat([image,depth],rank)

    def get_depth_chng_from_disparity_chng(self,disparity,disparity_change):

        disparity_change = tf.add(disparity,disparity_change)

        depth1 = self.get_depth_from_disparity(disparity)
        calcdepth = self.get_depth_from_disparity(disparity_change)

        return tf.subtract(depth1,calcdepth)


    def get_depth_from_disparity(self,disparity):
        # focal_length = 35
        # baseline = 1
        # focal_length * baseline = 35

        focal_length = tf.constant([35],dtype=tf.float32)

        disp_to_depth = tf.divide(focal_length,disparity)

        return disp_to_depth


    def iterate(self, filenames):
    
            self.batch_size = 1
            self.epochs = 20

            with tf.name_scope('input_pipeline'):
                # Create a list of filenames and pass it to a queue
                filename_queue = tf.train.string_input_producer(filenames)
                # Define a reader and read the next record
                recordReader = tf.TFRecordReader()

                key, fullExample = recordReader.read(filename_queue)

                features = self.tf_record_input_pipeline(fullExample)

                # shuffle the data and get them as batches
                self.imageBatch, self.labelBatch = tf.train.shuffle_batch([ features['input'], 
                                                                            features['label']],
                                                        batch_size=self.batch_size,
                                                        capacity=100,
                                                        num_threads=1,
                                                        min_after_dequeue=6)

            with tf.name_scope('create_graph'):

                self.X = tf.placeholder(dtype=tf.float32, shape=(self.batch_size, 160, 256, 8))
                self.Y = tf.placeholder(dtype=tf.float32, shape=(self.batch_size, 80, 128, 3))

                # build the network here
                predict_flow5, predict_flow2 = network.train_network(self.X)

                self.mse = tf.reduce_mean(network.change_nans_to_zeros(tf.sqrt(tf.reduce_sum((predict_flow2-self.Y)**2)+1e-3)))
                tf.summary.scalar('MSE', self.mse)

                self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.mse)

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



        merged_summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter('./tb',graph=tf.get_default_graph())

        sess.run([tf.global_variables_initializer(),tf.local_variables_initializer()])

        # initialize the threads coordinator
        coord = tf.train.Coordinator()
        # start enqueing the data to be dequeued for batch training
        threads = tf.train.start_queue_runners(sess, coord=coord)


        # print(a['input'].eval())
        # print('zzzzz')
        # print(a['label'].eval())
        loss = 0
        for i in range(0,self.epochs):

            batch_xs, batch_ys = sess.run([self.imageBatch, self.labelBatch])

            summary, opt,  epoch_loss = sess.run([merged_summary_op, self.optimizer, self.mse],feed_dict={self.X: batch_xs, self.Y: batch_ys})

            loss = loss + epoch_loss
            print(opt)

            print('Epoch: '+str(i)+'     Loss = ',str(epoch_loss))
            print('x min val = ' + str(np.abs(batch_xs).min()))
            print('x max val = ' + str(np.abs(batch_xs).max()))
            print('y min val = ' + str(np.abs(batch_ys).min()))
            print('y max val = ' + str(np.abs(batch_ys).max()))
            print('')

            summary_writer.add_summary(summary, i)

        summary_writer.close()
        # finalise
        coord.request_stop()
        coord.join(threads)

 
    def save_model(self,sess):
        saver = tf.train.Saver()
        saver.save(sess, 'ckpt/my_test_model.ckpt')


    def load_model_ckpt(self,sess):
        saver = tf.train.Saver()
        saver.restore(sess,'ckpt/my_test_model.ckpt')