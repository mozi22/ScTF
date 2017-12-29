import io
import network
import numpy as np
from   PIL import Image
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python import debug as tf_debug
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

        self.input_pipeline_dimensions = [540, 960]


        image = tf.to_float(image)
        # reshape data to its original form
        image = tf.reshape(image, [self.input_pipeline_dimensions[0],self.input_pipeline_dimensions[1], 3])
        disp = tf.reshape(disp, [self.input_pipeline_dimensions[0],self.input_pipeline_dimensions[1]])
        opt_flow = tf.reshape(opt_flow, [self.input_pipeline_dimensions[0],self.input_pipeline_dimensions[1],2])
        disp_chng = tf.reshape(disp_chng,[self.input_pipeline_dimensions[0],self.input_pipeline_dimensions[1]])
        depth,depth_change = self.get_depth_from_disparity(disp,disp_chng)

        # return opt_flow
        # image = tf.divide(image,[255]) -0.5
        image2 = tf.expand_dims(image,0)
        print(image2)
        tf.summary.image('image', image2)


        inputt = self.combine_depth_values(image,depth,2)
        label = self.combine_depth_values(opt_flow,depth_change,2)

        inputt = self.divide_inputs_to_patches(inputt,4)
        label = self.divide_inputs_to_patches(label,3)


        # padding_input = tf.constant([[0, 0],[5, 4],[0, 0]])
        padding_lbl = tf.constant([[0, 0],[5, 5],[0, 0],[0,0]])

        inputt = tf.pad(inputt,padding_lbl,'SYMMETRIC')
        label = tf.pad(label,padding_lbl,'SYMMETRIC')

        label = tf.image.resize_images(label,[32,48],tf.image.ResizeMethod.NEAREST_NEIGHBOR)


        return {
            'input': inputt,
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
    
            self.batch_size = 1
            self.epochs = 100

            if feed_for_train == False:
                img_pair = self.get_image_for_testing()



            with tf.name_scope('input_pipeline'):
                # Create a list of filenames and pass it to a queue
                filename_queue = tf.train.string_input_producer(filenames)
                # Define a reader and read the next record
                recordReader = tf.TFRecordReader()

                key, fullExample = recordReader.read(filename_queue)

                features = self.decode_features(fullExample)
                print(features)
                # shuffle the data and get them as batches
                self.imageBatch, self.labelBatch = tf.train.shuffle_batch([ features['input'], 
                                                                            features['label']],
                                                        batch_size=self.batch_size,
                                                        capacity=100,
                                                        num_threads=1,
                                                        min_after_dequeue=6,
                                                        enqueue_many=True)

            with tf.name_scope('create_graph'):

                if feed_for_train == False:
                    self.X = tf.placeholder(dtype=tf.float32, shape=(1, 80, 128, 4))
                    self.Y = tf.placeholder(dtype=tf.float32, shape=(1, 40, 64, 3))
                else:
                    self.X = tf.placeholder(dtype=tf.float32, shape=(self.batch_size, 64, 96, 4))
                    self.Y = tf.placeholder(dtype=tf.float32, shape=(self.batch_size, 32, 48, 3))

                # build the network here
                predict_flow5, predict_flow2 = network.train_network(self.X)

                # measure of error of our model
                # this needs to be minimised by adjusting W and b
                # self.mse = tf.reduce_mean(tf.squared_difference(predict_flow2, self.Y))
                self.mse = tf.reduce_mean(tf.sqrt(tf.reduce_sum((predict_flow2-self.Y)**2)))
                tf.summary.scalar('MSE', self.mse)
                # # # define training step which minimizes cross entropy
                self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.mse)

            sess = tf.Session()
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
        # start passing the data to the feed_dict here and running everything after initializing the variables
        # Retrieve a single instance:

        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

        # img = a['input'].eval(session=sess)
        # for i in range(0,img.shape[0]):
        #     for j in range(0,img.shape[1]):
        #         print(img[i,j])

        loss = 0
        for i in range(0,self.epochs):



            batch_xs, batch_ys = sess.run([self.imageBatch, self.labelBatch])

            # result = tf.is_nan(self.labelBatch, name=None)
            # arr = result.eval(session=sess)

            # print(arr.shape)
            # print(np.isnan(arr).any())
            # for i in range(arr.shape[0]):
            #     for j in range(arr.shape[1]):
            #         for k in range(arr.shape[2]):
            #             if arr[i,j,k] == True:
            #                 print('(i,j,k) = ','(', str(i),',',str(k),',',str(j),')')


            # print(batch_xs)
            # print('diff')
            # print(batch_ys)
            # break

            summary, _,  epoch_loss = sess.run([merged_summary_op,self.optimizer, self.mse],feed_dict={self.X: batch_xs, self.Y: batch_ys})

            loss = loss + epoch_loss

            print('Epoch: '+str(i)+'     Loss = ',str(epoch_loss))

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