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

        self.input_pipeline_dimensions = [540, 960]


        image1 = tf.to_float(image1)
        image2 = tf.to_float(image2)
        # reshape data to its original form
        image1 = tf.reshape(image1, [self.input_pipeline_dimensions[0],self.input_pipeline_dimensions[1], 3])
        image2 = tf.reshape(image2, [self.input_pipeline_dimensions[0],self.input_pipeline_dimensions[1], 3])
    

        disp1 = tf.reshape(disp1, [self.input_pipeline_dimensions[0],self.input_pipeline_dimensions[1]])
        disp2 = tf.reshape(disp2, [self.input_pipeline_dimensions[0],self.input_pipeline_dimensions[1]])

        opt_flow = tf.reshape(opt_flow, [self.input_pipeline_dimensions[0],self.input_pipeline_dimensions[1],2])
        disp_chng = tf.reshape(disp_chng,[self.input_pipeline_dimensions[0],self.input_pipeline_dimensions[1]])

        depth1 = self.get_depth_from_disparity(disp1)
        depth2 = self.get_depth_from_disparity(disp2)
        depth_chng = self.get_depth_from_disparity(disp_chng)


        # to check the largest values which are present as discrepencies
        # return {
        #     'depth1': depth1,
        #     'depth2': depth2,
        #     'image1': image1,
        #     'image2': image2,
        # }





        # normalize image RGB values b/w -0.5 to 0.5
        image1 = tf.divide(image1,[255]) -0.5
        image2 = tf.divide(image2,[255]) -0.5

        image1 = self.combine_depth_values(image1,depth1,2)
        image2 = self.combine_depth_values(image2,depth2,2)

        # depth should be added to both images before this line 
        img_pair = tf.concat([image1,image2],axis=-1)

        label_pair = self.combine_depth_values(opt_flow,depth_chng,2)

        result = self.sample_img_patch(img_pair)

        inputt = self.divide_inputs_to_patches(img_pair,8)
        label = self.divide_inputs_to_patches(label_pair,3)


        # padding_input = tf.constant([[0, 0],[5, 4],[0, 0]])
        padding_lbl = tf.constant([[0, 0],[5, 5],[0, 0],[0,0]])

        inputt = tf.pad(inputt,padding_lbl,'SYMMETRIC')
        # label = tf.pad(label,padding_lbl,'SYMMETRIC')

        label = tf.image.resize_images(label,[32,48],tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        return {
            'input': result,
            'label': label
        }



    def sample_img_patch(self,image):

        print('rantaka')
        begin = [20,10,0]
        size =  [256,256,8]

        return  tf.slice(image,begin,size)



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


    def get_depth_from_disparity(self,disparity):
        # focal_length = 35
        # baseline = 1
        # focal_length * baseline = 35

        focal_length = tf.constant([35],dtype=tf.float32)

        disp_to_depth = tf.divide(focal_length,disparity)

        return disp_to_depth


    def get_image_for_testing(self):
        img1 = np.array(Image.open('./dataset/training/Grove2/frame10.png'))
        img2 = np.array(Image.open('./dataset/training/Grove2/frame11.png'))

        img_pair = np.concatenate((img1,img2),axis=-1)

        return img_pair[np.newaxis,:]



    def iterate(self, filenames):
    
            self.batch_size = 1
            self.epochs = 10

            with tf.name_scope('input_pipeline'):
                # Create a list of filenames and pass it to a queue
                filename_queue = tf.train.string_input_producer(filenames)
                # Define a reader and read the next record
                recordReader = tf.TFRecordReader()

                key, fullExample = recordReader.read(filename_queue)

                features = self.decode_features(fullExample)
                # shuffle the data and get them as batches
            #     self.imageBatch, self.labelBatch = tf.train.shuffle_batch([ features['input'], 
            #                                                                 features['label']],
            #                                             batch_size=self.batch_size,
            #                                             capacity=100,
            #                                             num_threads=1,
            #                                             min_after_dequeue=6,
            #                                             enqueue_many=True)

            # with tf.name_scope('create_graph'):

            #     self.X = tf.placeholder(dtype=tf.float32, shape=(self.batch_size, 64, 96, 8))
            #     self.Y = tf.placeholder(dtype=tf.float32, shape=(self.batch_size, 32, 48, 3))

            #     # build the network here
            #     predict_flow5, predict_flow2 = network.train_network(self.X)



            # #     # self.Y = network.change_nans_to_zeros(self.Y)
            # #     # measure of error of our model
            # #     # this needs to be minimised by adjusting W and b
            #     # self.mse = tf.reduce_mean(tf.squared_difference(predict_flow2, self.Y))
            #     self.mse = tf.reduce_mean(network.change_nans_to_zeros(tf.sqrt(tf.reduce_sum((predict_flow2-self.Y)**2)+1e-3)))
            #     tf.summary.scalar('MSE', self.mse)
            # #     # # define training step which minimizes cross entropy
            #     self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.mse)

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
        # start passing the data to the feed_dict here and running everything after initializing the variables
        # Retrieve a single instance:

        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

        # img = a['input'].eval(session=sess)
        # for i in range(0,img.shape[0]):
        #     for j in range(0,img.shape[1]):
        #         print(img[i,j])

        # print('abs')
        # print(np.abs(a['label'].eval()).min())
        # print(np.abs(a['label'].eval()).max())
        # print('absf')

        print(a['input'].eval().shape)

        # loss = 0
        # for i in range(0,self.epochs):



        #     batch_xs, batch_ys = sess.run([self.imageBatch, self.labelBatch])

        #     # result = tf.is_nan(self.labelBatch, name=None)
        #     # arr = result.eval(session=sess)

        #     # print(arr.shape)
        #     # print(np.isnan(arr).any())
        #     # for i in range(arr.shape[0]):
        #     #     for j in range(arr.shape[1]):
        #     #         for k in range(arr.shape[2]):
        #     #             if arr[i,j,k] == True:
        #     #                 print('(i,j,k) = ','(', str(i),',',str(k),',',str(j),')')


        #     # print(batch_xs)
        #     # print('diff')
        #     # print(batch_ys)
        #     # break

        #     summary, opt,  epoch_loss = sess.run([merged_summary_op, self.optimizer, self.mse],feed_dict={self.X: batch_xs, self.Y: batch_ys})

        #     # loss = loss + epoch_loss
        #     # print(opt)

        #     print(opt)
        #     print('Epoch: '+str(i)+'     Loss = ',str(epoch_loss))

        #     summary_writer.add_summary(summary, i)

        # summary_writer.close()
        # finalise
        coord.request_stop()
        coord.join(threads)

 
    def save_model(self,sess):
        saver = tf.train.Saver()
        saver.save(sess, 'ckpt/my_test_model.ckpt')


    def load_model_ckpt(self,sess):
        saver = tf.train.Saver()
        saver.restore(sess,'ckpt/my_test_model.ckpt')