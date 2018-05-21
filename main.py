import os
import re
import time
import math
import sys
import network
import shutil
import logging
import numpy as np
import losses_helper
import tensorflow as tf
from six.moves import xrange
import tensorflow.contrib.slim as slim
from tensorflow.python import debug as tf_debug
from tensorflow.python.client import device_lib
from datetime import datetime
# import ijremote
import configparser as configp
import data_reader
'''
TRAINING:
    driving = 200
    flying = 22390
    monkaa = 6050
    ptb    = 20332
    --------------
            48972

TESTING:
    driving = 100
    flying = 4370
    monkaa = 2614
    ptb    = 1210
    --------------
            8294

'''

class DatasetReader:

    def delete_files_in_directories(self,dir_path):
        fileList = os.listdir(dir_path)
        for fileName in fileList:
            os.remove(dir_path+"/"+fileName)

    def create_and_remove_directories(self,dir_path,clean_existing_files,load_from_ckpt,testing_enabled):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path+'/train')
            os.makedirs(dir_path+'/test')
        else:
            if clean_existing_files == True and load_from_ckpt == False and testing_enabled == False:
                self.delete_files_in_directories(dir_path+'/train')
                self.delete_files_in_directories(dir_path+'/test')

    def create_input_pipeline(self,sections):
        
        prefix = self.FLAGS['DATASET_FOLDER']
        self.dataset_used = 1

        # memory_folder = '/dev/shm/'

        self.filenames_train  = ['driving_TRAIN.tfrecords','flying_TRAIN.tfrecords','monkaa_TRAIN.tfrecords','ptb_TRAIN.tfrecords']
        self.filenames_test  = ['driving_TEST.tfrecords','flying_TEST.tfrecords','monkaa_TEST.tfrecords','ptb_TEST.tfrecords']

        # driving
        if sections[self.section_type] == sections[0]:
            train_filenames = [prefix+self.filenames_train[0]]
            test_filenames = [prefix+self.filenames_test[0]]
            self.dataset_used = 1

        # driving, flying
        elif sections[self.section_type] == sections[1]:
            train_filenames = [prefix+self.filenames_train[0],prefix+self.filenames_train[1]]
            test_filenames = [prefix+self.filenames_test[0],prefix+self.filenames_test[1]]
            self.dataset_used = 2

        # driving, flying, monkaa
        elif sections[self.section_type] == sections[2]:
            train_filenames = [prefix+self.filenames_train[0],prefix+self.filenames_train[1],prefix+self.filenames_train[2]]
            test_filenames = [prefix+self.filenames_test[0],prefix+self.filenames_test[1],prefix+self.filenames_test[2]]
            self.dataset_used = 3

        # driving, flying, monkaa, ptb
        elif sections[self.section_type] == sections[3]:
            train_filenames = [prefix+self.filenames_train[0],prefix+self.filenames_train[1],prefix+self.filenames_train[2],prefix+self.filenames_train[3]]
            test_filenames = [prefix+self.filenames_test[0],prefix+self.filenames_test[1],prefix+self.filenames_test[2],prefix+self.filenames_test[3]]
            self.dataset_used = 4

        # only testing with ptb
        elif sections[self.section_type] == sections[4]:
            train_filenames = [prefix+self.filenames_train[3]]
            test_filenames = [prefix+self.filenames_test[3]]
            self.dataset_used = 1


        train_dataset = data_reader.read_with_dataset_api(self.FLAGS['BATCH_SIZE'],train_filenames,version='1')
        test_dataset = data_reader.read_with_dataset_api(self.FLAGS['TEST_BATCH_SIZE'],test_filenames,version='2')

        train_iterator = train_dataset.make_initializable_iterator()
        test_iterator = test_dataset.make_initializable_iterator()


        return train_iterator, test_iterator


    def preprocess(self):
        file = './configs/training.ini'

        self.section_type = 4

        parser = configp.ConfigParser()
        parser.read(file)
        sections = parser.sections()

        self.FLAGS = {
            # TRAIN
            'BATCH_SIZE': int(parser[sections[self.section_type]]['BATCH_SIZE']),
            'TRAIN_DIR': parser[sections[self.section_type]]['TRAIN_DIR'],
            'LOAD_FROM_CKPT': parser[sections[self.section_type]].getboolean('LOAD_FROM_CKPT'),
            'DEBUG_MODE': parser[sections[self.section_type]].getboolean('DEBUG_MODE'),
            'TOWER_NAME': parser[sections[self.section_type]]['TOWER_NAME'],
            'MAX_STEPS': int(parser[sections[self.section_type]]['MAX_STEPS']),
            'LOG_DEVICE_PLACEMENT': parser[sections[self.section_type]].getboolean('LOG_DEVICE_PLACEMENT'),
            'NUM_EPOCHS_PER_DECAY': int(parser[sections[self.section_type]]['NUM_EPOCHS_PER_DECAY']),
            'SHUFFLE_BATCH_QUEUE_CAPACITY': int(parser[sections[self.section_type]]['SHUFFLE_BATCH_QUEUE_CAPACITY']),
            'SHUFFLE_BATCH_THREADS': int(parser[sections[self.section_type]]['SHUFFLE_BATCH_THREADS']),
            'SHUFFLE_BATCH_MIN_AFTER_DEQUEUE': int(parser[sections[self.section_type]]['SHUFFLE_BATCH_MIN_AFTER_DEQUEUE']),
            'NUM_GPUS': int(parser[sections[self.section_type]]['NUM_GPUS']),
            'MOVING_AVERAGE_DECAY': float(parser[sections[self.section_type]]['MOVING_AVERAGE_DECAY']),
            'TOTAL_TRAIN_EXAMPLES': int(parser[sections[self.section_type]]['TOTAL_TRAIN_EXAMPLES']),
            'CLEAN_FILES': parser[sections[self.section_type]].getboolean('CLEAN_FILES'),
            'TRAIN_WITH_PTB' : parser[sections[self.section_type]].getboolean('TRAIN_WITH_PTB'),
            'DATASET_FOLDER': parser[sections[self.section_type]]['DATASET_FOLDER'],

            # TEST
            'TESTING_ENABLED': parser[sections[self.section_type]].getboolean('TESTING_ENABLED'),
            'TOTAL_TEST_EXAMPLES': int(parser[sections[self.section_type]]['TOTAL_TEST_EXAMPLES']),
            'TEST_BATCH_SIZE': int(parser[sections[self.section_type]]['TEST_BATCH_SIZE']),
            'TEST_AFTER_EPOCHS': int(parser[sections[self.section_type]]['TEST_AFTER_EPOCHS']),
            'TOTAL_TRAIN_EXAMPLES': int(parser[sections[self.section_type]]['TOTAL_TRAIN_EXAMPLES']),
            'TEST_ON_PTB_ONLY': parser[sections[self.section_type]].getboolean('TEST_ON_PTB_ONLY'),

            # LR
            'START_LEARNING_RATE': float(parser[sections[self.section_type]]['START_LEARNING_RATE']),
            'END_LEARNING_RATE': float(parser[sections[self.section_type]]['END_LEARNING_RATE']),
            'POWER': int(parser[sections[self.section_type]]['POWER'])
        }

        self.create_and_remove_directories(self.FLAGS['TRAIN_DIR'],
                                           self.FLAGS['CLEAN_FILES'],
                                           self.FLAGS['LOAD_FROM_CKPT'],
                                           self.FLAGS['TESTING_ENABLED'])

        self.TRAIN_DIR_LIST = self.FLAGS['TRAIN_DIR'].split('/')

        # gives the # of steps required to complete 1 epoch
        self.TRAIN_EPOCH = math.ceil(self.FLAGS['TOTAL_TRAIN_EXAMPLES'] / self.FLAGS['BATCH_SIZE'])
        self.TEST_EPOCH = math.ceil(self.FLAGS['TOTAL_TEST_EXAMPLES'] / self.FLAGS['TEST_BATCH_SIZE'])

        train_iterator, test_iterator = self.create_input_pipeline(sections)

        # for testing
        self.X = tf.placeholder(dtype=tf.float32, shape=(self.FLAGS['TEST_BATCH_SIZE'] * self.dataset_used, 224, 384, 8))
        self.Y = tf.placeholder(dtype=tf.float32, shape=(self.FLAGS['TEST_BATCH_SIZE'] * self.dataset_used, 224, 384, 3))

        return train_iterator, test_iterator



    def train(self,iterator_train,iterator_test):

        self.global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(0), trainable=False)

        decay_steps = self.FLAGS['MAX_STEPS']
        start_learning_rate = self.FLAGS['START_LEARNING_RATE']
        end_learning_rate = self.FLAGS['END_LEARNING_RATE']
        power = self.FLAGS['POWER']

        learning_rate = tf.train.polynomial_decay(start_learning_rate, self.global_step,
                                                  decay_steps, end_learning_rate,
                                                  power=power)


        opt = tf.train.AdamOptimizer(learning_rate)
    
        # images, labels = tf.train.shuffle_batch(
        #                     [ features_train['input_n'] , features_train['label_n'] ],
        #                     batch_size=self.FLAGS['BATCH_SIZE'],
        #                     capacity=self.FLAGS['SHUFFLE_BATCH_QUEUE_CAPACITY'],
        #                     num_threads=self.FLAGS['SHUFFLE_BATCH_THREADS'],
        #                     min_after_dequeue=self.FLAGS['SHUFFLE_BATCH_MIN_AFTER_DEQUEUE'],
        #                     enqueue_many=False)

        # batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue(
        #     [images, labels], capacity=self.FLAGS['SHUFFLE_BATCH_QUEUE_CAPACITY'] * self.FLAGS['NUM_GPUS'])



        # if self.FLAGS['TESTING_ENABLED'] == True:
        #     self.images_test, self.labels_test = tf.train.shuffle_batch(
        #                         [ features_test['input_n'] , features_test['label_n'] ],
        #                         batch_size=self.FLAGS['TEST_BATCH_SIZE'],
        #                         capacity=self.FLAGS['SHUFFLE_BATCH_QUEUE_CAPACITY'],
        #                         num_threads=self.FLAGS['SHUFFLE_BATCH_THREADS'],
        #                         min_after_dequeue=self.FLAGS['SHUFFLE_BATCH_MIN_AFTER_DEQUEUE'],
        #                         enqueue_many=False)
    
            # self.batch_queue_test = tf.contrib.slim.prefetch_queue.prefetch_queue(
            #     [self.images_test, self.labels_test], capacity=self.FLAGS['SHUFFLE_BATCH_QUEUE_CAPACITY'] * self.FLAGS['NUM_GPUS'])
        

        
        tower_grads = []
        with tf.variable_scope(tf.get_variable_scope()):
          for i in xrange(self.FLAGS['NUM_GPUS']):
            with tf.device('/gpu:%d' % i):
              with tf.name_scope('%s_%d' % ('tower', i)) as scope:

                # Dequeues one batch for the GPU
                image_batch, label_batch = self.combine_batches_from_datasets(iterator_train.get_next())

                # Calculate the loss for one tower of the CIFAR model. This function
                # constructs the entire CIFAR model but shares the variables across
                # all towers.
                self.loss = self.tower_loss(scope, image_batch, label_batch)

                # Reuse variables for the next tower.
                tf.get_variable_scope().reuse_variables()

                # Retain the summaries from the final tower.
                summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)


                # Calculate the gradients for the batch of data on this CIFAR tower.
                grads = opt.compute_gradients(self.loss)
                # grads = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in grads]

                # Keep track of the gradients across all towers.
                tower_grads.append(grads)


        # We must calculate the mean of each gradient. Note that this is the
        # synchronization point across all towers.
        grads = self.average_gradients(tower_grads)



        if not self.section_type is 4:
            # Add a summary to track the learning rate.
            summaries.append(tf.summary.scalar('learning_rate', learning_rate))
    
            # Add histograms for gradients.
            for grad, var in grads:
                if grad is not None:
                    summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))


        # Apply the gradients to adjust the shared variables.
        apply_gradient_op = opt.apply_gradients(grads, global_step=self.global_step)

        # Add histograms for trainable variables.
        for var in tf.trainable_variables():
            summaries.append(tf.summary.histogram(var.op.name, var))

        # Track the moving averages of all trainable variables.
        variable_averages = tf.train.ExponentialMovingAverage(
            self.FLAGS['MOVING_AVERAGE_DECAY'], self.global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())

        # Group all updates to into a single train op.
        train_op = tf.group(apply_gradient_op, variables_averages_op)

        # Create a saver.
        saver = tf.train.Saver(tf.global_variables())


        # Build the summary operation from the last tower summaries.
        self.summary_op = tf.summary.merge(summaries)

        # Build an initialization operation to run below.
        init = tf.global_variables_initializer()

        self.update_test_datasets_image_summaries()

        # Start running operations on the Graph. allow_soft_placement must be set to
        # True to build towers on GPU, as some of the ops do not have GPU
        # implementations.
        tf_config = tf.ConfigProto(allow_soft_placement=True,
            log_device_placement=self.FLAGS['LOG_DEVICE_PLACEMENT'])

        if 'dacky' in os.uname()[1]:
            logging.info('Dacky: Running with memory usage limits')
            # change tf_config for dacky to use only 1 GPU
            tf_config.gpu_options.per_process_gpu_memory_fraction = 0.6
            os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        else:
            # change tf_config for lmb_cluster so that GPU is visible and utilized
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"

        sess = tf.Session(config=tf_config)
        sess.run(init)
        sess.run(train_iterator.initializer)
        sess.run(iterator_test.initializer)


        if self.FLAGS['LOAD_FROM_CKPT'] == True or self.section_type == 4:
            print('loading from ckpt...')
            saver.restore(sess,tf.train.latest_checkpoint(self.FLAGS['TRAIN_DIR']+'/train/'))


        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)

        # for debugging


        if self.section_type == 4:
            if os.path.exists(self.FLAGS['TRAIN_DIR'] +'/ptb_test'):
                shutil.rmtree(self.FLAGS['TRAIN_DIR'] +'/ptb_test')
            os.makedirs(self.FLAGS['TRAIN_DIR'] +'/ptb_test')                
            self.test_summary_writer = tf.summary.FileWriter(self.FLAGS['TRAIN_DIR']+'/ptb_test', sess.graph)
        else: 
            summary_writer = tf.summary.FileWriter(self.FLAGS['TRAIN_DIR']+'/train', sess.graph)
            self.test_summary_writer = tf.summary.FileWriter(self.FLAGS['TRAIN_DIR']+'/test', sess.graph)


        # just to make sure we start from where we left, if load_from_ckpt = True
        loop_start = tf.train.global_step(sess, self.global_step)
        loop_stop = loop_start + self.FLAGS['MAX_STEPS']

        if self.FLAGS['DEBUG_MODE']:
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)

        first_iteration = True

        # this will print in console for which time we are calculating the test loss.
        # first time or second time or third time and more
        test_loss_calculating_index = 1


        # this will not train and only test the model on ptb dataset.
        if self.section_type == 4:
            self.test_model_on_ptb(sess,iterator_test)
            sys.exit('Done Testing')


        # main loop
        for step in range(loop_start,loop_stop):

            start_time = time.time()

            _, loss_value = sess.run([train_op, self.loss])


            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'



            # ####################### Testing #######################
            # # after every 10 epochs. calculate test loss
            # if step % (self.TRAIN_EPOCH * FLAGS.TEST_AFTER_EPOCHS) == 0 and first_iteration==False:

            #     message = 'Printing Test loss for '+str(test_loss_calculating_index)+' time'

            #     self.log()
            #     self.log(message)
            #     self.log()

            #     self.perform_testing(sess,step)

            #     # increment index to know how many times we've calculated the test loss
            #     test_loss_calculating_index = test_loss_calculating_index + 1

            # ####################### Testing #######################
            if step % 10 == 0 or first_iteration==True:
                num_examples_per_step = self.FLAGS['BATCH_SIZE'] * self.FLAGS['NUM_GPUS']
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = duration / self.FLAGS['NUM_GPUS']
                first_iteration = False

            format_str = ('%s: step %d, DIR: %s, loss = %.15f (%.1f examples/sec; %.3f '
                          'sec/batch)')
            self.log(message=(format_str % (datetime.now(),step,self.TRAIN_DIR_LIST[-1], loss_value,
                                 examples_per_sec, sec_per_batch)))

            # if loss_value > 100:
            #     summary_str = sess.run(self.summary_op)
            #     summary_writer.add_summary(summary_str, step)

            if step % 100 == 0 and step!=0:
                summary_str = sess.run(self.summary_op)
                summary_writer.add_summary(summary_str, step)

                # write a summary for test
                test_image_batch, test_label_batch = self.combine_batches_from_datasets(iterator_test.get_next())
                test_image_batch, test_label_batch = self.get_network_input_forward(test_image_batch,test_label_batch)
                test_image_batch, test_label_batch = sess.run([test_image_batch, test_label_batch])

                loss_value, summary_str_test = sess.run([self.loss,self.summary_op],feed_dict={ 
                            self.X: test_image_batch,
                            self.Y: test_label_batch
                })

                self.test_summary_writer.add_summary(summary_str_test, step)

            # Save the model checkpoint periodically.
            if step % 500 == 0 or (step + 1) == self.FLAGS['MAX_STEPS']:
                checkpoint_path = os.path.join(self.FLAGS['TRAIN_DIR']+'/train', 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

        summary_writer.close()
    
    def test_model_on_ptb(self,sess,iterator_test):
        self.log()
        self.log(message='Testing ..., Total Test Epochs = ' + str(self.TEST_EPOCH))
        self.log()



        # iterator_test = sess.run(iterator_test)
        for step in range(0,self.TEST_EPOCH + 10):

            image_batch, label_batch = self.combine_batches_from_datasets(iterator_test.get_next())
            image_batch, label_batch = self.get_network_input_forward(image_batch,label_batch)

            image,label = sess.run([image_batch, label_batch])

            loss_value,summary_str = sess.run([self.loss,self.summary_op],feed_dict={self.X: image, self.Y: label})

            format_str = ('%s: Testing step %d, loss = %.15f')
            self.log(message=(format_str % (datetime.now(), step, np.log10(loss_value))))

            if step % 50 == 0:
                self.test_summary_writer.add_summary(summary_str, step)

        self.log()
        self.log(message='Continue Training ...')
        self.log()


    def update_test_datasets_image_summaries(self):

        # if TEST_ON_PTB_ONLY = True, than we'll only have 4 values in the dataset i.e [4,:,:,:]. Hence we only use
        # the first part and the rest is left out in the summary.
        if self.section_type == 4:
            flow_u_1 = 'test_flow_u_1_ptb'
            flow_v_1 = 'test_flow_v_1_ptb'
            input_image1 = 'test_input_ptb'
            input_image2 = 'test_input_ptb'
            depth_image1 = 'test_depth_ptb'
            depth_image2 = 'test_depth_ptb'
        else:
            flow_u_1 = 'test_flow_u_1_driving'
            flow_v_1 = 'test_flow_v_1_driving'
            input_image1 = 'test_input_image1_driving'
            input_image2 = 'test_input_image2_driving'
            depth_image1 = 'test_depth_image1_driving'
            depth_image2 = 'test_depth_image2_driving'


        # driving or ptb in case of TEST_ON_PTB_ONLY = True
        tf.summary.image(flow_u_1,self.Y[0:self.FLAGS['BATCH_SIZE'],:,:,0:1])
        tf.summary.image(flow_v_1,self.Y[0:self.FLAGS['BATCH_SIZE'],:,:,1:2])
        tf.summary.image(input_image1,self.X[0:self.FLAGS['BATCH_SIZE'],:,:,0:3])
        tf.summary.image(input_image2,self.X[0:self.FLAGS['BATCH_SIZE'],:,:,4:7])
        tf.summary.image(depth_image1,tf.expand_dims(self.X[0:self.FLAGS['BATCH_SIZE'],:,:,3],axis=-1))
        tf.summary.image(depth_image2,tf.expand_dims(self.X[0:self.FLAGS['BATCH_SIZE'],:,:,7],axis=-1))


        if self.section_type is not 4:

            # flying
            tf.summary.image('test_flow_u_1_flying',self.Y[self.FLAGS['BATCH_SIZE']: (self.FLAGS['BATCH_SIZE']*2),:,:,0:1])
            tf.summary.image('test_flow_v_1_flying',self.Y[self.FLAGS['BATCH_SIZE']: (self.FLAGS['BATCH_SIZE']*2),:,:,1:2])
            tf.summary.image('test_input_image1_flying',self.X[self.FLAGS['BATCH_SIZE']: (self.FLAGS['BATCH_SIZE']*2),:,:,0:3])
            tf.summary.image('test_input_image2_flying',self.X[self.FLAGS['BATCH_SIZE']: (self.FLAGS['BATCH_SIZE']*2),:,:,4:7])
            tf.summary.image('test_depth_image1_flying',tf.expand_dims(self.X[self.FLAGS['BATCH_SIZE']: (self.FLAGS['BATCH_SIZE']*2),:,:,3],axis=-1))
            tf.summary.image('test_depth_image2_flying',tf.expand_dims(self.X[self.FLAGS['BATCH_SIZE']: (self.FLAGS['BATCH_SIZE']*2),:,:,7],axis=-1))

            # monkaa
            tf.summary.image('test_flow_u_1_monkaa',self.Y[self.FLAGS['BATCH_SIZE']: (self.FLAGS['BATCH_SIZE']*3),:,:,0:1])
            tf.summary.image('test_flow_v_1_monkaa',self.Y[self.FLAGS['BATCH_SIZE']: (self.FLAGS['BATCH_SIZE']*3),:,:,1:2])
            tf.summary.image('test_input_image1_monkaa',self.X[self.FLAGS['BATCH_SIZE']: (self.FLAGS['BATCH_SIZE']*3),:,:,0:3])
            tf.summary.image('test_input_image2_monkaa',self.X[self.FLAGS['BATCH_SIZE']: (self.FLAGS['BATCH_SIZE']*3),:,:,4:7])
            tf.summary.image('test_depth_image1_monkaa',tf.expand_dims(self.X[self.FLAGS['BATCH_SIZE']: (self.FLAGS['BATCH_SIZE']*3),:,:,3],axis=-1))
            tf.summary.image('test_depth_image2_monkaa',tf.expand_dims(self.X[self.FLAGS['BATCH_SIZE']: (self.FLAGS['BATCH_SIZE']*3),:,:,7],axis=-1))


            if self.section_type == 3:
                # ptb
                tf.summary.image('test_flow_u_1_ptb',self.Y[self.FLAGS['BATCH_SIZE']: (self.FLAGS['BATCH_SIZE']*4),:,:,0:1])
                tf.summary.image('test_flow_v_1_ptb',self.Y[self.FLAGS['BATCH_SIZE']: (self.FLAGS['BATCH_SIZE']*4),:,:,1:2])
                tf.summary.image('test_input_image1_ptb',self.X[self.FLAGS['BATCH_SIZE']: (self.FLAGS['BATCH_SIZE']*4),:,:,0:3])
                tf.summary.image('test_input_image2_ptb',self.X[self.FLAGS['BATCH_SIZE']: (self.FLAGS['BATCH_SIZE']*4),:,:,4:7])
                tf.summary.image('test_depth_image1_ptb',tf.expand_dims(self.X[self.FLAGS['BATCH_SIZE']: (self.FLAGS['BATCH_SIZE']*3),:,:,3],axis=-1))
                tf.summary.image('test_depth_image2_ptb',tf.expand_dims(self.X[self.FLAGS['BATCH_SIZE']: (self.FLAGS['BATCH_SIZE']*3),:,:,7],axis=-1))




    def combine_batches_from_datasets(self,batches):


        driving_batch_img = batches[0][0]
        driving_batch_lbl = batches[0][1]

        # this will be ptb only than
        if self.section_type == 4:
            final_img_batch = tf.concat((driving_batch_img),axis=0)
            final_lbl_batch = tf.concat((driving_batch_lbl),axis=0)

            return final_img_batch, final_lbl_batch

        flying_batch_img = batches[1][0]
        flying_batch_lbl = batches[1][1]

        monkaa_batch_img = batches[2][0]
        monkaa_batch_lbl = batches[2][1]


        if self.section_type == 3:
            ptb_batch_img = batches[3][0]
            ptb_batch_lbl = batches[3][1]

            final_img_batch = tf.concat((driving_batch_img,flying_batch_img,monkaa_batch_img,ptb_batch_img),axis=0)
            final_lbl_batch = tf.concat((driving_batch_lbl,flying_batch_lbl,monkaa_batch_lbl,ptb_batch_lbl),axis=0)
        else:
            final_img_batch = tf.concat((driving_batch_img,flying_batch_img,monkaa_batch_img),axis=0)
            final_lbl_batch = tf.concat((driving_batch_lbl,flying_batch_lbl,monkaa_batch_lbl),axis=0)

        return final_img_batch, final_lbl_batch

    

    def further_resize_imgs_lbls(self,network_input_images,network_input_labels):

        network_input_images = tf.image.resize_images(network_input_images,[160,256],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        network_input_labels = tf.image.resize_images(network_input_labels,[160,256],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        network_input_labels_u = network_input_labels[:,:,:,0] * 0.714285714
        network_input_labels_v = network_input_labels[:,:,:,1] * 0.666666667
        network_input_labels_w = network_input_labels[:,:,:,2]

        network_input_labels_u = tf.expand_dims(network_input_labels_u,axis=-1)
        network_input_labels_v = tf.expand_dims(network_input_labels_v,axis=-1)
        network_input_labels_w = tf.expand_dims(network_input_labels_w,axis=-1)

        network_input_labels = tf.concat([network_input_labels_u,network_input_labels_v,network_input_labels_w],axis=3)

        return network_input_images, network_input_labels


    def remove_ptb_records(self,network_input_images,network_input_labels):
        return network_input_images[0:12,:,:,:], network_input_labels[0:12,:,:,:]


    def tower_loss(self,scope, images, labels):
        """Calculate the total loss on a single tower running the CIFAR model.
        Args:
        scope: unique prefix string identifying the CIFAR tower, e.g. 'tower_0'
        images: Images. 4D tensor of shape [batch_size, height, width, 3].
        labels: Labels. 1D tensor of shape [batch_size].
        Returns:
         Tensor of shape [] containing the total loss for a batch of data
        """
        network_input_images, network_input_labels = self.get_network_input_forward(images,labels)
        network_input_images_back, network_input_labels_back = self.get_network_input_backward(images,labels)

        network_input_images, network_input_labels = self.further_resize_imgs_lbls(network_input_images,network_input_labels)
        network_input_images_back, network_input_labels_back = self.further_resize_imgs_lbls(network_input_images_back,network_input_labels_back)

        # FB = forward-backward
        concatenated_FB_images = tf.concat([network_input_images,network_input_images_back],axis=0)


        # Build inference Graph. - forward flow
        predict_flows = network.train_network(concatenated_FB_images)

        # backward_flow_images = losses_helper.forward_backward_loss(predict_flow)
        # Build inference Graph. - backward flow
        # Build the portion of the Graph calculating the losses. Note that we will
        # assemble the total_loss using a custom function below.

        # losses sections[self.section_type]

        # with tf.variable_scope('fb_loss_refine_3'):
        #     _ = losses_helper.forward_backward_loss(predict_flows[1])
        # with tf.variable_scope('fb_loss_refine_3'):
        #     _ = losses_helper.forward_backward_loss(predict_flows[2])
        # with tf.variable_scope('fb_loss_refine_3'):
        #     _ = losses_helper.forward_backward_loss(predict_flows[3])


        flows_dict = self.get_predict_flow_forward_backward(predict_flows,network_input_labels,concatenated_FB_images)

        self.write_flows_concatenated_side_by_side(network_input_labels,flows_dict['predict_flow'][0])

        # predict_flow2_label = losses_helper.downsample_label(network_input_labels)

       # predict_flow2_refine3 = losses_helper.downsample_label(predict_flows[1],
       #                                  size=[20,32],factorU=0.125,factorV=0.125)
       #  predict_flow2_refine2 = losses_helper.downsample_label(predict_flows[2],
       #                                  size=[40,64],factorU=0.25,factorV=0.25)
       #  predict_flow2_refine1 = losses_helper.downsample_label(predict_flows[3],
       #                                  size=[80,128],factorU=0.5,factorV=0.5)

        '''
            Applying pc loss on lower resolutions
        '''

        # network_input_images_refine3 = tf.image.resize_images(network_input_images,[20,32],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        # network_input_images_refine2 = tf.image.resize_images(network_input_images,[40,64],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        # network_input_images_refine1 = tf.image.resize_images(network_input_images,[80,128],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        # with tf.variable_scope('photoconsistency_loss_refine_3'):
        #     _ = losses_helper.photoconsistency_loss(network_input_images_refine3,flows_dict['predict_flow_ref3'][0])
        # with tf.variable_scope('photoconsistency_loss_refine_2'):
        #     _ = losses_helper.photoconsistency_loss(network_input_images_refine2,flows_dict['predict_flow_ref2'][0])
        # with tf.variable_scope('photoconsistency_loss_refine_1'):
        #     _ = losses_helper.photoconsistency_loss(network_input_images_refine1,flows_dict['predict_flow_ref1'][0])



        # unsupervised losses done. Now remove ptb. Since it doesn't have ground truth.
        if self.section_type == 3:
            network_input_images, network_input_labels = self.remove_ptb_records(network_input_images, network_input_labels)


        # supervised

        '''
            Applying epe loss on full resolution
        '''
        _ = losses_helper.endpoint_loss(network_input_labels,flows_dict['predict_flow'][0])

        '''
            Applying epe loss on lower resolutions
        '''

        network_input_labels_refine3 = losses_helper.downsample_label(network_input_labels,
                                        size=[20,32],factorU=0.125,factorV=0.125)
        network_input_labels_refine2 = losses_helper.downsample_label(network_input_labels,
                                        size=[40,64],factorU=0.25,factorV=0.25)
        network_input_labels_refine1 = losses_helper.downsample_label(network_input_labels,
                                        size=[80,128],factorU=0.5,factorV=0.5)




        with tf.variable_scope('epe_loss_refine_3'):
            _ = losses_helper.endpoint_loss(network_input_labels_refine3,flows_dict['predict_flow_ref3'][0],100)
        with tf.variable_scope('epe_loss_refine_2'):
            _ = losses_helper.endpoint_loss(network_input_labels_refine2,flows_dict['predict_flow_ref2'][0],100)
        with tf.variable_scope('epe_loss_refine_1'):
            _ = losses_helper.endpoint_loss(network_input_labels_refine1,flows_dict['predict_flow_ref1'][0],100)




        # _ = losses_helper.photoconsistency_loss(network_input_images,predict_flows[0])
        # _ = losses_helper.depth_consistency_loss(network_input_images,predict_flows[0])


        # scale_invariant_gradient_image_gt = losses_helper.scale_invariant_gradient(network_input_labels,
        #                                                                         np.array([1,2,4,8,16]),
        #                                                                         np.array([1,1,1,1,1]))

        # scale_invariant_gradient_image_pred = losses_helper.scale_invariant_gradient(predict_flow2_forward,
        #                                                                         np.array([1,2,4,8,16]),
        #                                                                         np.array([1,1,1,1,1]))

        # _ = losses_helper.scale_invariant_gradient_loss(
        #         scale_invariant_gradient_image_pred,
        #         scale_invariant_gradient_image_gt,
        #         0.0001,
        #         self.FLAGS['MAX_STEPS'],
        #         self.global_step)


 
        # _ = losses_helper.depth_loss(predict_flow5_label,predict_flow5)


        # Assemble all of the losses for the current tower only.
        losses = tf.get_collection('losses', scope)

        # Calculate the total loss for the current tower.
        total_loss = tf.add_n(losses, name='total_loss')

        # Attach a scalar summary to all individual losses and the total loss; do the
        # same for the averaged version of the losses.
        for l in losses + [total_loss]:
            # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
            # session. This helps the clarity of presentation on tensorboard.

            loss_name = re.sub('%s_[0-9]*/' % 'tower', '', l.op.name)
            tf.summary.scalar(loss_name, l)

        return total_loss




    def get_predict_flow_forward_backward(self,predict_flows,network_input_labels,concatenated_FB_images):
        batch_size = predict_flows[0].get_shape().as_list()[0]
        batch_half = batch_size // 2

        predict_flow = predict_flows[0]
        predict_flow_ref3 = predict_flows[1]
        predict_flow_ref2 = predict_flows[2]
        predict_flow_ref1 = predict_flows[3]

        # for other losses, we only consider forward flow
        predict_flow_forward = predict_flow[0:batch_half,:,:,:]
        predict_flow_backward = predict_flow[batch_half:batch_size,:,:,:]

        predict_flow_forward_ref3 = predict_flow_ref3[0:batch_half,:,:,:]
        predict_flow_backward_ref3 = predict_flow_ref3[batch_half:batch_size,:,:,:]

        predict_flow_forward_ref2 = predict_flow_ref2[0:batch_half,:,:,:]
        predict_flow_backward_ref2 = predict_flow_ref2[batch_half:batch_size,:,:,:]

        predict_flow_forward_ref1 = predict_flow_ref1[0:batch_half,:,:,:]
        predict_flow_backward_ref1 = predict_flow_ref1[batch_half:batch_size,:,:,:]


        tf.summary.image('flow_u_1',network_input_labels[:,:,:,0:1])
        tf.summary.image('flow_v_1',network_input_labels[:,:,:,1:2])

        tf.summary.image('input_image1',concatenated_FB_images[:,:,:,0:3])
        tf.summary.image('input_image2',concatenated_FB_images[:,:,:,4:7])
        tf.summary.image('depth_image1',tf.expand_dims(concatenated_FB_images[:,:,:,3],axis=-1))
        tf.summary.image('depth_image2',tf.expand_dims(concatenated_FB_images[:,:,:,7],axis=-1))

       
        return {
            'predict_flow': [predict_flow_forward, predict_flow_backward],
            'predict_flow_ref3': [predict_flow_forward_ref3,predict_flow_backward_ref3],
            'predict_flow_ref2': [predict_flow_forward_ref2, predict_flow_backward_ref2],
            'predict_flow_ref1': [predict_flow_forward_ref1, predict_flow_backward_ref1]
        }

    def write_flows_concatenated_side_by_side(self,network_input_labels,predict_flow2):
        concated_flows_u = tf.concat([network_input_labels[:,:,:,0:1],predict_flow2[:,:,:,0:1]],axis=-2)
        concated_flows_v = tf.concat([network_input_labels[:,:,:,1:2],predict_flow2[:,:,:,1:2]],axis=-2)

        tf.summary.image('gt_predict_flow_u',concated_flows_u)
        tf.summary.image('gt_predict_flow_v',concated_flows_v)

    def get_network_input_forward(self,image_batch,label_batch):
        return image_batch[:,0,:,:,:], label_batch[:,0,:,:,:]

    def get_network_input_backward(self,image_batch,label_batch):
        return image_batch[:,1,:,:,:], label_batch[:,1,:,:,:]

    def average_gradients(self,tower_grads):
        """Calculate the average gradient for each shared variable across all towers.
        Note that this function provides a synchronization point across all towers.
        Args:
            tower_grads: List of lists of (gradient, variable) tuples. The outer list
            is over individual gradients. The inner list is over the gradient
            calculation for each tower.
        Returns:
            List of pairs of (gradient, variable) where the gradient has been averaged
            across all towers.
        """
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            # Note that each grad_and_vars looks like the following:
            #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            grads = []
            for g, _ in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)

                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)

            # Average over the 'tower' dimension.
            grad = tf.concat(axis=0, values=grads)
            grad = tf.reduce_mean(grad, 0)

            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads




    def log(self,message=' '):
        print(message)





################################# MAIN ######################################

reader = DatasetReader()
train_iterator,test_iterator = reader.preprocess()
reader.train(train_iterator,test_iterator)
