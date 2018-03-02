import io
import network
import numpy as np
from   PIL import Image
import tensorflow as tf
import helpers as hpl
import losses_helper
import tensorflow.contrib.slim as slim
from tensorflow.python import debug as tf_debug
from tensorflow.python.client import device_lib
import time
from datetime import datetime
# import ijremote

# things to check before running the script
'''
    1) The self.train_type variable
    2) self.module variable
    3) training iteration loop starting from 0 to self.iteraions
    4) test iteration should be starting from 0 to self.test_iterations
'''


class DatasetReader:


    def main(self, features_train, features_test):
            self.global_step = tf.train.get_or_create_global_step()
            self.batch_size = 16
            self.total_iterations = 100000
            self.module = 'driving'
            self.ckpt_number = 6520
            self.train_start_iteration = self.ckpt_number + 1
            # 0 means only driving dataset.
            self.train_type = ['conv10/train','conv10/test']
            self.ckpt_load_path = 'conv10/train'

            # 50 iterations = 1 epoch ( i.e total_items=3136/batch_size=64 )
            self.test_iterations = 2
            # self.batch_size = 1
            # self.iterations = 1
            # self.module = 'driving'
            # self.ckpt_number = 3999




            self.train_imageBatch, self.train_labelBatch = tf.train.shuffle_batch(
                                                    [ features_train['input_n'], 
                                                    features_train['label_n']],
                                                    batch_size=self.batch_size,
                                                    capacity=100,
                                                    num_threads=10,
                                                    min_after_dequeue=6)
            self.test_imageBatch, self.test_labelBatch = tf.train.shuffle_batch(
                                                    [ features_test['input_n'], 
                                                    features_test['label_n']],
                                                    batch_size=self.batch_size,
                                                    capacity=100,
                                                    num_threads=10,
                                                    min_after_dequeue=6)

            with tf.name_scope('create_graph'):

                # self.X = tf.placeholder(dtype=tf.float32, shape=(self.batch_size, 224, 384, 6))
                # self.Y = tf.placeholder(dtype=tf.float32, shape=(self.batch_size, 224, 384, 2))
                self.X = self.train_imageBatch
                self.Y = self.train_labelBatch
                tf.summary.histogram('X',self.X)
                tf.summary.histogram('Y',self.Y)




                # build the network her
                predict_flow5, predict_flow2 = network.train_network(self.X)
                tf.summary.histogram('pflow',predict_flow2)
                # self.mse = tf.reduce_mean(network.change_nans_to_zeros(tf.sqrt(tf.reduce_sum((predict_flow2-self.Y)**2)+1e-3)))


                self.mse = losses_helper.mse_loss(self.Y,predict_flow2)
                
                MOVING_AVERAGE_DECAY = 0.9999
                variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, self.global_step)
                variables_averages_op = variable_averages.apply(tf.trainable_variables())

            sess = tf.InteractiveSession()
            self.saver = tf.train.Saver()

            tf.summary.scalar('MSE', self.mse)



            # learning rate decay
            decay_steps = self.total_iterations
            start_learning_rate = 0.0001
            end_learning_rate = 0.000001
            power = 6

            learning_rate = tf.train.polynomial_decay(start_learning_rate, self.global_step,
                                                      decay_steps, end_learning_rate,
                                                      power=power)


            tf.summary.scalar('learning_rate_decay', learning_rate)
            self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.mse)


            self.merged_summary_op = tf.summary.merge_all()
            self.summary_writer_train = tf.summary.FileWriter('./tb/'+self.module+'/'+self.train_type[0]+'/',graph=tf.get_default_graph())
            self.summary_writer_test = tf.summary.FileWriter('./tb/'+self.module+'/'+self.train_type[1]+'/',graph=tf.get_default_graph())
            sess.run([tf.global_variables_initializer(),tf.local_variables_initializer()])
            # tf.train.latest_checkpoint('./ckpt/'+self.module+'/'+self.train_type+'/')
            # self.load_model_ckpt(sess,self.ckpt_number)


            with tf.control_dependencies([self.optimizer, variables_averages_op]):
                train_op = tf.no_op(name='train')


            self.perform_learning(train_op)



            # self.run_network(sess)


    def perform_learning(self,train_op):
        class _LoggerHook(tf.train.SessionRunHook):
          """Logs loss and runtime."""

          def __init__(self,mse,batch_size):
            self.mse = mse
            self.batch_size = batch_size

          def begin(self):
            self._step = -11
            self._start_time = time.time()

          def before_run(self, run_context):
            self._step += 1
            return tf.train.SessionRunArgs(self.mse)  # Asks for loss value.

          def after_run(self, run_context, run_values):
            if self._step % 10 == 0:
              current_time = time.time()
              duration = current_time - self._start_time
              self._start_time = current_time

              loss_value = run_values.results
              examples_per_sec = 10 * self.batch_size / duration
              sec_per_batch = float(duration / 10)

              format_str = ('%s: step %d, loss = %.15f (%.1f examples/sec; %.3f '
                            'sec/batch)')
              print (format_str % (datetime.now(), self._step, np.log10(loss_value),
                                   examples_per_sec, sec_per_batch))



        ckpt_directory = './ckpt/'+self.module+'/'+self.train_type[0]+'/'
        with tf.train.MonitoredTrainingSession(
            checkpoint_dir='./ckpt/driving/conv10_cont/train/',
            hooks=[tf.train.StopAtStepHook(last_step=self.total_iterations),
                   tf.train.NanTensorHook(self.mse),
                   _LoggerHook(self.mse,self.batch_size)],
            config=tf.ConfigProto(
                log_device_placement=False)) as mon_sess:
          while not mon_sess.should_stop():
            mon_sess.run(train_op)


    def start_coordinators(self,sess):


        # initialize the threads coordinator
        self.coord = tf.train.Coordinator()

        # start enqueing the data to be dequeued for batch training
        threads = tf.train.start_queue_runners(sess, coord=self.coord)

        # print(a['label'].eval())
        # np.set_printoptions(threshold=np.nan)

        return threads

    def stop_coordinators(self,threads):
        self.summary_writer_train.close()
        self.summary_writer_test.close()

        # finalise
        self.coord.request_stop()
        self.coord.join(threads)



    def run_network(self,sess,data=None):

        threads = self.start_coordinators(sess)
        # bias = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'down_convs/conv1x/bias:0')[0]
        # print('printingggg bias')
        # print(bias.eval())
        # print('printedgggg bias')

        # print(data['disp1'].eval())
        self.train_model(sess)

        # print(sess.run(data['disp1']))
        # print(sess.run(self.mse))
        # print('tozi')
        # print(sess.run(self.mse))
    
        self.stop_coordinators(threads)

    def show_images(self,train_batch,i): 
        r1, r2 = np.split(train_batch,2,axis=3)

        imgg1 = np.squeeze(r1[:,:,:,0:3]) * 255
        imgg2 = np.squeeze(r2[:,:,:,0:3]) * 255

        imgg1 = imgg1.astype(np.uint8)
        imgg2 = imgg2.astype(np.uint8)
        im1 = Image.fromarray(imgg1,'RGB')
        im2 = Image.fromarray(imgg2,'RGB')
        im1.save('./img/abc'+str(i)+'.png')
        im2.save('./img/abc'+str(i)+'.png')
    def show_optical_flow(self,label_batch,i): 

        factor = 0.4
        input_size = int(960 * factor), int(540 * factor)

        opt_u = np.squeeze(label_batch[:,:,:,0]) * input_size[0]
        opt_v = np.squeeze(label_batch[:,:,:,1]) * input_size[1]

        # ijremote.setImage('InputU',opt_u)
        # ijremote.setImage('InputV',opt_v)

        opt_u = Image.fromarray(opt_u)
        opt_v = Image.fromarray(opt_v)

        opt_u.save('./img/flow_u'+str(i)+'.tiff')
        opt_v.save('./img/flow_v'+str(i)+'.tiff')
        opt_u.show()
        opt_v.show()

    def train_model(self,sess):
        for i in range(self.train_start_iteration,self.total_iterations + self.train_start_iteration):

            train_batch_xs, train_batch_ys = sess.run([self.train_imageBatch, self.train_labelBatch])
            # self.show_images(train_batch_xs,i)
            # self.show_optical_flow(train_batch_ys,i)
            # batch_xs = batch_xs / np.abs(batch_xs).max()

            # batch_ys = batch_ys * 0.4

            # batch_ys[:,:,:,0][ batch_ys[:,:,:,0] > 216 ] = 0
            # batch_ys[:,:,:,1][ batch_ys[:,:,:,1] > 384 ] = 0

            # batch_ys[:,:,:,2][ batch_ys[:,:,:,2] > 500 ] = 0

            # batch_ys = batch_ys / np.abs(batch_ys).max()

            summary, opt,  epoch_loss = sess.run([self.merged_summary_op, self.optimizer, self.mse],feed_dict={self.X: train_batch_xs, self.Y: train_batch_ys})
            # loss = loss + epoch_loss

            print('Iteration: '+str(i)+'     Loss = ',str(epoch_loss))
            # print('x min val = ' + str(np.abs(train_batch_xs).min()))
            # print('x max val = ' + str(np.abs(train_batch_xs).max()))
            # print('y min val = ' + str(np.abs(train_batch_ys).min()))
            # print('y max val = ' + str(np.abs(train_batch_ys).max()))
            # print('')

            if i%40==0:
                # perform testing after each 10 epochs. 1 epoch = 133 iterations
                self.perform_test_loss(sess,i)
                print('wrote summary test '+str(i))
                self.summary_writer_test.add_summary(summary, i)
                # self.save_model(sess,i)
            if i%10==0:
                print('wrote summary '+str(i))
                self.summary_writer_train.add_summary(summary, i)
            if i%100==0 or i==self.total_iterations - 1:
                self.save_model(sess,i)


    def perform_test_loss(self,sess,i):
        print('checking test loss...')

        for looper in range(0,self.test_iterations):
            test_batch_xs, test_batch_ys = sess.run([self.test_imageBatch, self.test_labelBatch])
            summary,  epoch_loss = sess.run([self.merged_summary_op, self.mse],feed_dict={self.X: test_batch_xs, self.Y: test_batch_ys})
            print('test loss = '+str(epoch_loss))

        print('Iteration Test '+str(i))
        print('')

        self.summary_writer_test.add_summary(summary, i)
 
    def save_model(self,sess,i):
        print('saving checkpoint '+ str(i))
        self.saver.save(sess, 'ckpt/'+self.module+'/'+self.train_type[0]+'/model_ckpt_'+str(i)+'.ckpt')
        # self.saver.save(sess, 'ckpt/'+self.module+'/'+self.train_type+'/model_ckpt_'+str(i)+'.ckpt', global_step=i)


    def load_model_ckpt(self,sess,i):
        self.saver.restore(sess, './ckpt/'+self.module+'/'+self.ckpt_load_path+'/model_ckpt_'+str(i)+'.ckpt')
