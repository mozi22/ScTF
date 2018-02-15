import io
import network
import numpy as np
from   PIL import Image
import tensorflow as tf
import helpers as hpl
import tensorflow.contrib.slim as slim
from tensorflow.python import debug as tf_debug


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
            self.batch_size = 64
            self.total_iterations = 2000
            self.module = 'driving'
            self.ckpt_number = 9999
            self.train_start_iteration = self.ckpt_number + 2
            # 0 means only driving dataset.
            self.train_type = ['3c/train','3c/test']
            self.ckpt_load_path = '3/train'

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

                self.X = tf.placeholder(dtype=tf.float32, shape=(self.batch_size, 224, 384, 8))
                self.Y = tf.placeholder(dtype=tf.float32, shape=(self.batch_size, 224, 384, 3))
                tf.summary.histogram('X',self.X)
                tf.summary.histogram('Y',self.Y)




                # build the network her
                predict_flow5, predict_flow2 = network.train_network(self.X)
                tf.summary.histogram('pflow',predict_flow2)
                # self.mse = tf.reduce_mean(network.change_nans_to_zeros(tf.sqrt(tf.reduce_sum((predict_flow2-self.Y)**2)+1e-3)))

                self.mse = tf.losses.mean_squared_error(self.Y,predict_flow2)

            sess = tf.InteractiveSession()
            self.saver = tf.train.Saver()

            tf.summary.scalar('MSE', self.mse)
            tf.summary.scalar('Global_Step', self.global_step)



            # learning rate decay
            decay_steps = self.total_iterations
            starter_learning_rate = 0.00009
            end_learning_rate = 0.00001
            power = 4

            learning_rate = tf.train.polynomial_decay(starter_learning_rate, self.global_step,
                                                      decay_steps, end_learning_rate,
                                                      power=power)


            tf.summary.scalar('learning_rate_decay', learning_rate)
            self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.mse,global_step=self.global_step)
            sess.run([tf.global_variables_initializer(),tf.local_variables_initializer()])
            # tf.train.latest_checkpoint('./ckpt/'+self.module+'/'+self.train_type+'/')
            self.load_model_ckpt(sess,self.ckpt_number)
            self.run_network(sess)

    def start_coordinators(self,sess):
        self.merged_summary_op = tf.summary.merge_all()
        self.summary_writer_train = tf.summary.FileWriter('./tb/'+self.module+'/'+self.train_type[0]+'/',graph=tf.get_default_graph())
        self.summary_writer_test = tf.summary.FileWriter('./tb/'+self.module+'/'+self.train_type[1]+'/',graph=tf.get_default_graph())


        # initialize the threads coordinator
        self.coord = tf.train.Coordinator()

        # start enqueing the data to be dequeued for batch training
        threads = tf.train.start_queue_runners(sess, coord=self.coord)

        # print(a['label'].eval())        
        # np.set_printoptions(threshold=np.nan)

        return threads

    def stop_coordinators(self,summary_writer,threads):
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


    def train_model(self,sess):
        for i in range(self.train_start_iteration,self.total_iterations + self.train_start_iteration):

            train_batch_xs, train_batch_ys = sess.run([self.train_imageBatch, self.train_labelBatch])

            # batch_xs = batch_xs / np.abs(batch_xs).max()

            # batch_ys = batch_ys * 0.4

            # batch_ys[:,:,:,0][ batch_ys[:,:,:,0] > 216 ] = 0
            # batch_ys[:,:,:,1][ batch_ys[:,:,:,1] > 384 ] = 0

            # batch_ys[:,:,:,2][ batch_ys[:,:,:,2] > 500 ] = 0

            # batch_ys = batch_ys / np.abs(batch_ys).max()

            summary, opt,  epoch_loss = sess.run([self.merged_summary_op, self.optimizer, self.mse],feed_dict={self.X: train_batch_xs, self.Y: train_batch_ys})
            # loss = loss + epoch_loss

            print('Iteration: '+str(i)+'     Loss = ',str(epoch_loss))
            print('x min val = ' + str(np.abs(train_batch_xs).min()))
            print('x max val = ' + str(np.abs(train_batch_xs).max()))
            print('y min val = ' + str(np.abs(train_batch_ys).min()))
            print('y max val = ' + str(np.abs(train_batch_ys).max()))
            print('')

            if i%40==0:
                # perform testing after each 10 epochs. 1 epoch = 133 iterations
                self.perform_test_loss(sess,i)
                print('wrote summary test '+str(i))
                self.summary_writer_test.add_summary(summary, i)
                self.save_model(sess,i)
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
