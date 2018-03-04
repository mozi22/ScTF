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
import math
from datetime import datetime
# import ijremote

# things to check before running the script
'''
    1) The self.train_type variable
    2) self.module variable
    3) training iteration loop starting from 0 to self.iteraions
    4) test iteration should be starting from 0 to self.test_iterations
'''

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('TRAIN_DIR', '/ckpt/driving/multi_gpu/',
                           """Directory where to write event logs """
                           """and checkpoint.""")

tf.app.flags.DEFINE_string('TOWER_NAME', 'tower',
                           """The name of the tower """)

tf.app.flags.DEFINE_integer('MAX_STEPS', 1000000,
                            """Number of batches to run.""")

tf.app.flags.DEFINE_integer('NUM_GPUS', 1,
                            """How many GPUs to use.""")

tf.app.flags.DEFINE_boolean('LOG_DEVICE_PLACEMENT', False,
                            """Whether to log device placement.""")

tf.app.flags.DEFINE_integer('EXAMPLES_PER_EPOCH_TRAIN', 200,
                            """How many samples are there in one epoch of training.""")

tf.app.flags.DEFINE_integer('EXAMPLES_PER_EPOCH_TEST', 100,
                            """How many samples are there in one epoch of testing.""")

tf.app.flags.DEFINE_integer('BATCH_SIZE', 16,
                            """How many samples are there in one epoch of testing.""")

tf.app.flags.DEFINE_integer('NUM_EPOCHS_PER_DECAY', 1,
                            """How many epochs per decay.""")

tf.app.flags.DEFINE_integer('SHUFFLE_BATCH_QUEUE_CAPACITY', 100,
                            """How many elements will be there in the queue to be dequeued.""")

tf.app.flags.DEFINE_integer('SHUFFLE_BATCH_THREADS', 10,
                            """How many elements will be there in the queue to be dequeued.""")

tf.app.flags.DEFINE_integer('SHUFFLE_BATCH_MIN_AFTER_DEQUEUE', 10,
                            """How many elements will be there in the queue to be dequeued.""")

tf.app.flags.DEFINE_integer('NUM_GPUS', len(get_available_gpus()),
                            """How fast the learning rate should go down.""")

tf.app.flags.DEFINE_float('MOVING_AVERAGE_DECAY', 0.9999,
                            """How fast the learning rate should go down.""")




# Polynomial Learning Rate

tf.app.flags.DEFINE_float('START_LEARNING_RATE', 0.0005,
                            """Where to start the learning.""")
tf.app.flags.DEFINE_float('END_LEARNING_RATE', 0.000001,
                            """Where to end the learning.""")
tf.app.flags.DEFINE_float('POWER', 4,
                            """How fast the learning rate should go down.""")




class DatasetReader:

    def testing_flat(self):
        print(FLAGS.max_steps)

    def main(self, features_train, features_test):
            # self.sess = tf.InteractiveSession()



        self.global_step = tf.train.get_or_create_global_step()
        self.batch_size = 16
        self.total_iterations = 200000
        self.module = 'driving'
        self.ckpt_number = 0
        self.train_start_iteration = self.ckpt_number + 1
        # 0 means only driving dataset.
        self.train_type = ['conv10_cont/train','conv10_cont/test']
        self.ckpt_load_path = 'conv10_cont/train'

        self.train_location = './ckpt/driving/cifar_wid_test/train/'

        # 50 iterations = 1 epoch ( i.e total_items=3136/batch_size=64 )
        self.test_iterations = math.ceil(100 / 16)
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
            
            # calculate the moving average of trainable variables in the network.
            MOVING_AVERAGE_DECAY = 0.9999
            variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, self.global_step)
            variables_averages_op = variable_averages.apply(tf.trainable_variables())


        tf.summary.scalar('MSE', self.mse)
        tf.summary.scalar('global_step', self.global_step)



        # learning rate decay
        decay_steps = self.total_iterations
        start_learning_rate = 0.0005
        end_learning_rate = 0.000001
        power = 4

        learning_rate = tf.train.polynomial_decay(start_learning_rate, self.global_step,
                                                  decay_steps, end_learning_rate,
                                                  power=power)


        tf.summary.scalar('learning_rate_decay', learning_rate)
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.mse,global_step=self.global_step)


        # self.merged_summary_op = tf.summary.merge_all()
        # self.summary_writer_train = tf.summary.FileWriter('./tb/'+self.module+'/'+self.train_type[0]+'/',graph=tf.get_default_graph())
        # self.summary_writer_test = tf.summary.FileWriter(self.train_location,graph=tf.get_default_graph())
        # sess.run([tf.global_variables_initializer(),tf.local_variables_initializer()])
        # tf.train.latest_checkpoint('./ckpt/'+self.module+'/'+self.train_type+'/')
        # self.load_model_ckpt(sess,self.ckpt_number)


        with tf.control_dependencies([self.optimizer, variables_averages_op]):
            train_op = tf.no_op(name='train')


        self.perform_learning(train_op)



        # self.run_network(sess)


    def train(self,features_train):
        global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(0), trainable=False)

        num_batches_per_epoch = (FLAGS.EXAMPLES_PER_EPOCH_TRAIN / FLAGS.BATCH_SIZE)
        decay_steps = int(num_batches_per_epoch * FLAGS.NUM_EPOCHS_PER_DECAY)
        start_learning_rate = FLAGS.START_LEARNING_RATE
        end_learning_rate = FLAGS.END_LEARNING_RATE
        power = FLAGS.POWER

        learning_rate = tf.train.polynomial_decay(start_learning_rate, self.global_step,
                                                  decay_steps, end_learning_rate,
                                                  power=power)


        self.optimizer = tf.train.AdamOptimizer(learning_rate)
#                               .minimize(self.mse,global_step=self.global_step)



        self.train_imageBatch, self.train_labelBatch = tf.train.shuffle_batch(
                                                [ features_train['input_n'], 
                                                features_train['label_n']],
                                                batch_size=self.batch_size,
                                                capacity=FLAGS.SHUFFLE_BATCH_QUEUE_CAPACITY,
                                                num_threads=FLAGS.SHUFFLE_BATCH_THREADS,
                                                min_after_dequeue=FLAGS.SHUFFLE_BATCH_MIN_AFTER_DEQUEUE)
        
        batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue(
            [images, labels], capacity=FLAGS.QUEUE_CAPACITY * FLAGS.NUM_GPUS)
    
        tower_grads = []
        with tf.variable_scope(tf.get_variable_scope()):
          for i in xrange(FLAGS.NUM_GPUS):
            with tf.device('/gpu:%d' % i):
              with tf.name_scope('%s_%d' % ('tower', i)) as scope:
                # Dequeues one batch for the GPU
                image_batch, label_batch = batch_queue.dequeue()
                # Calculate the loss for one tower of the CIFAR model. This function
                # constructs the entire CIFAR model but shares the variables across
                # all towers.
                loss = self.tower_loss(scope, image_batch, label_batch)

                # Reuse variables for the next tower.
                tf.get_variable_scope().reuse_variables()

                # Retain the summaries from the final tower.
                summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

                # Calculate the gradients for the batch of data on this CIFAR tower.
                grads = opt.compute_gradients(loss)

                # Keep track of the gradients across all towers.
                tower_grads.append(grads)


        # We must calculate the mean of each gradient. Note that this is the
        # synchronization point across all towers.
        grads = average_gradients(tower_grads)

        # Add a summary to track the learning rate.
        summaries.append(tf.summary.scalar('learning_rate', learning_rate))


        # Add histograms for gradients.
        for grad, var in grads:
            if grad is not None:
                summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))


        # Apply the gradients to adjust the shared variables.
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        # Add histograms for trainable variables.
        for var in tf.trainable_variables():
            summaries.append(tf.summary.histogram(var.op.name, var))

        # Track the moving averages of all trainable variables.
        variable_averages = tf.train.ExponentialMovingAverage(
            FLAGS.MOVING_AVERAGE_DECAY, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())

        # Group all updates to into a single train op.
        train_op = tf.group(apply_gradient_op, variables_averages_op)

        # Create a saver.
        saver = tf.train.Saver(tf.global_variables())

        # Build the summary operation from the last tower summaries.
        summary_op = tf.summary.merge(summaries)

        # Build an initialization operation to run below.
        init = tf.global_variables_initializer()

        # Start running operations on the Graph. allow_soft_placement must be set to
        # True to build towers on GPU, as some of the ops do not have GPU
        # implementations.
        sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=FLAGS.LOG_DEVICE_PLACEMENT))
        sess.run(init)


        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)


        summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)

        for step in xrange(FLAGS.MAX_STEPS):
            start_time = time.time()
            _, loss_value = sess.run([train_op, loss])
            duration = time.time() - start_time


        assert not np.isnan(loss_value), 'Model diverged with loss = NaN'


        if step % 10 == 0:
            num_examples_per_step = FLAGS.BATCH_SIZE * FLAGS.NUM_GPUS
            examples_per_sec = num_examples_per_step / duration
            sec_per_batch = duration / FLAGS.NUM_GPUS

        format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                      'sec/batch)')
        print (format_str % (datetime.now(), step, loss_value,
                             examples_per_sec, sec_per_batch))

      if step % 100 == 0:
        summary_str = sess.run(summary_op)
        summary_writer.add_summary(summary_str, step)

      # Save the model checkpoint periodically.
      if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)

    def tower_loss(self,scope, images, labels):
        """Calculate the total loss on a single tower running the CIFAR model.
        Args:
        scope: unique prefix string identifying the CIFAR tower, e.g. 'tower_0'
        images: Images. 4D tensor of shape [batch_size, height, width, 3].
        labels: Labels. 1D tensor of shape [batch_size].
        Returns:
         Tensor of shape [] containing the total loss for a batch of data
        """

        # Build inference Graph.
        predict_flow5, predict_flow2 = network.train_network(images)

        # Build the portion of the Graph calculating the losses. Note that we will
        # assemble the total_loss using a custom function below.
        self.mse = losses_helper.mse_loss(labels,predict_flow2)

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

    def perform_learning(self,train_op):
        class _LoggerHook(tf.train.SessionRunHook):
          """Logs loss and runtime."""

          def __init__(self,mse,batch_size):
            self.mse = mse
            self.batch_size = batch_size

          def begin(self):
            self._step = -1
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
              print (format_str % (datetime.now(), self._step, loss_value,
                                   examples_per_sec, sec_per_batch))


    
        with tf.train.MonitoredTrainingSession(
            checkpoint_dir=self.train_location,
            hooks=[
                    #tf.train.StopAtStepHook(last_step=self.total_iterations),
                    tf.train.NanTensorHook(self.mse),
                   _LoggerHook(self.mse,self.batch_size)
                   ],
            config=tf.ConfigProto(
                log_device_placement=False),
            save_checkpoint_secs=60) as mon_sess:
    
            # i = 0
            # threads = self.start_coordinators(mon_sess)
            while not mon_sess.should_stop():
                mon_sess.run(train_op)
            # for i in range(self.train_start_iteration,self.total_iterations + self.train_start_iteration):

            #     if i%200 == 0:
            #         print('performing test')
            #         print('')
            #         self.perform_test_loss(mon_sess,i)
            #         continue

            #     i = i + 1

            # self.stop_coordinators(threads)

    def start_coordinators(self,sess):


        # initialize the threads coordinator
        self.coord = tf.train.Coordinator()

        # start enqueing the data to be dequeued for batch training
        threads = tf.train.start_queue_runners(sess, coord=self.coord)

        # print(a['label'].eval())
        # np.set_printoptions(threshold=np.nan)

        return threads

    def stop_coordinators(self,threads):
        print('stop krdia')
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
