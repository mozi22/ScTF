import os
import re
import time
import math
import network
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

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']



# Training Variables

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('TRAIN_DIR', './ckpt/driving/train_with_test/train',
                           """Directory where to write event logs """
                           """and checkpoint.""")

tf.app.flags.DEFINE_string('TEST_DIR', './ckpt/driving/train_with_test/test',
                           """Directory where to write event logs """
                           """and checkpoint.""")

tf.app.flags.DEFINE_boolean('LOAD_FROM_CKPT', False,
                            """Whether to log device placement.""")

tf.app.flags.DEFINE_boolean('DEBUG_MODE', False,
                            """Run training in Debug Mode.""")

tf.app.flags.DEFINE_string('TOWER_NAME', 'tower',
                           """The name of the tower """)

tf.app.flags.DEFINE_integer('MAX_STEPS', 10000,
                            """Number of batches to run.""")

tf.app.flags.DEFINE_boolean('LOG_DEVICE_PLACEMENT', False,
                            """Whether to log device placement.""")

tf.app.flags.DEFINE_integer('BATCH_SIZE', 16,
                            """How many samples are there in one epoch of testing.""")

tf.app.flags.DEFINE_integer('NUM_EPOCHS_PER_DECAY', 1,
                            """How many epochs per decay.""")

tf.app.flags.DEFINE_integer('SHUFFLE_BATCH_QUEUE_CAPACITY', 100,
                            """How many elements will be there in the queue to be dequeued.""")

tf.app.flags.DEFINE_integer('SHUFFLE_BATCH_THREADS', 48,
                            """How many elements will be there in the queue to be dequeued.""")

tf.app.flags.DEFINE_integer('SHUFFLE_BATCH_MIN_AFTER_DEQUEUE', 10,
                            """How many elements will be there in the queue to be dequeued.""")

tf.app.flags.DEFINE_integer('NUM_GPUS', len(get_available_gpus()),
                            """How many GPUs to use.""")

tf.app.flags.DEFINE_float('MOVING_AVERAGE_DECAY', 0.9999,
                            """How fast the learning rate should go down.""")

tf.app.flags.DEFINE_integer('TOTAL_TRAIN_EXAMPLES', 200,
                            """How many samples are there in one epoch of testing.""")


# Testing Variables

tf.app.flags.DEFINE_integer('TOTAL_TEST_EXAMPLES', 100,
                            """How many samples are there in one epoch of testing.""")

tf.app.flags.DEFINE_integer('TEST_BATCH_SIZE', 16,
                            """How many samples are there in one epoch of testing.""")

tf.app.flags.DEFINE_integer('TEST_AFTER_EPOCHS', 4,
                            """After how many epochs should the test phase start.""")


# Polynomial Learning Rate

tf.app.flags.DEFINE_float('START_LEARNING_RATE', 0.0009,
                            """Where to start the learning.""")
tf.app.flags.DEFINE_float('END_LEARNING_RATE', 0.0000005,
                            """Where to end the learning.""")
tf.app.flags.DEFINE_float('POWER', 3,
                            """How fast the learning rate should go down.""")

class DatasetReader:

    def __init__(self):
        # for testing
        self.X = tf.placeholder(dtype=tf.float32, shape=(FLAGS.TEST_BATCH_SIZE, 224, 384, 8))
        self.Y = tf.placeholder(dtype=tf.float32, shape=(FLAGS.TEST_BATCH_SIZE, 224, 384, 3))

        # gives the # of steps required to complete 1 epoch
        self.TRAIN_EPOCH = math.ceil(FLAGS.TOTAL_TRAIN_EXAMPLES / FLAGS.BATCH_SIZE)
        self.TEST_EPOCH = math.ceil(FLAGS.TOTAL_TEST_EXAMPLES / FLAGS.TEST_BATCH_SIZE)


    def train(self,features_train,features_test):

        global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(0), trainable=False)

        decay_steps = FLAGS.MAX_STEPS
        start_learning_rate = FLAGS.START_LEARNING_RATE
        end_learning_rate = FLAGS.END_LEARNING_RATE
        power = FLAGS.POWER

        learning_rate = tf.train.polynomial_decay(start_learning_rate, global_step,
                                                  decay_steps, end_learning_rate,
                                                  power=power)


        opt = tf.train.AdamOptimizer(learning_rate)
    
        images, labels = tf.train.shuffle_batch(
                            [ features_train['input_n'] , features_train['label_n'] ],
                            batch_size=FLAGS.BATCH_SIZE,
                            capacity=FLAGS.SHUFFLE_BATCH_QUEUE_CAPACITY,
                            num_threads=FLAGS.SHUFFLE_BATCH_THREADS,
                            min_after_dequeue=FLAGS.SHUFFLE_BATCH_MIN_AFTER_DEQUEUE,
                            enqueue_many=False)

        self.images_test, self.labels_test = tf.train.shuffle_batch(
                            [ features_test['input_n'] , features_test['label_n'] ],
                            batch_size=FLAGS.TEST_BATCH_SIZE,
                            capacity=FLAGS.SHUFFLE_BATCH_QUEUE_CAPACITY,
                            num_threads=FLAGS.SHUFFLE_BATCH_THREADS,
                            min_after_dequeue=FLAGS.SHUFFLE_BATCH_MIN_AFTER_DEQUEUE,
                            enqueue_many=False)
        
        batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue(
            [images, labels], capacity=FLAGS.SHUFFLE_BATCH_QUEUE_CAPACITY * FLAGS.NUM_GPUS)

        self.batch_queue_test = tf.contrib.slim.prefetch_queue.prefetch_queue(
            [self.images_test, self.labels_test], capacity=FLAGS.SHUFFLE_BATCH_QUEUE_CAPACITY * FLAGS.NUM_GPUS)
        
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
                self.loss = self.tower_loss(scope, image_batch, label_batch)

                # Reuse variables for the next tower.
                tf.get_variable_scope().reuse_variables()

                # Retain the summaries from the final tower.
                summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)


                # Calculate the gradients for the batch of data on this CIFAR tower.
                grads = opt.compute_gradients(self.loss)

                # Keep track of the gradients across all towers.
                tower_grads.append(grads)


        # We must calculate the mean of each gradient. Note that this is the
        # synchronization point across all towers.
        grads = self.average_gradients(tower_grads)

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
        self.summary_op = tf.summary.merge(summaries)

        # Build an initialization operation to run below.
        init = tf.global_variables_initializer()

        # Start running operations on the Graph. allow_soft_placement must be set to
        # True to build towers on GPU, as some of the ops do not have GPU
        # implementations.

        tf_config = tf.ConfigProto(allow_soft_placement=True,
            log_device_placement=FLAGS.LOG_DEVICE_PLACEMENT)

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


        if FLAGS.LOAD_FROM_CKPT == True:
            saver.restore(sess,tf.train.latest_checkpoint(FLAGS.TRAIN_DIR))


        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)

        # for debugging

        summary_writer = tf.summary.FileWriter(FLAGS.TRAIN_DIR, sess.graph)
        self.test_summary_writer = tf.summary.FileWriter(FLAGS.TEST_DIR, sess.graph)


        # just to make sure we start from where we left, if load_from_ckpt = True
        loop_start = tf.train.global_step(sess, global_step)
        loop_stop = loop_start + FLAGS.MAX_STEPS

        if FLAGS.DEBUG_MODE:
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)

        first_iteration = True

        # this will print in console for which time we are calculating the test loss.
        # first time or second time or third time and more
        test_loss_calculating_index = 1

        test_image_batch, test_label_batch = self.batch_queue_test.dequeue()
        self.test_image_batch, self.test_label_batch = self.get_network_input_forward(test_image_batch,test_label_batch)

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
                num_examples_per_step = FLAGS.BATCH_SIZE * FLAGS.NUM_GPUS
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = duration / FLAGS.NUM_GPUS
                first_iteration = False



            format_str = ('%s: step %d, loss = %.15f (%.1f examples/sec; %.3f '
                          'sec/batch)')
            self.log(message=(format_str % (datetime.now(), step, np.log10(loss_value),
                                 examples_per_sec, sec_per_batch)))

            if step % 10 == 0:
                summary_str = sess.run(self.summary_op)
                summary_writer.add_summary(summary_str, step)

                # testing summary writing
                image,label = sess.run([self.test_image_batch, self.test_label_batch])
                loss_value,summary_str = sess.run([self.loss,self.summary_op],feed_dict={self.X: image, self.Y: label})
                self.test_summary_writer.add_summary(summary_str, step)

            # Save the model checkpoint periodically.
            if step % 1000 == 0 or (step + 1) == FLAGS.MAX_STEPS:
                checkpoint_path = os.path.join(FLAGS.TRAIN_DIR, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)



        summary_writer.close()

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
        # network_input_images_back, network_input_labels_back = self.get_network_input_backward(images,labels)

        # FB = forward-backward
        # concatenated_FB_images = tf.concat([network_input_images,network_input_images_back],axis=0)

        # backward_flow_images = losses_helper.forward_backward_loss()

        # Build inference Graph. - forward flow

        predict_flow5, predict_flow2 = network.train_network(network_input_images)


        # Build inference Graph. - backward flow
        # Build the portion of the Graph calculating the losses. Note that we will
        # assemble the total_loss using a custom function below.

        # _ = losses_helper.forward_backward_loss(predict_flow2)

        batch_size = predict_flow2.get_shape().as_list()[0]
        batch_half = batch_size // 2

        # for other losses, we only consider forward flow
        # predict_flow2_forward = predict_flow2[0:batch_half,:,:,:]
        # predict_flow2_backward = predict_flow2[batch_half:batch_size,:,:,:]

        # predict_flow5_forward = predict_flow5[0:batch_half,:,:,:]
        # predict_flow5_backward = predict_flow5[batch_half:batch_size,:,:,:]
        tf.summary.image('flow_u_1',network_input_labels[:,:,:,0:1])
        tf.summary.image('flow_v_1',network_input_labels[:,:,:,1:2])

        # predict_flow2_label = losses_helper.downsample_label(network_input_labels)
        _ = losses_helper.endpoint_loss(network_input_labels,predict_flow2)
        # _ = losses_helper.photoconsistency_loss(network_input_images,predict_flow2_forward)
        # # _ = losses_helper.depth_consistency_loss(network_input_images,predict_flow2_forward)

        # scale_invariant_gradient_image_gt = losses_helper.scale_invariant_gradient(network_input_labels,
        #                                                                         np.array([1,2,4,8,16]),
        #                                                                         np.array([1,1,1,1,1]))

        # scale_invariant_gradient_image_pred = losses_helper.scale_invariant_gradient(predict_flow2_forward,
        #                                                                         np.array([1,2,4,8,16]),
        #                                                                         np.array([1,1,1,1,1]))

        # _ = losses_helper.scale_invariant_gradient_loss(scale_invariant_gradient_image_pred,scale_invariant_gradient_image_gt,0.0001)

        predict_flow5_label = losses_helper.downsample_label(network_input_labels,
                                        size=[7,12],
                                        factorU=0.031,
                                        factorV=0.026)

        _ = losses_helper.endpoint_loss(predict_flow5_label,predict_flow5)
        # _ = losses_helper.depth_loss(predict_flow5_label,predict_flow5)

        # tf.summary.histogram('prediction_flow2_forward',predict_flow2_forward)
        # tf.summary.histogram('prediction_flow5_forward',predict_flow5_forward)
        # tf.summary.histogram('prediction_flow2_backward',predict_flow2_backward)
        # tf.summary.histogram('prediction_flow5_backward',predict_flow5_backward)

        # tf.summary.histogram('gt_flow2_forward',network_input_labels)


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
