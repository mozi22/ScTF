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
import re
# import ijremote
from six.moves import xrange
import os

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']



# these variables can be tuned to help training
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('TRAIN_DIR', './ckpt/driving/multi_gpu/',
                           """Directory where to write event logs """
                           """and checkpoint.""")

tf.app.flags.DEFINE_string('TOWER_NAME', 'tower',
                           """The name of the tower """)

tf.app.flags.DEFINE_integer('MAX_STEPS', 10000,
                            """Number of batches to run.""")


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

tf.app.flags.DEFINE_integer('SHUFFLE_BATCH_THREADS', 48,
                            """How many elements will be there in the queue to be dequeued.""")

tf.app.flags.DEFINE_integer('SHUFFLE_BATCH_MIN_AFTER_DEQUEUE', 10,
                            """How many elements will be there in the queue to be dequeued.""")

tf.app.flags.DEFINE_integer('NUM_GPUS', len(get_available_gpus()),
                            """How many GPUs to use.""")

tf.app.flags.DEFINE_float('MOVING_AVERAGE_DECAY', 0.9999,
                            """How fast the learning rate should go down.""")




# Polynomial Learning Rate

tf.app.flags.DEFINE_float('START_LEARNING_RATE', 0.001,
                            """Where to start the learning.""")
tf.app.flags.DEFINE_float('END_LEARNING_RATE', 0.000001,
                            """Where to end the learning.""")
tf.app.flags.DEFINE_float('POWER', 4,
                            """How fast the learning rate should go down.""")




class DatasetReader:

    def train(self,features_train):
        global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(0), trainable=False)

        num_batches_per_epoch = (FLAGS.EXAMPLES_PER_EPOCH_TRAIN / FLAGS.BATCH_SIZE)
        decay_steps = int(num_batches_per_epoch * FLAGS.NUM_EPOCHS_PER_DECAY)

        start_learning_rate = FLAGS.START_LEARNING_RATE
        end_learning_rate = FLAGS.END_LEARNING_RATE
        power = FLAGS.POWER

        learning_rate = tf.train.polynomial_decay(start_learning_rate, global_step,
                                                  decay_steps, end_learning_rate,
                                                  power=power)


        opt = tf.train.AdamOptimizer(learning_rate)
#                               .minimize(self.mse,global_step=self.global_step)


    

        images, labels = tf.train.shuffle_batch(
                            [ features_train['input_n'], 
                            features_train['label_n']],
                            batch_size=FLAGS.BATCH_SIZE,
                            capacity=FLAGS.SHUFFLE_BATCH_QUEUE_CAPACITY,
                            num_threads=FLAGS.SHUFFLE_BATCH_THREADS,
                            min_after_dequeue=FLAGS.SHUFFLE_BATCH_MIN_AFTER_DEQUEUE)
        
        batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue(
            [images, labels], capacity=FLAGS.SHUFFLE_BATCH_QUEUE_CAPACITY * FLAGS.NUM_GPUS)
    
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


        summary_writer = tf.summary.FileWriter(FLAGS.TRAIN_DIR, sess.graph)



        # main loop
        for step in xrange(FLAGS.MAX_STEPS):
            start_time = time.time()
            _, loss_value = sess.run([train_op, loss])
            duration = time.time() - start_time


            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % 10 == 0:
                num_examples_per_step = FLAGS.BATCH_SIZE * FLAGS.NUM_GPUS
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = duration / FLAGS.NUM_GPUS

            format_str = ('%s: step %d, loss = %.15f (%.1f examples/sec; %.3f '
                          'sec/batch)')
            print (format_str % (datetime.now(), step, loss_value,
                                 examples_per_sec, sec_per_batch))

            if step % 100 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)

            # Save the model checkpoint periodically.
            if step % 1000 == 0 or (step + 1) == FLAGS.MAX_STEPS:
                checkpoint_path = os.path.join(FLAGS.TRAIN_DIR, 'model.ckpt')
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
        _ = losses_helper.mse_loss(labels,predict_flow2)

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
