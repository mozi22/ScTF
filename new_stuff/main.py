from tensorflow.contrib import slim
from datasets import flowers
import tensorflow as tf
import helpers


def test_var(var,g):
    sess = tf.Session(graph=g)
    # initialize the variables
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    # initialize the threads coordinator
    coord = tf.train.Coordinator()
    # start enqueing the data to be dequeued for batch training
    threads = tf.train.start_queue_runners(sess, coord=coord)

    print(var.eval(session=sess))
    # finalise
    coord.request_stop()
    coord.join(threads)

# This might take a few minutes.
train_dir = '/tmp/tfslim_model/'
print('Will save model to %s' % train_dir)

g = tf.Graph()
with g.as_default():
    tf.logging.set_verbosity(tf.logging.INFO)


    dataset = flowers.get_split('train', './tffiles')
    image1, image2, label = helpers.load_batch(dataset)
    img_pair = tf.concat([image1,image2],axis=-1,name="img_pair")
    # combine the images here.
    print(img_pair)



    # # Create the model:
    X = tf.placeholder(dtype=tf.float32, shape=(4, 96, 128, 6))
    Y = tf.placeholder(dtype=tf.float32, shape=(4, 24, 32, 2))

    predict_flow5, predict_flow2 = helpers.my_cnn(X)

    # # measure of error of our model
    # # this needs to be minimised by adjusting W and b
    mse = tf.reduce_mean(tf.squared_difference(predict_flow2, Y))

    # # define training step which minimizes cross entropy
    optimizer = tf.train.AdamOptimizer(1e-4).minimize(mse)

    print(predict_flow2)
    #
    # # Specify the loss function:
    # one_hot_labels = slim.one_hot_encoding(labels, dataset.num_classes)
    # slim.losses.softmax_cross_entropy(logits, one_hot_labels)
    # total_loss = slim.losses.get_total_loss()

    mse = tf.reduce_mean(tf.squared_difference(predict_flow2, Y))
    # # Create some summaries to visualize the training process:
    # tf.summary.scalar('losses/Total Loss', total_loss)
  
    # # Specify the optimizer and create the train op:
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    train_op = slim.learning.create_train_op(mse, optimizer)
    # # Run the training:
    final_loss = slim.learning.train(
      train_op,
      logdir=train_dir,
      number_of_steps=1, # For speed, we just do 1 epoch
      save_summaries_secs=1)
  
    # print('Finished training. Final batch loss %d' % final_loss)


