import tensorflow.contrib.slim as slim
import tensorflow as tf

with tf.Session() as sess:    
    saver = tf.train.import_meta_graph('muazzam/my_test_model-1000.meta')
    saver.restore(sess,tf.train.latest_checkpoint('./muazzam'))
    print(sess.run('w1:0'))