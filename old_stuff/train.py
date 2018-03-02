import network
import numpy as np
from   PIL import Image
import tensorflow as tf
import helpers as hpl


def main(features_train, features_test):
        self.batch_size = 64
        self.total_iterations = 50000
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
