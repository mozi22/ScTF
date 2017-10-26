#
#  DeMoN - Depth Motion Network
#  Copyright (C) 2017  Benjamin Ummenhofer, Huizhong Zhou
#  
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
from .blocks_original import *



class BootstrapNet:
    def __init__(self, session, data_format='channels_first'):
        """Creates the network

        session: tf.Session
            Tensorflow session

        data_format: str
            Either 'channels_first' or 'channels_last'.
            Running on the cpu requires 'channels_last'.
        """
        self.session = session
        if data_format=='channels_first':
            self.placeholder_image_pair = tf.placeholder(dtype=tf.float32, shape=(1,6,192,256))
            self.placeholder_image2_2 = tf.placeholder(dtype=tf.float32, shape=(1,3,48,64))
        else:
            self.placeholder_image_pair = tf.placeholder(dtype=tf.float32, shape=(1,192,256,6))
            self.placeholder_image2_2 = tf.placeholder(dtype=tf.float32, shape=(1,48,64,3))

        with tf.variable_scope('netFlow1'):
            netFlow1_result = flow_block_demon_original(self.placeholder_image_pair, data_format=data_format )
            self.netFlow1_result = netFlow1_result
            self.predict_flow5, self.predict_conf5 = tf.split(value=netFlow1_result['predict_flowconf5'], num_or_size_splits=2, axis=1 if data_format=='channels_first' else 3)
            self.predict_flow2, self.predict_conf2 = tf.split(value=netFlow1_result['predict_flowconf2'], num_or_size_splits=2, axis=1 if data_format=='channels_first' else 3)

            self.session.run(self.predict_flow5)


    def eval(self, image_pair, image2_2):
        """Runs the bootstrap network
        
        image_pair: numpy.ndarray
            Array with shape [1,6,192,256] if data_format=='channels_first'
            
            Image pair in the range [-0.5, 0.5]

        image2_2: numpy.ndarray
            Second image at resolution level 2 (downsampled two times)

            The shape for data_format=='channels_first' is [1,3,48,64]

        Returns a dict with the preditions of the bootstrap net
        """
        
        fetches = {
                'predict_flow5': self.predict_flow5,
                'predict_flow2': self.predict_flow2,
                'predict_depth2': self.netDM1_result['predict_depth2'],
                'predict_normal2': self.netDM1_result['predict_normal2'],
                'predict_rotation': self.netDM1_result['predict_rotation'],
                'predict_translation': self.netDM1_result['predict_translation'],
                }
        feed_dict = {
                self.placeholder_image_pair: image_pair,
                self.placeholder_image2_2: image2_2,
                }
        return self.session.run(fetches, feed_dict=feed_dict)

