import tensorflow as tf

import run_network
import helpers

# # read data
# img1 = Image.open(os.path.join(examples_dir,'sculpture1.png'))
# img2 = Image.open(os.path.join(examples_dir,'sculpture2.png'))

# input_data = prepare_input_data(img1,img2,data_format)

# print(get_available_gpus())
prefix = '../../danda/'

# with tf.name_scope("datareader"):
filenames = [prefix+'driving_TRAIN.tfrecords']
reader = run_network.DatasetReader()
reader.iterate(filenames)
# print(helpers.readPFM('0006.pfm'))