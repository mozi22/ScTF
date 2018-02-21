import tensorflow as tf

import run_network
import helpers
import manual_tester
import data_reader

# # read data
# img1 = Image.open(os.path.join(examples_dir,'sculpture1.png'))
# img2 = Image.open(os.path.join(examples_dir,'sculpture2.png'))

# input_data = prepare_input_data(img1,img2,data_format)

# print(get_available_gpus())
prefix = '../../dataset_synthetic/tfrecords2/'

# with tf.name_scope("datareader"):
train_filenames = [
				prefix+'driving_TRAIN.tfrecords'
				# prefix+'flyingthings3d_TRAIN.tfrecords',
				# prefix+'monkaa_TRAIN.tfrecords'
			]

test_filenames = [
				prefix+'driving_TEST.tfrecords'
				# prefix+'flyingthings3d_TEST.tfrecords',
				# prefix+'monkaa_TEST.tfrecords'
]

train_features = data_reader.tf_record_input_pipeline(train_filenames,version='1')
test_features = data_reader.tf_record_input_pipeline(test_filenames,version='2')




reader = run_network.DatasetReader()
reader.main(train_features,test_features)
# print(helpers.readPFM('0006.pfm'))