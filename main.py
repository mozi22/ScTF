import run_network
import data_reader
import multiprocessing
# # read data
# img1 = Image.open(os.path.join(examples_dir,'sculpture1.png'))
# img2 = Image.open(os.path.join(examples_dir,'sculpture2.png'))

# input_data = prepare_input_data(img1,img2,data_format)

# print(get_available_gpus())
prefix = '../dataset_synthetic/'

# memory_folder = '/dev/shm/'

filenames_train  = ['driving_TRAIN.tfrecords','flying_TRAIN.tfrecords','monkaa_TRAIN.tfrecords']
filenames_test  = ['driving_TEST.tfrecords','flying_TEST.tfrecords','monkaa_TEST.tfrecords']

# # with tf.name_scope("datareader"):
train_filenames = [
				prefix+filenames_train[0]
				prefix+filenames_train[1],
				prefix+filenames_train[2]
			]

test_filenames = [
				prefix+filenames_test[0]
				prefix+filenames_test[1],
				prefix+filenames_test[2]
]

# from shutil import copyfile
# copyfile(train_filenames[0], memory_folder + filenames_train[0])


train_features = data_reader.tf_record_input_pipeline(train_filenames,version='1')
test_features = data_reader.tf_record_input_pipeline(test_filenames,version='2')

reader = run_network.DatasetReader()
reader.train(train_features,test_features)
# # reader.main(train_features,test_features)
# # print(helpers.readPFM('0006.pfm'))