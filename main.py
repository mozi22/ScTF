import tensorflow as tf

import dataset_reader


# # read data
# img1 = Image.open(os.path.join(examples_dir,'sculpture1.png'))
# img2 = Image.open(os.path.join(examples_dir,'sculpture2.png'))

# input_data = prepare_input_data(img1,img2,data_format)

# print(get_available_gpus())
        



# with tf.name_scope("datareader"):
filenames = ['optical_flow.tfrecords']
reader = dataset_reader.DatasetReader()
reader.iterate(filenames)
