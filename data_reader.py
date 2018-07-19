import tensorflow as tf
import numpy as np
import lmbspecialops as sops
from PIL import Image
import ijremote as ij
import os
import csv
import re
def tf_record_input_pipeline(filenames,version='1'):

    # Create a list of filenames and pass it to a queue
    filename_queue = tf.train.string_input_producer(filenames,name='InputProducerV'+version,shuffle=False)

    # Define a reader and read the next record
    recordReader = tf.TFRecordReader(name="TfReaderV"+version)

    key, fullExample = recordReader.read(filename_queue)

    # Decode the record read by the reader
    features = tf.parse_single_example(fullExample, {
        'depth1': tf.FixedLenFeature([], tf.string),
        'depth2': tf.FixedLenFeature([], tf.string),
        'depth_change': tf.FixedLenFeature([], tf.string),
        'opt_flow': tf.FixedLenFeature([], tf.string),
        'image1': tf.FixedLenFeature([], tf.string),
        'image2': tf.FixedLenFeature([], tf.string)
    },
    name="ExampleParserV"+version)

    # Convert the image data from binary back to arrays(Tensors)
    depth1 = tf.decode_raw(features['depth1'], tf.float32)
    depth2 = tf.decode_raw(features['depth2'], tf.float32)
    depth_chng = tf.decode_raw(features['depth_change'], tf.float32)

    image1 = tf.decode_raw(features['image1'], tf.uint8)
    image2 = tf.decode_raw(features['image2'], tf.uint8)
    opt_flow = tf.decode_raw(features['opt_flow'], tf.float32)




    input_pipeline_dimensions = [224, 384]
    image1 = tf.to_float(image1)
    image2 = tf.to_float(image2)


    # reshape data to its original form
    image1 = tf.reshape(image1, [input_pipeline_dimensions[0],input_pipeline_dimensions[1], 3],name="reshape_img1")
    image2 = tf.reshape(image2, [input_pipeline_dimensions[0],input_pipeline_dimensions[1], 3],name="reshape_img2")

    depth1 = tf.reshape(depth1, [input_pipeline_dimensions[0],input_pipeline_dimensions[1]],name="reshape_disp1")
    depth2 = tf.reshape(depth2, [input_pipeline_dimensions[0],input_pipeline_dimensions[1]],name="reshape_disp2")

    optical_flow = tf.reshape(opt_flow, [input_pipeline_dimensions[0],input_pipeline_dimensions[1],2],name="reshape_opt_flow")
    depth_chng = tf.reshape(depth_chng,[input_pipeline_dimensions[0],input_pipeline_dimensions[1]],name="reshape_depth_change")


    image1 = tf.divide(image1,[255])
    image2 = tf.divide(image2,[255])

    # return train_for_opticalflow(image1,image2,optical_flow)
    return train_for_sceneflow(image1,image2,depth1,depth2,depth_chng,optical_flow)

def _parse_function(example_proto):

    features = tf.parse_single_example(example_proto, {
        'depth1': tf.FixedLenFeature([], tf.string),
        'depth2': tf.FixedLenFeature([], tf.string),
        'depth_change': tf.FixedLenFeature([], tf.string),
        'opt_flow': tf.FixedLenFeature([], tf.string),
        'image1': tf.FixedLenFeature([], tf.string),
        'image2': tf.FixedLenFeature([], tf.string),
        'filename1':tf.FixedLenFeature([], tf.string),
        'filename2':tf.FixedLenFeature([], tf.string)
    },
    name="ExampleParserV")
    # Convert the image data from binary back to arrays(Tensors)
    depth1 = tf.decode_raw(features['depth1'], tf.float32)
    depth2 = tf.decode_raw(features['depth2'], tf.float32)
    depth_chng = tf.decode_raw(features['depth_change'], tf.float32)

    image1 = tf.decode_raw(features['image1'], tf.uint8)
    image2 = tf.decode_raw(features['image2'], tf.uint8)
    opt_flow = tf.decode_raw(features['opt_flow'], tf.float32)

    input_pipeline_dimensions = [224, 384]
    image1 = tf.to_float(image1)
    image2 = tf.to_float(image2)

    # reshape data to its original form
    image1 = tf.reshape(image1, [input_pipeline_dimensions[0],input_pipeline_dimensions[1], 3],name="reshape_img1")
    image2 = tf.reshape(image2, [input_pipeline_dimensions[0],input_pipeline_dimensions[1], 3],name="reshape_img2")

    depth1 = tf.reshape(depth1, [input_pipeline_dimensions[0],input_pipeline_dimensions[1]],name="reshape_disp1")
    depth2 = tf.reshape(depth2, [input_pipeline_dimensions[0],input_pipeline_dimensions[1]],name="reshape_disp2")

    optical_flow = tf.reshape(opt_flow, [input_pipeline_dimensions[0],input_pipeline_dimensions[1],2],name="reshape_opt_flow")
    depth_chng = tf.reshape(depth_chng,[input_pipeline_dimensions[0],input_pipeline_dimensions[1]],name="reshape_depth_change")


    image1 = tf.divide(image1,[255])
    image2 = tf.divide(image2,[255])

    final_result = train_for_sceneflow(image1,image2,depth1,depth2,depth_chng,optical_flow)

    return final_result["input_n"], final_result["label_n"], features['filename1'], features['filename2']

def _parse_function_ptb(example_proto):

    features = tf.parse_single_example(example_proto, {
        'depth1': tf.FixedLenFeature([], tf.string),
        'depth2': tf.FixedLenFeature([], tf.string),
        'image1': tf.FixedLenFeature([], tf.string),
        'image2': tf.FixedLenFeature([], tf.string),
        'filename1':tf.FixedLenFeature([], tf.string),
        'filename2':tf.FixedLenFeature([], tf.string)
    },
    name="ExampleParserV")

    # Convert the image data from binary back to arrays(Tensors)
    depth1 = tf.decode_raw(features['depth1'], tf.float32)
    depth2 = tf.decode_raw(features['depth2'], tf.float32)

    image1 = tf.decode_raw(features['image1'], tf.uint8)
    image2 = tf.decode_raw(features['image2'], tf.uint8)

    input_pipeline_dimensions = [224, 384]
    image1 = tf.to_float(image1)
    image2 = tf.to_float(image2)


    # reshape data to its original form
    image1 = tf.reshape(image1, [input_pipeline_dimensions[0],input_pipeline_dimensions[1], 3],name="reshape_img1")
    image2 = tf.reshape(image2, [input_pipeline_dimensions[0],input_pipeline_dimensions[1], 3],name="reshape_img2")

    depth1 = tf.reshape(depth1, [input_pipeline_dimensions[0],input_pipeline_dimensions[1]],name="reshape_disp1")
    depth2 = tf.reshape(depth2, [input_pipeline_dimensions[0],input_pipeline_dimensions[1]],name="reshape_disp2")

    optical_flow1 = tf.zeros_like(depth1)
    optical_flow2 = tf.zeros_like(depth1)
    depth_chng = tf.zeros_like(depth1)

    optical_flow1 = tf.expand_dims(optical_flow1,axis=2)
    optical_flow2 = tf.expand_dims(optical_flow2,axis=2)
    optical_flow = tf.concat([optical_flow1,optical_flow2],axis=2)

    image1 = tf.divide(image1,[255])
    image2 = tf.divide(image2,[255])

    final_result = train_for_sceneflow(image1,image2,depth1,depth2,depth_chng,optical_flow)

    return final_result["input_n"], final_result["label_n"], features['filename1'], features['filename2']



def read_with_dataset_api(batch_size,filenames,version='1'):

    # parallel cpu calls
    num_parallel_calls = 16
    buffer_size = 50

    mapped_data = []

    for name in filenames:
        data = tf.data.TFRecordDataset(name)

        if 'ptb' in name:
            data = data.map(map_func=_parse_function_ptb, num_parallel_calls=num_parallel_calls)
        else:
            data = data.map(map_func=_parse_function, num_parallel_calls=num_parallel_calls)

        mapped_data.append(data)

    data = tuple(mapped_data)

    dataset = tf.data.Dataset.zip(data)

    dataset = dataset.shuffle(buffer_size=50).repeat().apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
    dataset = dataset.prefetch(batch_size)
    # testing_ds_api(dataset)

    return dataset



def read_with_dataset_api_test(batch_size,filenames,version='1'):

    # parallel cpu calls
    num_parallel_calls = 16
    buffer_size = 50

    mapped_data = []

    for name in filenames:
        data = tf.data.TFRecordDataset(name)

        if 'ptb' in name:
            data = data.map(map_func=_parse_function_ptb, num_parallel_calls=num_parallel_calls)
        else:
            data = data.map(map_func=_parse_function, num_parallel_calls=num_parallel_calls)

        mapped_data.append(data)

    data = tuple(mapped_data)

    dataset = tf.data.Dataset.zip(data)

    dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
    dataset = dataset.prefetch(batch_size)
    # testing_ds_api(dataset)

    return dataset



def testing_ds_api(dataset):

    dataset = dataset.make_initializable_iterator()
    sess = tf.InteractiveSession()
    sess.run(dataset.initializer)

    next_batch = dataset.get_next()

    final_img_batch, final_lbl_batch = combine_batches_from_datasets(next_batch)

    next_batch_forward = final_img_batch[:,0,:,:,:]
    next_batch_backward = final_img_batch[:,1,:,:,:]

    forward, backward = sess.run([next_batch_forward,next_batch_backward])
    ij.setHost('tcp://linus:13463')
    ij.setImage('myimage_f',np.transpose(forward,[0,3,1,2]))
    ij.setImage('myimage_b',np.transpose(backward,[0,3,1,2]))


def combine_batches_from_datasets(batches):

    imgs = []
    lbls = []

    # driving

    # batches[x][y] = (4, 2, 224, 384, 8)

    imgs.append(batches[0][0])
    lbls.append(batches[0][1])

    imgs.append(batches[1][0])
    lbls.append(batches[1][1])

    imgs.append(batches[2][0])
    lbls.append(batches[2][1])


    final_img_batch = tf.concat(tuple(imgs),axis=0)
    final_lbl_batch = tf.concat(tuple(lbls),axis=0)

    return final_img_batch, final_lbl_batch


def train_for_opticalflow(image1,image2,optical_flow):

    img_pair_rgb = tf.concat([image1,image2],axis=-1)
    img_pair_rgb_swapped = tf.concat([image2,image1],axis=-1)

    optical_flow_swapped = tf.zeros(optical_flow.get_shape())

    # inputt = divide_inputs_to_patches(img_pair,8)
    # label = divide_inputs_to_patches(label_pair,3)

    # padding_input = tf.constant([[0, 0],[5, 4],[0, 0]])
    x_dimension_padding = tf.constant([[4, 4],[0, 0],[0,0]])
    # padding2 = tf.constant([[4, 4],[0,0]])

    padded_img_pair_rgb = tf.pad(img_pair_rgb,x_dimension_padding,'CONSTANT')
    padded_optical_flow = tf.pad(optical_flow,x_dimension_padding,'CONSTANT')

    padded_img_pair_rgb_swapped = tf.pad(img_pair_rgb_swapped,x_dimension_padding,'CONSTANT')
    padded_optical_flow_swapped = tf.pad(optical_flow_swapped,x_dimension_padding,'CONSTANT')

    fb_rgb_img_pair = tf.stack([padded_img_pair_rgb,padded_img_pair_rgb_swapped])
    fb_rgb_optflow = tf.stack([padded_optical_flow,padded_optical_flow_swapped])

    return {
        'input_n': fb_rgb_img_pair,
        'label_n': fb_rgb_optflow
    }

def decode_pfm(path):
    data = loadpfm.readPFM(path)
    return data[0]

def decode_webp(path):
    data = Image.open(path)
    return np.array(data)


def train_for_sceneflow(image1,image2,depth1,depth2,depth_chng,optical_flow):

    max_depth1 = tf.reduce_max(depth1)
    depth1 = depth1 / max_depth1
    depth2 = depth2 / max_depth1

    depth1 = sops.replace_nonfinite(depth1)
    depth2 = sops.replace_nonfinite(depth2)

    image1 = combine_depth_values(image1,depth1,2)
    image2 = combine_depth_values(image2,depth2,2)

    img_pair_rgbd = tf.concat([image1,image2],axis=-1)
    img_pair_rgbd_swapped = tf.concat([image2,image1],axis=-1)

    # optical_flow = optical_flow / 50
    # comment for optical flow. Uncomment for Sceneflow
    optical_flow_with_depth_change = combine_depth_values(optical_flow,depth_chng,2)
    optical_flow_with_depth_change_swapped = tf.zeros(optical_flow_with_depth_change.get_shape())

    # inputt = divide_inputs_to_patches(img_pair,8)
    # label = divide_inputs_to_patches(label_pair,3)

    # padding_input = tf.constant([[0, 0],[5, 4],[0, 0]])
    # x_dimension_padding = tf.constant([[4, 4],[0, 0],[0,0]])
    # padding2 = tf.constant([[4, 4],[0,0]])
    # padded_img_pair_rgbd = tf.pad(img_pair_rgbd,x_dimension_padding,'CONSTANT')
    # padded_optical_flow_with_depth_change = tf.pad(optical_flow_with_depth_change,x_dimension_padding,'CONSTANT')

    # padded_img_pair_rgbd_swapped = tf.pad(img_pair_rgbd_swapped,x_dimension_padding,'CONSTANT')
    # padded_optical_flow_with_depth_change_swapped = tf.pad(optical_flow_with_depth_change_swapped,x_dimension_padding,'CONSTANT')

    fb_rgbd_img_pair = tf.stack([img_pair_rgbd,img_pair_rgbd_swapped])
    fb_rgbd_optflow_with_depth_change = tf.stack([optical_flow_with_depth_change,optical_flow_with_depth_change_swapped])

    return {
        'input_n': fb_rgbd_img_pair,
        'label_n': fb_rgbd_optflow_with_depth_change
    }



