
# import flow_test as ft
import helpers as hpl
import numpy as np
from   PIL import Image
import matplotlib as plt

# import synthetic_tf_converter as converter
import tensorflow as tf
import data_reader as dr
# import matplotlib.mlab as mlab
import ijremote as ij
import losses_helper as lhpl
folder = '../dataset_synthetic/driving/'
# folder = '/misc/lmbraid19/muazzama/dataset_synthetic/driving/'
import synthetic_tf_converter as stc

img1 = folder + 'frames_finalpass_webp/35mm_focallength/scene_backwards/fast/left/0001.webp'
img2 = folder + 'frames_finalpass_webp/35mm_focallength/scene_backwards/fast/left/0002.webp'
disparity1 = folder + 'disparity/35mm_focallength/scene_backwards/fast/left/0001.pfm'
disparity2 = folder + 'disparity/35mm_focallength/scene_backwards/fast/left/0002.pfm'
opt_flow = folder + 'optical_flow/35mm_focallength/scene_backwards/fast/into_future/left/OpticalFlowIntoFuture_0001_L.pfm'
disp_change = folder + 'disparity_change/35mm_focallength/scene_backwards/fast/into_future/left/0001.pfm'

img3 = folder + 'frames_finalpass_webp/35mm_focallength/scene_backwards/fast/left/0107.webp'
opt_flow3 = folder + 'optical_flow/35mm_focallength/scene_backwards/fast/into_future/left/OpticalFlowIntoFuture_0106_L.pfm'
disp_change3 = folder + 'disparity_change/35mm_focallength/scene_backwards/fast/into_future/left/0106.pfm'
disparity3 = folder + 'disparity/35mm_focallength/scene_backwards/fast/left/0107.pfm'

''' ********************************************* this is the reading part ********************************************* '''
''' ********************************************* this is the reading part ********************************************* '''
''' ********************************************* this is the reading part ********************************************* '''
''' ********************************************* this is the reading part ********************************************* '''
''' ********************************************* this is the reading part ********************************************* '''
''' ********************************************* this is the reading part ********************************************* '''
# def show_optical_flow(label_batch): 

# 	factor = 0.4
# 	input_size = int(960 * factor), int(540 * factor)

# 	opt_u = np.squeeze(label_batch[:,:,:,0]) * input_size[0]
# 	opt_v = np.squeeze(label_batch[:,:,:,1]) * input_size[1]

# 	opt_u = opt_u.astype(np.uint8)
# 	opt_v = opt_v.astype(np.uint8)

# 	opt_u = Image.fromarray(opt_u) 
# 	opt_v = Image.fromarray(opt_v)


# 	opt_u.show()
# 	opt_v.show()

# sess = tf.InteractiveSession()
# img = tf.constant([[1,-2,3],[4,5,6],[-7,8,9]],dtype=tf.float32)
# warped = tf.constant([[1,2,-3],[4,0,-6],[-7,0,0]],dtype=tf.float32)

# print(sess.run(tf.abs(lh.get_occulation_aware_image(img,warped))))
# factor = 0.4
# input_size = int(960 * factor), int(540 * factor)

# features_train = dr.tf_record_input_pipelinev2(['one_record.tfrecords'])
# train_imageBatch, train_labelBatch = tf.train.shuffle_batch(
#                                         [features_train['input_n'], 
#                                         features_train['label_n']],
#                                         batch_size=1,
#                                         capacity=100,
#                                         num_threads=10,
#                                         min_after_dequeue=6)

# sess = tf.InteractiveSession()
# # summary_writer_train = tf.summary.FileWriter('./tbtest/')
# coord = tf.train.Coordinator()
# threads = tf.train.start_queue_runners(sess, coord=coord)

# train_batch_xs, train_batch_ys = sess.run([train_imageBatch,train_labelBatch])
# # r1, r2 = np.split(train_batch_xs,2,axis=3)

# train_batch_ys = np.squeeze(train_batch_ys)
# # ij.setImage('loadedImage_u',train_batch_ys[:,:,0] * input_size[0])
# ij.setImage('loadedImage_v',train_batch_ys[:,:,1] * input_size[1])

# # imgg1 = np.squeeze(r1[:,:,:,0:3])
# # imgg2 = np.squeeze(r2[:,:,:,0:3]).astype(np.uint8)

# # flow = np.squeeze(train_batch_ys)
# # # Image.fromarray(flow[:,:,0]).save('n_opt_flow_u_load.tiff')
# # flow = predictor.denormalize_flow(flow,False)
# # Image.fromarray(flow).save('opf_2v.tiff')

# # Image.fromarray(flow[:,:,0]).save('test_flowu.tiff')
# # Image.fromarray(flow[:,:,1]).save('test_flowv.tiff')
# # flow = predictor.warp(imgg1,flow)

# # predictor.show_image(flow.eval()[0].astype(np.uint8),'warped_img')
# coord.request_stop()
# coord.join(threads)



''' ********************************************* from file example ********************************************* '''
''' ********************************************* from file example ********************************************* '''
''' ********************************************* from file example ********************************************* '''
''' ********************************************* from file example ********************************************* '''
''' ********************************************* from file example ********************************************* '''
''' ********************************************* from file example ********************************************* '''

# predictor = ft.FlowPredictor()
# predictor.preprocess(img1,img2,disparity1,disparity2)

# denormu = Image.open('flow_u1.tiff')
# denormv = Image.open('flow_v1.tiff')

# # data = np.dstack((denormu,denormv),dtype=np.float32)

# flow_gt = hpl.readPFM(opt_flow)[0]
# factor = 0.4
# input_size = int(960 * factor), int(540 * factor)

# converter = converter.SyntheticTFRecordsWriter()
# flow_gt = converter.downsample_opt_flow(flow_gt,input_size)
# ij.setImage('InputU',np.array(denormu))
# ij.setImage('InputV',np.array(denormv))

# flow_gt = np.delete(flow_gt,2,axis=2)
# flow_gt_u = flow_gt[:,:,0]
# flow_gt_v = flow_gt[:,:,1]


''' ********************************************* this is the writing part ********************************************* '''
''' ********************************************* this is the writing part ********************************************* '''
''' ********************************************* this is the writing part ********************************************* '''
''' ********************************************* this is the writing part ********************************************* '''
''' ********************************************* this is the writing part ********************************************* '''
''' ********************************************* this is the writing part ********************************************* '''

sess = tf.InteractiveSession()
results = stc.convert_for_testing().from_paths_to_data(disparity1,disparity2,disp_change,opt_flow,img1,img2,'L')


img1 = np.array(Image.open(img1))
img2 = np.array(Image.open(img2))

opt_flow = hpl.readPFM(opt_flow)[0]
opt_flow = opt_flow[:,:,0:2]

# opt_flow = tf.convert_to_tensor(opt_flow,dtype=tf.float32)
# opt_flow = tf.image.resize_images(opt_flow,[224,384])

img2 = tf.expand_dims(tf.convert_to_tensor(results[0]['web_p2'],dtype=tf.float32),axis=0)
# flow = tf.expand_dims(opt_flow,axis=0)
# img2 = tf.expand_dims(tf.convert_to_tensor(img2,dtype=tf.float32),axis=0)
flow = tf.expand_dims(tf.convert_to_tensor(results[0]['optical_flow'],dtype=tf.float32)[:,:,0:2],axis=0)

warped_img = lhpl.flow_warp(img2,flow)
warped_img = sess.run(warped_img)

warped_img = np.squeeze(warped_img)

Image.fromarray(np.uint8(warped_img)).show()


