
import flow_test as ft
import helpers as hpl
import numpy as np
from   PIL import Image
import matplotlib.pyplot as plt
import synthetic_tf_converter as converter
import tensorflow as tf
import data_reader as dr


folder = '/home/muazzam/mywork/python/thesis/server/dataset_synthetic/driving/'
# folder = '/misc/lmbraid19/muazzama/dataset_synthetic/driving/'

img1 = folder + 'frames_finalpass_webp/35mm_focallength/scene_backwards/fast/left/0101.webp'
img2 = folder + 'frames_finalpass_webp/35mm_focallength/scene_backwards/fast/left/0102.webp'
disparity1 = folder + 'disparity/35mm_focallength/scene_backwards/fast/left/0101.pfm'
disparity2 = folder + 'disparity/35mm_focallength/scene_backwards/fast/left/0102.pfm'
opt_flow = folder + 'optical_flow/35mm_focallength/scene_backwards/fast/into_future/left/OpticalFlowIntoFuture_0101_L.pfm'
disp_change = folder + 'disparity_change/35mm_focallength/scene_backwards/fast/into_future/left/0101.pfm'

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


# features_train = dr.tf_record_input_pipelinev2(['one_record.tfrecords'])
# train_imageBatch, train_labelBatch = tf.train.shuffle_batch(
#                                         [features_train['input_n'], 
#                                         features_train['label_n']],
#                                         batch_size=1,
#                                         capacity=100,
#                                         num_threads=10,
#                                         min_after_dequeue=6)

# predictor = ft.FlowPredictor()
# predictor.preprocess(img1,img2,disparity1,disparity2)

# sess = tf.InteractiveSession()
# summary_writer_train = tf.summary.FileWriter('./tbtest/')
# coord = tf.train.Coordinator()
# threads = tf.train.start_queue_runners(sess, coord=coord)

# train_batch_xs, train_batch_ys = sess.run([train_imageBatch,train_labelBatch])
# show_optical_flow(train_batch_ys)
# r1, r2 = np.split(train_batch_xs,2,axis=3)


# imgg1 = np.squeeze(r1[:,:,:,0:3])
# imgg2 = np.squeeze(r2[:,:,:,0:3]).astype(np.uint8)

# flow = np.squeeze(train_batch_ys)
# flow = predictor.denormalize_flow(flow,False)


# flow = predictor.warp(imgg1,flow)

# predictor.show_image(flow.eval()[0].astype(np.uint8),'warped_img')
# coord.request_stop()
# coord.join(threads)


''' ********************************************* this is the writing part ********************************************* '''
''' ********************************************* this is the writing part ********************************************* '''
''' ********************************************* this is the writing part ********************************************* '''
''' ********************************************* this is the writing part ********************************************* '''
''' ********************************************* this is the writing part ********************************************* '''
''' ********************************************* this is the writing part ********************************************* '''

# def create_tf_example(self,patches,writer):

# 	for item in patches:

# 		# downsampled_opt_flow = self.downsample_labels(np.array(item['opt_fl']),2)
# 		# downsampled_disp_chng = self.downsample_labels(np.array(item['disp_chng']),0)

# 		width , height = item['depth'].shape[0] , item['depth'].shape[1]
# 		depth = item['depth'].tostring()
# 		depth2 = item['depth2'].tostring()

# 		opt_flow = item['optical_flow'].tostring()
# 		depth_chng = item['disp_change'].tostring()
# 		frames_finalpass_webp = item['web_p'].tostring()
# 		frames_finalpass_webp2 = item['web_p2'].tostring()




# 		example = tf.train.Example(features=tf.train.Features(
# 			feature={
# 				'width': self._int64_feature(width),
# 				'height': self._int64_feature(height),
# 				'depth1': self._bytes_feature(depth),
# 				'depth2': self._bytes_feature(depth2),
# 				'disp_chng': self._bytes_feature(depth_chng),
# 				'opt_flow': self._bytes_feature(opt_flow),
# 				'image1': self._bytes_feature(frames_finalpass_webp),
# 				'image2': self._bytes_feature(frames_finalpass_webp2)
# 		    }),
# 		)

# 		writer.write(example.SerializeToString())
# 		writer.close()




# from here

# opt_flow2 = hpl.readPFM(opt_flow)[0]
# Image.fromarray(opt_flow2[:,:,0]).show()
# Image.fromarray(opt_flow2[:,:,1]).show()

predictor = ft.FlowPredictor()
predictor.preprocess(img1,img2,disparity1,disparity2)
predictor.predict()



# for testing with ground truth


# converter = converter.SyntheticTFRecordsWriter()
# result = converter.from_paths_to_data(disparity1,
# 									  disparity2,
# 									  disp_change,
# 									  opt_flow,
# 									  img1,
# 									  img2,
# 									  1)
# train_writer = tf.python_io.TFRecordWriter('./one_record.tfrecords')
# create_tf_example(converter,result,train_writer)



# lbl = predictor.read_gt(opt_flow,disp_change)
# opt_flow = np.pad(lbl,((4,4),(0,0),(0,0)),'constant')
# predictor.postprocess(flow=opt_flow,show_flow=True,gt=True)

# for testing

# opt = hpl.readPFM(opt_flow)[0]
# Image.fromarray(opt.astype(np.uint8)).show()






# dispar1 = hpl.readPFM(disparity1)[0]
# dispar2 = hpl.readPFM(disparity2)[0]
# opt_flow = hpl.readPFM(opt_flow)[0]



# dispar_chng = hpl.readPFM(disp_change)[0]
# result1 = predictor.get_depth_from_disp(dispar1)
# result2 = predictor.get_depth_from_disp(dispar2)
# result3 = predictor.get_depth_from_disp(dispar_chng)
# result3 = predictor.get_depth_chng_from_disp_chng(dispar1,dispar_chng)
# Image.open(img1).show()
# Image.open(img2).show()
# Image.fromarray(result1).show()
# Image.fromarray(result2).show()
# Image.fromarray(result3).show()
# Image.fromarray(opt_flow[:,:,0]).show()
# Image.fromarray(opt_flow[:,:,1]).show()
# print(opt_flow[:,:,1])

# plt.hist(opt_flow[:,:,1], bins='auto')  # arguments are passed to np.histogram
# plt.title("Histogram with 'auto' bins")
# plt.show()