import tensorflow as tf
import lmbspecialops as sops
import os
import csv
import re
import numpy as np
from PIL import Image
import network
import losses_helper

def write_flyingthings3d_dataset_csv():


    root = '../dataset_synthetic/flyingthings3d/'
    disp_folder = 'disparity'
    disp_chng_folder = 'disparity_change'
    optical_flow_folder = 'optical_flow'
    images_folder = 'frames_finalpass_webp'

    modules = ['/TRAIN/C/']


    with open('flyingA.csv', 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for module_name in modules:
            for filename in os.listdir(root + disp_folder + module_name):

                disparity_path = root + disp_folder + module_name + filename + '/left/'
                disparity_chng_path = root + disp_chng_folder + module_name + filename + '/into_future/left/'
                optical_flow_path = root + optical_flow_folder + module_name + filename + '/into_future/left/'
                images_path = root + images_folder + module_name + filename + '/left/'


                for file_id in range(6,15):

                    optical_flow_path_final = optical_flow_path + 'OpticalFlowIntoFuture_'+str("%04d" % (file_id,))+'_L.pfm'
                    disparity_chng_path_final = disparity_chng_path + str("%04d" % (file_id,)) + '.pfm'

                    disparity1_path_final =  disparity_path + str("%04d" % (file_id,)) + '.pfm'
                    disparity2_path_final =  disparity_path + str("%04d" % (file_id+1,)) + '.pfm'
                    images1_path_final = images_path + str("%04d" % (file_id,)) + '.webp'
                    images2_path_final = images_path + str("%04d" % (file_id+1,)) + '.webp'


                    spamwriter.writerow([images1_path_final, 
                                         images2_path_final, 
                                         disparity1_path_final,
                                         disparity2_path_final,
                                         optical_flow_path_final,
                                         disparity_chng_path_final])



def write_driving_dataset_csv():


    root = '../dataset_synthetic/flyingthings3d/'
    disp_folder = 'disparity'
    disp_chng_folder = 'disparity_change'
    optical_flow_folder = 'optical_flow'
    images_folder = 'frames_finalpass_webp'


    folder_into_future = '/35mm_focallength/scene_forwards/fast/'



    with open('driving.csv', 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for module_name in modules:
            for filename in os.listdir(root + disp_folder + module_name):

                disparity_path = root + disp_folder + folder_into_future + 'left/'
                disparity_chng_path = root + disp_chng_folder  + folder_into_future + '/into_future/left/'
                optical_flow_path = root + optical_flow_folder + folder_into_future + '/into_future/left/'
                images_path = root + images_folder + folder_into_future + '/left/'


                for file_id in range(6,15):

                    optical_flow_path_final = optical_flow_path + 'OpticalFlowIntoFuture_'+str("%04d" % (file_id,))+'_L.pfm'
                    disparity_chng_path_final = disparity_chng_path + str("%04d" % (file_id,)) + '.pfm'

                    disparity1_path_final =  disparity_path + str("%04d" % (file_id,)) + '.pfm'
                    disparity2_path_final =  disparity_path + str("%04d" % (file_id+1,)) + '.pfm'
                    images1_path_final = images_path + str("%04d" % (file_id,)) + '.webp'
                    images2_path_final = images_path + str("%04d" % (file_id+1,)) + '.webp'


                    spamwriter.writerow([images1_path_final, 
                                         images2_path_final, 
                                         disparity1_path_final,
                                         disparity2_path_final,
                                         optical_flow_path_final,
                                         disparity_chng_path_final])



def readPFM(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().decode('utf-8').rstrip())
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale




def preprocessing(x):
    """
    # Create Normal distribution
    x = x.astype('float32')
    x[:, :, :, 0] = (x[:, :, :, 0] - np.mean(x[:, :, :, 0])) / np.std(x[:, :, :, 0])
    x[:, :, :, 1] = (x[:, :, :, 1] - np.mean(x[:, :, :, 1])) / np.std(x[:, :, :, 1])
    x[:, :, :, 2] = (x[:, :, :, 2] - np.mean(x[:, :, :, 2])) / np.std(x[:, :, :, 2])
    """
    x = x/127.5 - 1 # -1 ~ 1
    return x

def get_depth_from_disp(disparity):
    focal_length = 1050.0
    disp_to_depth = disparity / focal_length
    return disp_to_depth


def normalizeOptFlow(flow,input_size):

    # separate the u and v values 
    flow_u = flow[:,:,0]
    flow_v = flow[:,:,1]
    flow_w = flow[:,:,2]


    # normalize the values by the image dimensions
    flow_u = flow_u / input_size[0]
    flow_v = flow_v / input_size[1]

    # combine them back and return
    return np.dstack((flow_u,flow_v,flow_w))


def parse_files(files):


    input_size = 960, 540

    img1,img2,disp1,disp2,opt_flow,disp_chng = files[0][0],files[0][1],files[0][2],files[0][3],files[0][4],files[0][5]

    # read from paths to arrays
    disp1 = readPFM(disp1)[0]
    disp2 = readPFM(disp2)[0]
    disp_chng = readPFM(disp_chng)[0]
    opt_flow = readPFM(opt_flow)[0]
    img1 = Image.open(img1)
    img2 = Image.open(img2)

    # change img to array
    img1_arr = np.array(img1)
    img2_arr = np.array(img2)


    # set values between -1 - 1
    img1_st = img1_arr / 255
    img2_st = img2_arr / 255

    max_disp = np.max(disp1)
    disp1 = disp1 / max_disp
    disp2 = disp2 / max_disp
    disp_chng = disp_chng / max_disp


    disp1_st = get_depth_from_disp(disp1)
    disp2_st = get_depth_from_disp(disp2)
    disp_chng_st = get_depth_from_disp(disp_chng)

   # opt_flow_st = preprocessing(opt_flow)

    opt_flow_st = normalizeOptFlow(opt_flow,input_size)


    return img1_st.astype(np.float32), \
            img2_st.astype(np.float32), \
            disp1_st.astype(np.float32), \
            disp2_st.astype(np.float32), \
            disp_chng_st.astype(np.float32), \
            opt_flow_st.astype(np.float32)


def endpoint_loss(gt_flow,predicted_flow,weight=1,scope='epe_loss',stop_grad=False,summary_type='_train'):

  with tf.variable_scope(scope):

    # if stop_grad == False:
    #   gt_flow = tf.stop_gradient(gt_flow)


    
    # get u & v value for gt
    gt_u = tf.slice(gt_flow,[0,0,0,0],[-1,-1,-1,1])
    gt_v = tf.slice(gt_flow,[0,0,0,1],[-1,-1,-1,1])
    # gt_w = tf.slice(gt_flow,[0,0,0,2],[-1,-1,-1,1])

    # get u & v value for predicted_flow
    pred_u = tf.slice(predicted_flow,[0,0,0,0],[-1,-1,-1,1])
    pred_v = tf.slice(predicted_flow,[0,0,0,1],[-1,-1,-1,1])
    # pred_w = tf.slice(predicted_flow,[0,0,0,2],[-1,-1,-1,1])


    diff_u = sops.replace_nonfinite(gt_u - pred_u)
    diff_v = sops.replace_nonfinite(gt_v - pred_v)
    # diff_w = sops.replace_nonfinite(gt_w - pred_w)

    epe_loss = tf.sqrt((diff_u**2) + (diff_v**2) + 1e-6)

    epe_loss = tf.reduce_mean(sops.replace_nonfinite(epe_loss))

    # epe_loss = tf.check_numerics(epe_loss,'numeric checker')
    # epe_loss = tf.Print(epe_loss,[epe_loss],'epeloss ye hai ')

    tf.losses.compute_weighted_loss(epe_loss,weights=weight)
  
  return epe_loss

def read_flying_ds():

    filename_queue = tf.train.string_input_producer(["flyingA.csv"])

    reader = tf.TextLineReader()
    key, value = reader.read(filename_queue)

    # Default values, in case of empty columns. Also specifies the type of the
    # decoded result.
    record_defaults = [[''], [''], [''], [''], [''], ['']]
    col1, col2, col3, col4, col5, col6 = tf.decode_csv(
        value, record_defaults=record_defaults)
    features = tf.stack([col1, col2, col3, col4, col5, col6])

    shuffled_data = tf.train.shuffle_batch(
      [features], batch_size=1, capacity=500,
      num_threads=32,
      min_after_dequeue=200)


    a1,a2,a3,a4,a5,a6 = tf.py_func(parse_files, [shuffled_data], [tf.float32,tf.float32,tf.float32,tf.float32,tf.float32,tf.float32])


    # integrate all images to (12,540,960) single image.
    a1 = tf.transpose(a1,[2,0,1])
    a2 = tf.transpose(a2,[2,0,1])
    a6 = tf.transpose(a6,[2,0,1])
    combined_depths_as_channels = tf.concat([tf.expand_dims(a3,axis=0),
                                             tf.expand_dims(a4,axis=0)
                                             # tf.expand_dims(a5,axis=0)
                                             ],axis=0)


    combined_depths_as_channels = tf.concat([a1,a2,a6,combined_depths_as_channels],axis=0)

    # crop the same place from all the 12 portions.
    combined_depths_as_channels = tf.random_crop(combined_depths_as_channels,[11,256,256])


    # disintegrate all information to it's original form, in [C,H,W] tensor
    img1 = combined_depths_as_channels[0:3,:,:]
    img2 = combined_depths_as_channels[3:6,:,:]
    opt_flow = combined_depths_as_channels[6:8,:,:] # note here we are only taking 2 channels of optical flow. Third one is 0 anyways.
    depth1 = tf.expand_dims(combined_depths_as_channels[9,:,:],axis=0)
    depth2 = tf.expand_dims(combined_depths_as_channels[10,:,:],axis=0)
    # depth_chng = tf.expand_dims(combined_depths_as_channels[11,:,:],axis=0)


    img_pair_depth = tf.concat([img1,depth1,img2,depth2],axis=0)
    # optical_flow = tf.concat([opt_flow,depth_chng],axis=0)
    optical_flow = opt_flow

    img_pair_depth = tf.transpose(img_pair_depth,[1,2,0])
    gt_flow = tf.transpose(optical_flow,[1,2,0])

    img_pair_depth = tf.expand_dims(img_pair_depth,axis=0)
    gt_flow = tf.expand_dims(gt_flow,axis=0)

    predict_flows = network.train_network(img_pair_depth)[0]

    epe_loss = endpoint_loss(gt_flow,predict_flows)


    MAX_STEPS = 50000

    global_step = tf.get_variable(
        'global_step', [],
        initializer=tf.constant_initializer(0), trainable=False)

    learning_rate = tf.train.polynomial_decay(0.0001, global_step,
                                              MAX_STEPS, 0.000001,
                                              power=3)

    opt = tf.train.AdamOptimizer(learning_rate).minimize(epe_loss)



    summaries = []
    summaries.append(tf.summary.scalar('epe_loss', epe_loss))
    summaries.append(tf.summary.image('predicted_flow_u',tf.expand_dims(predict_flows[:,:,:,0],axis=-1)))
    summaries.append(tf.summary.image('predicted_flow_v',tf.expand_dims(predict_flows[:,:,:,1],axis=-1)))
    # summaries.append(tf.summary.image('predicted_flow_w',tf.expand_dims(predict_flows[:,:,:,2],axis=-1)))
    summaries.append(tf.summary.image('gt_flow_u',tf.expand_dims(gt_flow[:,:,:,0],axis=-1)))
    summaries.append(tf.summary.image('gt_flow_v',tf.expand_dims(gt_flow[:,:,:,1],axis=-1)))
    # summaries.append(tf.summary.image('gt_flow_w',tf.expand_dims(gt_flow[:,:,:,2],axis=-1)))
    summaries.append(tf.summary.image('inp_img1',img_pair_depth[:,:,:,0:3]))
    summaries.append(tf.summary.image('inp_img2',img_pair_depth[:,:,:,4:7]))
    summaries.append(tf.summary.image('inp_depth1',tf.expand_dims(img_pair_depth[:,:,:,3],axis=-1)))
    summaries.append(tf.summary.image('inp_depth2',tf.expand_dims(img_pair_depth[:,:,:,7],axis=-1)))
    # summaries.append(tf.summary.image('depth_change',tf.expand_dims(depth_chng,axis=-1)))
    summary_op = tf.summary.merge(summaries)



    saver = tf.train.Saver(tf.global_variables())


    save_directory = 'ckptp/'

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter(save_directory, sess.graph)

        for i in range(0,MAX_STEPS):

            _, loss = sess.run([opt,epe_loss])


            print('Iteration: ' + str(i) + ' ---- Epe loss : '+ str(loss))

            if i % 100 == 0:
                summmary = sess.run(summary_op)
                summary_writer.add_summary(summmary,i)

            # Save the model checkpoint periodically.
            if i % 1000 == 0:
                checkpoint_path = os.path.join(save_directory, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=i)

        coord.request_stop()
        coord.join(threads)


read_flying_ds()


