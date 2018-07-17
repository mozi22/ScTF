import data_reader
import losses_helper
import tensorflow as tf
import network
import numpy as np
import ijremote as ij
from math import pi
ij.setHost('tcp://linus:13463')

def get_network_input_forward(image_batch,label_batch):
    return image_batch[:,0,:,:,:], label_batch[:,0,:,:,:]
 
ds = 'flying'
filee = ['../dataset_synthetic/'+ds+'_TEST.tfrecords']



test_dataset = data_reader.read_with_dataset_api_test(1,filee,version='1')
test_iterator = test_dataset.make_one_shot_iterator()

def create_flow_color_legend(size=64):
    """Creates the flow color code legend

    size: int
        Number of pixels for width and height

    Returns the rgb tensor with the image
    """
    X, Y = np.meshgrid(np.linspace(-1,1,64),np.linspace(-1,1,64))
    flow = np.concatenate((X[np.newaxis,:,:],Y[np.newaxis,:,:]))
    flow_tensor = tf.constant(flow[np.newaxis,:,:,:], dtype=tf.float32)
    return create_flow_color_image(flow_tensor)[0]



def create_flow_color_image(flow, denormalize=False, max_length=None, name=None):
    """Creates a color coded optical flow image with the magnitude as saturation
    and the hue as direction.
    White pixels indicate nonfinite values in the x or y component.
    
    flow: Tensor 
        Tensor in NCHW format with C=2. The first channel is the x component
    
    denormalize: bool
        If True denormalized the optical flow with the image dimensions
        
    max_length: Tensor or None
        This value is used to normalize the flow vectors.
        If None each batch item will be normalized individually by the largest
        flow vector.
        
    Returns a flow rgb image in NHWC format with C=3 and the vector of values used for normalization
    """
    with tf.name_scope(name, 'create_flow_color_image', [flow]) as scope:
        # flow = tf.convert_to_tensor(flow, name='flow')
        shape = flow.get_shape().as_list()
        assert len(shape)==4
        assert shape[1]==2
        if not max_length is None:
            max_length = tf.convert_to_tensor(max_length, name='max_length',dtype=flow.dtype)
            max_length.get_shape().with_rank(1)
            max_length_shape = max_length.get_shape().as_list()
            assert shape[0] == max_length_shape[0] or 1 == max_length_shape[0]
            if max_length_shape[0] == 1:
                max_length = tf.tile(max_length,[shape[0]])

        width = shape[-1]
        height = shape[-2]

        flow_nhwc = tf.transpose(flow,[0,2,3,1])

        fx, fy = tf.split(flow_nhwc, num_or_size_splits=2, axis=-1)
        if denormalize:
            fx *= width
            fy *= height

        veclength = tf.sqrt(fx**2+fy**2)
            
        mask = tf.is_finite(veclength)
        veclength_clean = tf.where(mask, veclength, tf.zeros_like(veclength))
        if max_length is None:
            max_veclength = tf.reduce_max(veclength_clean, axis=[1,2,3],keep_dims=True)
        else:
            max_veclength = tf.reshape(max_length, [shape[0],1,1,1])
            
        veclength_clean_norm = veclength_clean/max_veclength
        angle = tf.atan2(fy, fx)

        print(angle,mask, veclength_clean_norm,max_veclength)

        hue = tf.where(mask, (angle+pi)/(2*pi), tf.zeros_like(veclength))
        sat = tf.where(mask, tf.ones_like(veclength), tf.zeros_like(veclength))
        val = tf.where(mask, veclength_clean_norm, tf.ones_like(veclength))
        hsv = tf.clip_by_value(tf.concat([hue,sat,val], axis=-1), 0, 1)
        rgb = tf.image.hsv_to_rgb(hsv)

    return rgb
def load_model_ckpt(sess,filename):
	saver = tf.train.Saver()
	saver.restore(sess, tf.train.latest_checkpoint(filename))

def further_resize_imgs_lbls(network_input_images,network_input_labels):

    network_input_images = tf.image.resize_images(network_input_images,[160,256],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    network_input_labels = tf.image.resize_images(network_input_labels,[160,256],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    network_input_labels_u = network_input_labels[:,:,:,0] * 0.714285714
    network_input_labels_v = network_input_labels[:,:,:,1] * 0.666666667
 
    network_input_labels_u = tf.expand_dims(network_input_labels_u,axis=-1)
    network_input_labels_v = tf.expand_dims(network_input_labels_v,axis=-1)
 
    network_input_labels = tf.concat([network_input_labels_u,network_input_labels_v],axis=3)

    return network_input_images, network_input_labels

def get_predict_flow_forward_backward(predict_flows):


    predict_flow = predict_flows[0]
    predict_flow_ref1 = predict_flows[1]
    predict_flow_ref2 = predict_flows[2]
    predict_flow_ref3 = predict_flows[3]

    # for other losses, we only consider forward flow
    predict_flow_forward = tf.expand_dims(predict_flow[0,:,:,:],axis=0)
    predict_flow_backward = tf.expand_dims(predict_flow[1,:,:,:],axis=0)

    predict_flow_forward_ref3 = tf.expand_dims(predict_flow_ref3[0,:,:,:],axis=0)
    predict_flow_backward_ref3 = tf.expand_dims(predict_flow_ref3[1,:,:,:],axis=0)

    predict_flow_forward_ref2 = tf.expand_dims(predict_flow_ref2[0,:,:,:],axis=0)
    predict_flow_backward_ref2 = tf.expand_dims(predict_flow_ref2[1,:,:,:],axis=0)

    predict_flow_forward_ref1 = tf.expand_dims(predict_flow_ref1[0,:,:,:],axis=0)
    predict_flow_backward_ref1 = tf.expand_dims(predict_flow_ref1[1,:,:,:],axis=0)

    if ds == 'mid':
        return {
            'predict_flow': [-predict_flow_forward, -predict_flow_backward],
            'predict_flow_ref3': [-predict_flow_forward_ref3,-predict_flow_backward_ref3],
            'predict_flow_ref2': [-predict_flow_forward_ref2, -predict_flow_backward_ref2],
            'predict_flow_ref1': [-predict_flow_forward_ref1, -predict_flow_backward_ref1]
        }
    else:
        print('without mid')
        return {
            'predict_flow': [predict_flow_forward, predict_flow_backward],
            'predict_flow_ref3': [predict_flow_forward_ref3,predict_flow_backward_ref3],
            'predict_flow_ref2': [predict_flow_forward_ref2, predict_flow_backward_ref2],
            'predict_flow_ref1': [predict_flow_forward_ref1, predict_flow_backward_ref1]
        }

X = tf.placeholder(dtype=tf.float32, shape=(1, 2, 224, 384, 8))
Y = tf.placeholder(dtype=tf.float32, shape=(1, 2, 224, 384, 3))

X_forward, Y_forward = X[:,0,:,:,:], Y[:,0,:,:,:]
X_backward, Y_backward = X[:,1,:,:,:], Y[:,1,:,:,:]

X_forward, Y_forward = further_resize_imgs_lbls(X_forward, Y_forward)
X_backward, Y_backward = further_resize_imgs_lbls(X_backward, Y_backward)

concatenated_FB_images = tf.concat([X_forward,X_backward],axis=0)


predict_flows = network.train_network(concatenated_FB_images)
predict_flows2 = network.train_network(predict_flows[0],'s_evolution2','s_evolution2')
flows_dict = get_predict_flow_forward_backward(predict_flows2)

################ epe loss #######################
denormalized_flow = losses_helper.denormalize_flow(flows_dict['predict_flow'][0])


# total_loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(Y_forward, denormalized_flow))))
total_loss = losses_helper.endpoint_loss(Y_forward,flows_dict['predict_flow'][0],scope='epe_loss_evolution1')
# total_loss = losses_helper.forward_backward_loss(flows_dict['predict_flow'][0],flows_dict['predict_flow'][1],scope='epe_loss_evolution1')


# sess.run(test_iterator.initializer)

test_image_batch, test_label_batch, filename1, filename2 = test_iterator.get_next()[0]

summaies = []

img_forward = tf.concat([X_forward[:,:,:,0:3],X_forward[:,:,:,4:7]],axis=2)
img_backward = tf.concat([X_backward[:,:,:,0:3],X_backward[:,:,:,4:7]],axis=2)

depth_forward = tf.concat([tf.expand_dims(X_forward[:,:,:,3],axis=-1),tf.expand_dims(X_forward[:,:,:,7],axis=-1)],axis=2)
depth_backward = tf.concat([tf.expand_dims(X_backward[:,:,:,3],axis=-1),tf.expand_dims(X_backward[:,:,:,7],axis=-1)],axis=2)

network_input_labels_refine3 = losses_helper.downsample_label(Y_forward,
                                size=[20,32],factorU=0.125,factorV=0.125)
network_input_labels_refine2 = losses_helper.downsample_label(Y_forward,
                                size=[40,64],factorU=0.25,factorV=0.25)
network_input_labels_refine1 = losses_helper.downsample_label(Y_forward,
                                size=[80,128],factorU=0.5,factorV=0.5)

img_forward_predict_u = tf.concat([tf.expand_dims(Y_forward[:,:,:,0],axis=-1),tf.expand_dims(flows_dict['predict_flow'][0][:,:,:,0],axis=-1)],axis=2)
img_forward_predict_v = tf.concat([tf.expand_dims(Y_forward[:,:,:,1],axis=-1),tf.expand_dims(flows_dict['predict_flow'][0][:,:,:,1],axis=-1)],axis=2)

img_forward_predict_u_ref3 = tf.concat([tf.expand_dims(network_input_labels_refine3[:,:,:,0],axis=-1),tf.expand_dims(flows_dict['predict_flow_ref3'][0][:,:,:,0],axis=-1)],axis=2)
img_forward_predict_v_ref3 = tf.concat([tf.expand_dims(network_input_labels_refine3[:,:,:,1],axis=-1),tf.expand_dims(flows_dict['predict_flow_ref3'][0][:,:,:,1],axis=-1)],axis=2)

img_forward_predict_u_ref2 = tf.concat([tf.expand_dims(network_input_labels_refine2[:,:,:,0],axis=-1),tf.expand_dims(flows_dict['predict_flow_ref2'][0][:,:,:,0],axis=-1)],axis=2)
img_forward_predict_v_ref2 = tf.concat([tf.expand_dims(network_input_labels_refine2[:,:,:,1],axis=-1),tf.expand_dims(flows_dict['predict_flow_ref2'][0][:,:,:,1],axis=-1)],axis=2)

img_forward_predict_u_ref1 = tf.concat([tf.expand_dims(network_input_labels_refine1[:,:,:,0],axis=-1),tf.expand_dims(flows_dict['predict_flow_ref1'][0][:,:,:,0],axis=-1)],axis=2)
img_forward_predict_v_ref1 = tf.concat([tf.expand_dims(network_input_labels_refine1[:,:,:,1],axis=-1),tf.expand_dims(flows_dict['predict_flow_ref1'][0][:,:,:,1],axis=-1)],axis=2)


img_fb_predict = tf.concat([tf.expand_dims(flows_dict['predict_flow'][0][:,:,:,0],axis=-1),tf.expand_dims(flows_dict['predict_flow'][1][:,:,:,1],axis=-1)],axis=2)


warped_img = losses_helper.flow_warp(X_forward[:,:,:,4:7],denormalized_flow)

flow_color_legend = create_flow_color_legend()
flow_color_image = create_flow_color_image(tf.transpose(denormalized_flow,[0,3,1,2]))

summaies.append(tf.summary.image('warped_img',tf.concat([X_forward[:,:,:,0:3],warped_img],axis=2)))
summaies.append(tf.summary.image('image_forward',img_forward))
summaies.append(tf.summary.image('flow_color_legend',tf.expand_dims(flow_color_legend,axis=0)))
summaies.append(tf.summary.image('flow_color_image',flow_color_image))
summaies.append(tf.summary.image('image_backward',img_backward))
summaies.append(tf.summary.image('depth_forward',depth_forward))
summaies.append(tf.summary.image('depth_backward',depth_backward))
summaies.append(tf.summary.scalar('rmse_loss',total_loss))

summaies.append(tf.summary.image('lbl_pred_final_flow_u',img_forward_predict_u))
summaies.append(tf.summary.image('lbl_pred_final_flow_v',img_forward_predict_v))
summaies.append(tf.summary.image('lbl_pred_ref3_flow_u',img_forward_predict_u_ref3))
summaies.append(tf.summary.image('lbl_pred_ref3_flow_v',img_forward_predict_v_ref3))
summaies.append(tf.summary.image('lbl_pred_ref2_flow_u',img_forward_predict_u_ref2))
summaies.append(tf.summary.image('lbl_pred_ref2_flow_v',img_forward_predict_v_ref2))
summaies.append(tf.summary.image('lbl_pred_ref1_flow_u',img_forward_predict_u_ref1))
summaies.append(tf.summary.image('lbl_pred_ref1_flow_v',img_forward_predict_v_ref1))
summaies.append(tf.summary.image('predfb_final_flow_u',img_fb_predict))

summary_op = tf.summary.merge(summaies)


sess = tf.InteractiveSession()
load_model_ckpt(sess,'ckpt/driving/evolutionary_network/train/')


test_summary_writer = tf.summary.FileWriter('./testboard/'+ds, sess.graph)

step = 0
total_losser = 0

while True:

    print('iteration '+str(step))

    try:
        test_image_batch_fine, test_label_batch_fine, filenamee1, filenamee2 = sess.run([test_image_batch, test_label_batch, filename1, filename2])


        summary_str_test, total_loss2,denormalize_f,Y_forwardd,X_forwardd = sess.run([summary_op,total_loss,denormalized_flow,Y_forward,X_forward],feed_dict={
                    X: test_image_batch_fine,
                    Y: test_label_batch_fine
        })

        step += 1
        total_losser += total_loss2
    except tf.errors.OutOfRangeError:
        print('finish')
        avg = total_losser / step
        print(avg)
        break

    if step == 5:
        break


    # print(filenamee1)
    # print(filenamee2)
    # ij.setImage('normal_img',np.transpose(X_forwardd[:,:,:,:],[0,3,1,2]))
    # ij.setImage('normal_lbl',np.transpose(Y_forwardd[:,:,:,:],[0,3,1,2]))
    # ij.setImage('prediction',np.transpose(denormalize_f,[0,3,1,2]))

    test_summary_writer.add_summary(summary_str_test, step)


test_summary_writer.close()