import data_reader
import losses_helper
import tensorflow as tf
import network

def get_network_input_forward(image_batch,label_batch):
    return image_batch[:,0,:,:,:], label_batch[:,0,:,:,:]
 
ds = 'ptb'
filee = ['../dataset_synthetic/'+ds+'_TEST.tfrecords']


test_dataset = data_reader.read_with_dataset_api_test(1,filee,version='1')
test_iterator = test_dataset.make_one_shot_iterator()


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
flows_dict = get_predict_flow_forward_backward(predict_flows)

################ epe loss #######################

total_loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(Y_forward, flows_dict['predict_flow'][0]))))


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


denormalized_flow = losses_helper.denormalize_flow(flows_dict['predict_flow'][0])
warped_img = losses_helper.flow_warp(X_forward[:,:,:,4:7],denormalized_flow)


summaies.append(tf.summary.image('warped_img',tf.concat([X_forward[:,:,:,0:3],warped_img],axis=2)))
summaies.append(tf.summary.image('image_forward',img_forward))
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
load_model_ckpt(sess,'ckpt/driving/network_with_fb/train/')


test_summary_writer = tf.summary.FileWriter('./testboard/'+ds, sess.graph)

step = 0
while True:

    print('iteration '+str(step))
    test_image_batch_fine, test_label_batch_fine, filenamee1, filenamee2 = sess.run([test_image_batch, test_label_batch, filename1, filename2])


    summary_str_test, total_loss2 = sess.run([summary_op,total_loss],feed_dict={
                X: test_image_batch_fine,
                Y: test_label_batch_fine
    })

    test_summary_writer.add_summary(summary_str_test, step)

    print(total_loss2)
    step+=1

test_summary_writer.close()