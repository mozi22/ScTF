import lmbspecialops as sops
import tensorflow as tf
a = tf.ones([100,100,100,100],dtype=tf.float32)

b = tf.ones([100,100,100,100],dtype=tf.float32)

# sops.correlation(input1=a,
# 				 input2=b,
# 				 kernel_size=5,
# 				 max_displacement=5,
# 				 stride1=1,
# 				 stride2=1,
# 				 pad_size=0)



sops.leaky_relu(a,leak=0.1)