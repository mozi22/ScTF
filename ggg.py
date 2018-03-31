import lmbspecialops as sops
import tensorflow as tf
a = tf.ones([100,100,100,100],dtype=tf.float32)

b = tf.ones([100,100,100,100],dtype=tf.float32)

result = sops.correlation(input1=a,
				 input2=b,
				 kernel_size=0,
				 max_displacement=15,
				 stride1=1,
				 stride2=1)


print(result)