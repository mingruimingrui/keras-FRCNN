# import numpy as np
#
# C3_shape = np.array([97, 97])
# C4_shape = np.array([48, 48])
# C5_shape = np.array([23, 23])
#
# P6_shape = np.array(np.ceil(C5_shape / 2))
# P7_shape = np.array(np.ceil(P6_shape / 2))
#
# print(C3_shape, C4_shape, C5_shape)
# print(P6_shape, P7_shape)

import tensorflow as tf

# Creates a graph.
a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
c = tf.matmul(a, b)
# Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.
print(sess.run(c))
