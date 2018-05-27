import numpy as np

C3_shape = np.array([97, 97])
C4_shape = np.array([48, 48])
C5_shape = np.array([23, 23])

P6_shape = np.array(np.ceil(C5_shape / 2))
P7_shape = np.array(np.ceil(P6_shape / 2))

print(C3_shape, C4_shape, C5_shape)
print(P6_shape, P7_shape)
