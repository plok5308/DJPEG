#2018-11-26 wrriten by Jinseok Park
#Double JPEG detection network

from Network_Functions import *

# input and labels
x = tf.placeholder(tf.float32, shape=[None, 256, 256, 3])
y_ = tf.placeholder(tf.float32, shape=[None, 2])
QT = tf.placeholder(tf.float32, shape=[None, 8, 8])
phase = tf.placeholder(tf.bool, name='phase')

# Quantization table to input
QT_v = tf.reshape(QT, [-1, 8 * 8])
yuv = tf.image.rgb_to_yuv(x)
x1 = yuv[:, :, :, 0]
x1 = tf.reshape(x1, [-1, 256, 256, 1])
x1 = x1 * 255

# extract DCT according to position(8x8)
# load DCT basis functions
DCT_basis_64 = load_DCT_basis_64()
np_basis = np.zeros((8, 8, 1, 64))
for i in range(64):
	np_basis[:, :, 0, i] = DCT_basis_64[:, :, i]

w_basis = tf.constant(np_basis.tolist())
x2 = conv2d_8(x1, w_basis)  # Nx32x32x64

gamma = 1e+06
for b in range(-60, 61):
	x3 = tf.divide(tf.reduce_sum(tf.sigmoid(tf.scalar_mul(gamma, tf.subtract(x2, b))), [1, 2]), 1024)
	x3 = tf.reshape(x3, [-1, 1, 64])

	if b == -60:
		x4 = x3
	else:
		# x4 = tf.concat(1,[x4,x3])
		x4 = tf.concat([x4, x3], 1)

x5 = x4[:, 0:120, :] - x4[:, 1:121, :]
x6 = tf.reshape(x5, [-1, 120, 64, 1])

# Convnet
# input Nx120x64x1
w1 = weight_variable([5, 5, 1, 64])
h1 = ReLU(batch_norm(conv2d(x6, w1), phase))
# 1-2
w1_2 = weight_variable([5, 5, 64, 64])
h1_2 = ReLU(batch_norm(conv2d(h1, w1_2), phase))
p1 = max_pool_2x2(h1_2)

# Nx60x32x64
w2 = weight_variable([5, 5, 64, 128])
h2 = ReLU(batch_norm(conv2d(p1, w2), phase))
p2 = max_pool_2x2(h2)

# Nx30x16x128
w3 = weight_variable([5, 5, 128, 256])
h3 = ReLU(batch_norm(conv2d(p2, w3), phase))
p3 = max_pool_2x2(h3)
p3_flat = tf.reshape(p3, [-1, 15 * 8 * 256])

# combine quantization table information 1
com_flat = tf.concat([QT_v, p3_flat], 1)  # [-1, 64] + [-1, 1800]

# Fully connected layer variable 1
w_fc1 = weight_variable([15 * 8 * 256 + 64, 500])
b_fc1 = bias_variable([500])
h_fc1 = ReLU(FC(com_flat, w_fc1) + b_fc1)

# combine quantization table information 2
com_flat2 = tf.concat([QT_v, h_fc1], 1)  # [-1, 64] + [-1, 500]

# Fully connected layer variable 2
w_fc2 = weight_variable([500 + 64, 500])
b_fc2 = bias_variable([500])
h_fc2 = ReLU(FC(com_flat2, w_fc2) + b_fc2)

# combine quantization table information 3
com_flat3 = tf.concat([QT_v, h_fc2], 1)  # [-1, 64] + [-1, 500]

# Fully connected layer variable 3
w_fc3 = weight_variable([500 + 64, 2])
b_fc3 = bias_variable([2])
y = FC(com_flat3, w_fc3) + b_fc3
y_softmax = tf.nn.softmax(y)
