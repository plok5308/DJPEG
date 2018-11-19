import tensorflow as tf
import numpy as np
from scipy.fftpack import dct,idct
from scipy.signal import convolve2d
from PIL import Image
import math

#caclulate DCT basis
def cal_scale(p,q):
	if p==0:
		ap = 1/(math.sqrt(8))
	else:
		ap = math.sqrt(0.25)
	if q==0:
		aq = 1/(math.sqrt(8))
	else:
		aq = math.sqrt(0.25)

	return ap,aq

def cal_basis(p,q):
	basis = np.zeros((8,8))
	ap,aq = cal_scale(p,q)
	for m in range(0,8):
		for n in range(0,8):
			basis[m,n] = ap*aq*math.cos(math.pi*(2*m+1)*p/16)*math.cos(math.pi*(2*n+1)*q/16)

	return basis

def load_DCT_basis_64():
	basis_64 = np.zeros((8,8,64))
	idx = 0
	for i in range(8):
		for j in range(8):
			basis_64[:,:,idx] = cal_basis(i,j)
			idx = idx + 1
	return basis_64


#networks functions
def l2(x):
	return tf.nn.l2_loss(x)

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def batch_norm(x,phase):
	return tf.contrib.layers.batch_norm(x, decay=0.9, epsilon=0.001, center=True, scale=True, is_training=phase)

def conv2d_v(x,W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="VALID")

def conv2d(x,W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")

def conv2d_2(x,W):
    return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding="SAME")

def conv2d_8(x,W):
    return tf.nn.conv2d(x, W, strides=[1, 8, 8, 1], padding="SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

def max_pool_3x1_2_v(x):
	return tf.nn.max_pool(x, ksize=[1, 3, 1, 1], strides=[1, 2, 1, 1], padding="VALID")

def avg_pool_2x2(x):
	return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

def FC(x,W):
	return tf.matmul(x,W)

def ReLU(x):
    return tf.nn.relu(x)
