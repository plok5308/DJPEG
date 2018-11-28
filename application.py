#2018-11-26 wrriten by Jinseok Park
#DJPEGnet application

from skimage.util.shape import view_as_windows
from DJPEG_net import *
from PIL import JpegImagePlugin
import numpy as np
import matplotlib.pyplot as plt


stride = 32
batch_size = 50
block_size = 256

def read_q_table(file_name):
    jpg = JpegImagePlugin.JpegImageFile(file_name)
    qtable = JpegImagePlugin.convert_dict_qtables(jpg.quantization)
    Y_qtable = qtable[0]
    Y_qtable_2d = np.zeros((8, 8))

    qtable_idx = 0
    for i in range(0, 8):
        for j in range(0, 8):
            Y_qtable_2d[i, j] = Y_qtable[qtable_idx]
            qtable_idx = qtable_idx + 1

    return Y_qtable_2d


def localizing_double_JPEG(img, q_table):
    batch_q = q_table.reshape(1, 8, 8)
    batch_q = batch_q.astype('float')

    h, w = img.shape[0:2]
    H = (h - block_size) // stride
    W = (w - block_size) // stride
    result = np.zeros((H, W))

    i_idx = 0
    for i in range(0, H * stride, stride):
        j_idx = 0
        for j in range(0, W * stride, stride):
            block = img[i:i + block_size, j:j + block_size,:]
            batch_x = block.reshape(1, block_size, block_size, 3)
            batch_x = batch_x.astype('float')
            batch_x = np.divide(batch_x, 255)
            pb = sess.run(y_softmax, feed_dict={x: batch_x, QT: batch_q, phase: False})
            # print pb
            result[i_idx, j_idx] = pb[0, 1]
            j_idx = j_idx + 1
        i_idx = i_idx + 1
        print(i)
    return result


if __name__ == "__main__":
    file_name = 'manp_PQ11.jpg'
    result_name = file_name.split('.')[0] + '_result.jpg'
    img = np.asarray(Image.open(file_name))
    #read quantization table of Y channel from jpeg images
    q_table = read_q_table(file_name)

    with tf.Session() as sess:
        saver = tf.train.Saver(write_version=tf.train.SaverDef.V2, max_to_keep=None)
        saver.restore(sess, "DJPEG_weights.ckpt")

        result = localizing_double_JPEG(img, q_table) #localizaing using trained detecting double JPEG network.

        #plot and save the result
        fig = plt.figure()
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(result, cmap='binary')
        plt.savefig(result_name)





