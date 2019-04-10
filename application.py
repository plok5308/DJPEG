#2019-04-10 wrriten by Jinseok Park
#DJPEGnet application

from DJPEG_net import *
from PIL import JpegImagePlugin
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os.path

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


def localizing_double_JPEG(img, q_table, stride):
    block_size = 256
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
            result[i_idx, j_idx] = pb[0, 0]
            j_idx = j_idx + 1
        i_idx = i_idx + 1
        print('[{}/{}] Detecting ...'.format(i,H*stride))
    return result


if __name__ == "__main__":
    stride = 32
    dir_name = './'
    file_name = 'copy_move_PQ11.jpg'
    result_name = file_name.split('.')[0] + '_result.jpg'
    file_path = os.path.join(dir_name, file_name)
    result_path = os.path.join(dir_name, result_name)

    #read an image
    img = np.asarray(Image.open(file_path))

    #read quantization table of Y channel from jpeg images
    q_table = read_q_table(file_path)

    with tf.Session() as sess:
        #load pre-trained weights
        saver = tf.train.Saver(write_version=tf.train.SaverDef.V2, max_to_keep=None)
        saver.restore(sess, "DJPEG_weights.ckpt")

        result = localizing_double_JPEG(img, q_table, stride) #localizaing using trained detecting double JPEG network.

        #plot and save the result
        fig = plt.figure()
        columns = 2
        rows = 1
        fig.add_subplot(rows, columns, 1)
        plt.imshow(Image.open(file_path))
        plt.title('input')

        fig.add_subplot(rows, columns, 2)
        result = result*255
        result = result.astype('uint8')
        img_result = Image.fromarray(result)
        img_result.convert("L")
        plt.imshow(img_result, cmap='gray', vmin=0, vmax=255)
        plt.title('result')
        plt.savefig(result_path)
        plt.show()






