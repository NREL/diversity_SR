import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def conv_layer_2d(x, filter_shape, stride, trainable=True):
    filter_ = tf.get_variable(
        name='weight',
        shape=filter_shape,
        dtype=tf.float32,
        initializer=tf.contrib.layers.xavier_initializer(),
        trainable=trainable)
    bias_ = tf.get_variable(
        name='bias',
        shape=[filter_shape[-1]],
        dtype=tf.float32,
        initializer=tf.contrib.layers.xavier_initializer(),
        trainable=trainable)
    x = tf.nn.bias_add(tf.nn.conv2d(
        input=x,
        filter=filter_,
        strides=[1, stride, stride, 1],
        padding='SAME'), bias_)
    return x

def deconv_layer_2d(x, filter_shape, output_shape, stride, trainable=True):
    filter_ = tf.get_variable(
        name='weight',
        shape=filter_shape,
        dtype=tf.float32,
        initializer=tf.contrib.layers.xavier_initializer(),
        trainable=trainable)
    bias_ = tf.get_variable(
        name='bias',
        shape=[output_shape[-1]],
        dtype=tf.float32,
        initializer=tf.contrib.layers.xavier_initializer(),
        trainable=trainable)
    x = tf.nn.bias_add(tf.nn.conv2d_transpose(
        value=x,
        filter=filter_,
        output_shape=output_shape,
        strides=[1, stride, stride, 1],
        padding='SAME'), bias_)
    return x

def flatten_layer(x):
    input_shape = x.get_shape().as_list()
    if len(input_shape) == 5:
        dim = input_shape[1] * input_shape[2] * input_shape[3]
        transposed = tf.transpose(x, (0, 4, 1, 2, 3))

        return tf.reshape(transposed, [-1, dim])
    elif len(input_shape) == 4:
        dim = input_shape[1] * input_shape[2]
        transposed = tf.transpose(x, (0, 3, 1, 2))

        return tf.reshape(transposed, [-1, dim])

def dense_layer(x, out_dim, trainable=True):
    in_dim = x.get_shape().as_list()[-1]
    W = tf.get_variable(
        name='weight',
        shape=[in_dim, out_dim],
        dtype=tf.float32,
        initializer=tf.truncated_normal_initializer(stddev=0.02),
        trainable=trainable)
    b = tf.get_variable(
        name='bias',
        shape=[out_dim],
        dtype=tf.float32,
        initializer=tf.constant_initializer(0.0),
        trainable=trainable)
    return tf.add(tf.matmul(x, W), b)

def plot_SR_data(idx, LR, SR, path):

    r0 = int(SR.shape[2]/LR.shape[1])
    r1 = int(SR.shape[3]/LR.shape[2])

    LR_exp = np.expand_dims(LR.repeat(r0, axis=1).repeat(r1, axis=2), axis=1)
    res_batch = SR - LR_exp
    
    N_subs = np.minimum(3, SR.shape[1])

    for i in range(LR.shape[0]):
        vmin0 = np.minimum(np.min(LR[i,:,:,0]), np.min(SR[i,:,:,0]))
        vmax0 = np.maximum(np.max(LR[i,:,:,0]), np.max(SR[i,:,:,0]))
        vmin1 = np.minimum(np.min(LR[i,:,:,1]), np.min(SR[i,:,:,1]))
        vmax1 = np.maximum(np.max(LR[i,:,:,1]), np.max(SR[i,:,:,1]))

        fig_width = 9 + 2*N_subs
        fig, ax = plt.subplots(2, 1+N_subs, figsize=(fig_width, 8))
        
        im = ax[0, 0].imshow(LR[i, :, :, 0], vmin=vmin0, vmax=vmax0, cmap='viridis', origin='lower')
        ax[0, 0].set_title('LR 0 Input', fontsize=9)
        fig.colorbar(im, ax=ax[0, 0])
        ax[0, 0].set_xticks([], [])
        ax[0, 0].set_yticks([], [])
            
        im = ax[1, 0].imshow(LR[i, :, :, 1], vmin=vmin1, vmax=vmax1, cmap='viridis', origin='lower')
        ax[1, 0].set_title('LR 1 Input', fontsize=9)
        fig.colorbar(im, ax=ax[1, 0])
        ax[1, 0].set_xticks([], [])
        ax[1, 0].set_yticks([], [])

        for j in range(N_subs):
            im = ax[0, j+1].imshow(SR[i, j, :, :, 0], vmin=vmin0, vmax=vmax0, cmap='viridis', origin='lower')
            ax[0, j+1].set_title('SR 0 - {}'.format(j), fontsize=9)
            fig.colorbar(im, ax=ax[0, j+1])
            ax[0, j+1].set_xticks([], [])
            ax[0, j+1].set_yticks([], [])

            im = ax[1, j+1].imshow(SR[i, j, :, :, 1], vmin=vmin1, vmax=vmax1, cmap='viridis', origin='lower')
            ax[1, j+1].set_title('SR 1 - {}'.format(j), fontsize=9)
            fig.colorbar(im, ax=ax[1, j+1])
            ax[1, j+1].set_xticks([], [])
            ax[1, j+1].set_yticks([], [])

        plt.savefig(path+'/{0:05d}fields.png'.format(idx[i]), dpi=200, bbox_inches='tight')
        plt.close()

        fig, ax = plt.subplots(2, 2, figsize=(8, 8))
        
        im = ax[0, 0].imshow(np.mean(res_batch[i, ..., 0], axis=0), cmap='viridis', origin='lower')
        ax[0, 0].set_title('Conditional Mean 0', fontsize=9)
        fig.colorbar(im, ax=ax[0, 0])
        ax[0, 0].set_xticks([], [])
        ax[0, 0].set_yticks([], [])
        
        im = ax[0, 1].imshow(np.std(res_batch[i, ..., 0], axis=0), cmap='viridis', origin='lower')
        ax[0, 1].set_title('Conditional Std. Dev. 0', fontsize=9)
        fig.colorbar(im, ax=ax[0, 1])
        ax[0, 1].set_xticks([], [])
        ax[0, 1].set_yticks([], [])
        
        im = ax[1, 0].imshow(np.mean(res_batch[i, ..., 1], axis=0), cmap='viridis', origin='lower')
        ax[1, 0].set_title('Conditional Mean 1', fontsize=9)
        fig.colorbar(im, ax=ax[1, 0])
        ax[1, 0].set_xticks([], [])
        ax[1, 0].set_yticks([], [])
        
        im = ax[1, 1].imshow(np.std(res_batch[i, ..., 1], axis=0), cmap='viridis', origin='lower')
        ax[1, 1].set_title('Conditional Std. Dev. 1', fontsize=9)
        fig.colorbar(im, ax=ax[1, 1])
        ax[1, 1].set_xticks([], [])
        ax[1, 1].set_yticks([], [])

        plt.savefig(path+'/{0:05d}stats.png'.format(idx[i]), dpi=200, bbox_inches='tight')
        plt.close()




