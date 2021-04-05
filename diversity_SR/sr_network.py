import tensorflow as tf
import numpy as np
import sys
from utils import *


class condSR_NETWORK():
    def __init__(self, x_LR=None, x_HR=None, condMean=None, condStdDiv=None, z=None, r=None, 
                       rep_size=None, alpha=[1., 0.001, 1.], status='training'):

        status = status.lower()
        if status not in ['training', 'testing']:
            print('Error in network status.')
            exit()

        if status == 'training':
            N, h, w, C = tf.shape(x_LR)[0], x_LR.get_shape()[1], x_LR.get_shape()[2], x_LR.get_shape()[3]
        else:
            N, h, w, C = tf.shape(x_LR)[0], tf.shape(x_LR)[1], tf.shape(x_LR)[2], x_LR.get_shape()[3]

        self.x_LR = x_LR
        self.x_LR_upsampled = tf.image.resize_nearest_neighbor(self.x_LR, [h*tf.reduce_prod(r), w*tf.reduce_prod(r)])
        
        self.z = z
    
        self.x_HR = x_HR    
        self.condMean, self.condStdDiv = condMean, condStdDiv
    
        if r is None:
            print('Error in SR scaling. Variable r must be specified.')
            exit()

        self.x_SR = self.generator(self.z, self.x_LR, r=r)

        self.g_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')

        if status == 'training':
            self.x_SR_reduced = tf.stack([self.downscale_image(self.x_SR[..., i], np.prod(r)) for i in range(C)], axis=-1)

            self.disc_HR = self.discriminator(x_HR, reuse=False)
            self.disc_SR = self.discriminator(self.x_SR, reuse=True)
            self.d_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')

            self.g_con_loss = tf.reduce_mean((self.x_LR - self.x_SR_reduced)**2, axis=[1, 2, 3])

            self.g_adv_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.disc_SR, labels=tf.ones_like(self.disc_SR))
            self.d_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.concat([self.disc_HR, self.disc_SR], axis=0),
                                                                  labels=tf.concat([tf.ones_like(self.disc_HR), tf.zeros_like(self.disc_SR)], axis=0))

            self.adv_perf = [tf.reduce_mean(tf.cast(tf.sigmoid(self.disc_HR) > 0.5, tf.float32)),
                             tf.reduce_mean(tf.cast(tf.sigmoid(self.disc_SR) < 0.5, tf.float32)),
                             tf.reduce_mean(tf.cast(tf.sigmoid(self.disc_SR) > 0.5, tf.float32)),
                             tf.reduce_mean(tf.cast(tf.sigmoid(self.disc_HR) < 0.5, tf.float32))]

            self.g_div_loss = self.compute_div_loss(self.x_LR_upsampled, self.x_SR, self.condMean, self.condStdDiv**2, rep_size)

            self.g_loss = alpha[0]*tf.reduce_mean(self.g_con_loss) + alpha[1]*tf.reduce_mean(self.g_adv_loss) + alpha[2]*tf.reduce_mean(self.g_div_loss)
            self.d_loss = tf.reduce_mean(self.d_loss)

            self.g_con_loss = tf.reduce_mean(self.g_con_loss)
            self.g_adv_loss = tf.reduce_mean(self.g_adv_loss)
            self.g_div_loss = tf.reduce_mean(self.g_div_loss)
        else:
            self.disc_HR, self.disc_SR, self.d_variables = None, None, None
            self.x_LR_upsampled = None
            self.g_loss, self.g_con_loss, self.g_adv_loss, g_div_loss = None, None, None, None
            self.d_loss, self.adv_perf = None, None

    def downscale_image(self, x, K):
        x = tf.expand_dims(x, axis=-1)
        weight = tf.constant(1.0/K**2, shape=[K, K, 1, 1], dtype=tf.float32)
        downscaled = tf.nn.conv2d(x, filter=weight, strides=[1, K, K, 1], padding='SAME')
        return downscaled[..., 0]

    def generator(self, z, x, r=None, reuse=False):
        N, h, w, C = tf.shape(x)[0], x.get_shape()[1], x.get_shape()[2], x.get_shape()[3]
        
        with tf.variable_scope('generator', reuse=reuse):
            with tf.variable_scope('deconv1'):
                C_in, C_out = C, 56
                x = deconv_layer_2d(x, [3, 3, C_out, C_in], [N, h, w, C_out], 1)
                x = tf.nn.relu(x)

                x = tf.concat([x, z], 3)

            skip_connection = x

            # B residual blocks
            C_in, C_out = C_out+8, 64
            for i in range(16):
                B_skip_connection = x

                with tf.variable_scope('block_{}a'.format(i+1)):
                    x = deconv_layer_2d(x, [3, 3, C_out, C_in], [N, h, w, C_out], 1)
                    x = tf.nn.relu(x)

                with tf.variable_scope('block_{}b'.format(i+1)):
                    x = deconv_layer_2d(x, [3, 3, C_out, C_in], [N, h, w, C_out], 1)

                x = tf.add(x, B_skip_connection)

            with tf.variable_scope('deconv2'):
                x = deconv_layer_2d(x, [3, 3, C_out, C_in], [N, h, w, C_out], 1)
                x = tf.add(x, skip_connection)

            # Super resolution scaling
            r_prod = 1
            for i, r_i in enumerate(r):
                C_out = (r_i**2)*C_in
                with tf.variable_scope('deconv{}'.format(i+3)):
                    x = deconv_layer_2d(x, [3, 3, C_out, C_in], [N, r_prod*h, r_prod*w, C_out], 1)
                    x = tf.depth_to_space(x, r_i)
                    x = tf.nn.relu(x)

                r_prod *= r_i

            C_out = C
            with tf.variable_scope('deconv_out'):
                x = deconv_layer_2d(x, [3, 3, C_out, C_in], [N, r_prod*h, r_prod*w, C_out], 1)

        return x


    def discriminator(self, x, reuse=False):
        N, h, w, C = tf.shape(x)[0], x.get_shape()[1], x.get_shape()[2], x.get_shape()[3]

        # alternate the stride between 1 and 2 every other layer
        with tf.variable_scope('discriminator', reuse=reuse):
            with tf.variable_scope('conv1'):
                x = conv_layer_2d(x, [3, 3, C, 32], 1)
                x = tf.nn.leaky_relu(x, alpha=0.2)

            with tf.variable_scope('conv2'):
                x = conv_layer_2d(x, [3, 3, 32, 32], 2)
                x = tf.nn.leaky_relu(x, alpha=0.2)

            with tf.variable_scope('conv3'):
                x = conv_layer_2d(x, [3, 3, 32, 64], 1)
                x = tf.nn.leaky_relu(x, alpha=0.2)

            with tf.variable_scope('conv4'):
                x = conv_layer_2d(x, [3, 3, 64, 64], 2)
                x = tf.nn.leaky_relu(x, alpha=0.2)

            with tf.variable_scope('conv5'):
                x = conv_layer_2d(x, [3, 3, 64, 128], 1)
                x = tf.nn.leaky_relu(x, alpha=0.2)

            with tf.variable_scope('conv6'):
                x = conv_layer_2d(x, [3, 3, 128, 128], 2)
                x = tf.nn.leaky_relu(x, alpha=0.2)

            with tf.variable_scope('conv7'):
                x = conv_layer_2d(x, [3, 3, 128, 256], 1)
                x = tf.nn.leaky_relu(x, alpha=0.2)

            with tf.variable_scope('conv8'):
                x = conv_layer_2d(x, [3, 3, 256, 256], 2)
                x = tf.nn.leaky_relu(x, alpha=0.2)

            x = flatten_layer(x)
            with tf.variable_scope('fully_connected1'):
                x = dense_layer(x, 1024)
                x = tf.nn.leaky_relu(x, alpha=0.2)
            
            with tf.variable_scope('fully_connected2'):
                x = dense_layer(x, 1)

        return x

    def compute_div_loss(self, x_LR, x_SR, condMean_true, condVar_true, rep_size):
        N, h, w, C = tf.shape(x_LR)[0], x_LR.get_shape()[1], x_LR.get_shape()[2], x_LR.get_shape()[3]

        x_LR = tf.reshape(x_LR, [-1, rep_size, int(h), int(w), int(C)])
        x_SR = tf.reshape(x_SR, [-1, rep_size, int(h), int(w), int(C)])

        x_diff = x_SR - x_LR
        condMean_gen, condVar_gen = tf.nn.moments(x_diff, axes=1)

        div_err_mean = tf.reduce_mean((condMean_true - condMean_gen)**2, axis=[1, 2])
        div_err_var  = tf.reduce_mean(condVar_true + condVar_gen - 2*tf.multiply(tf.sqrt(condVar_true), tf.sqrt(condVar_gen)), axis=[1, 2])

        return tf.reduce_mean(div_err_mean + div_err_var, axis=-1)

   