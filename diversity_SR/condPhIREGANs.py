import os
import numpy as np
import tensorflow as tf
from time import strftime, time
from utils import plot_SR_data
from sr_network import condSR_NETWORK

#os.environ['KMP_DUPLICATE_LIB_OK']='True'

class condPhIREGANs:
    # Network training meta-parameters
    DEFAULT_N_EPOCHS = 10 # Number of epochs of training
    DEFAULT_LEARNING_RATE = 1e-4 # Learning rate for gradient descent (may decrease to 1e-5 after initial training)
    DEFAULT_EPOCH_SHIFT = 0 # If reloading previously trained network, what epoch to start at
    DEFAULT_SAVE_EVERY = 10 # How frequently (in epochs) to save model weights
    DEFAULT_PRINT_EVERY = 2 # How frequently (in iterations) to write out performance

    def __init__(self, data_type, N_epochs=None, learning_rate=None, epoch_shift=None, save_every=None, print_every=None, mu_sig=None):

        self.N_epochs      = N_epochs if N_epochs is not None else self.DEFAULT_N_EPOCHS
        self.learning_rate = learning_rate if learning_rate is not None else self.DEFAULT_LEARNING_RATE
        self.epoch_shift   = epoch_shift if epoch_shift is not None else self.DEFAULT_EPOCH_SHIFT
        self.save_every    = save_every if save_every is not None else self.DEFAULT_SAVE_EVERY
        self.print_every   = print_every if print_every is not None else self.DEFAULT_PRINT_EVERY

        self.data_type = data_type
        self.mu_sig = mu_sig
        self.LR_data_shape = None

        # Set various paths for where to save data
        self.run_id        = '-'.join([self.data_type, strftime('%Y%m%d-%H%M%S')])
        self.model_name    = '/'.join(['models', self.run_id])
        self.data_out_path = '/'.join(['data_out', self.run_id])

    def setSave_every(self, in_save_every):
        self.save_every = in_save_every

    def setPrint_every(self, in_print_every):
        self.print_every = in_print_every

    def setEpochShift(self, shift):
        self.epoch_shift = shift

    def setNum_epochs(self, in_epochs):
        self.N_epochs = in_epochs

    def setLearnRate(self, learn_rate):
        self.learning_rate = learn_rate

    def setModel_name(self, in_model_name):
        self.model_name = in_model_name

    def set_data_out_path(self, in_data_path):
        self.data_out_path = in_data_path
    
    def reset_run_id(self):
        self.run_id        = '-'.join([self.data_type, strftime('%Y%m%d-%H%M%S')])
        self.model_name    = '/'.join(['model', self.run_id])
        self.data_out_path = '/'.join(['data_out', self.run_id])

    def train(self, r, data_path, model_path=None, batch_size=10, rep_size=10, alpha=[1., 0.01, 1.]):
        '''
            This method trains the generator using a disctiminator/adversarial training.

            inputs:
                r            - (int array) should be array of prime factorization of amount of super-resolution to perform
                data_path    - (string) path of training data file to load in
                model_path   - (string) path of previously pretrained or trained model to load
                batch_size   - (int) number of images to grab per batch. decrease if running out of memory
                rep_size     - (int) number of distinct realizations to generate from each LR field
                alpha_advers - (float array) scaling values for the content, adversarial, and diversity losses

            output:
                g_saved_model - (string) path to the trained generator model
        '''

        tf.reset_default_graph()

        if self.mu_sig is None:
            self.set_mu_sig(data_path, batch_size)

        self.set_LR_data_shape(data_path)
        h, w, C = self.LR_data_shape

        print('Initializing network ...', end=' ')
        x_LR = tf.placeholder(tf.float32, [None, h,             w,            C])
        x_HR = tf.placeholder(tf.float32, [None, h*np.prod(r),  w*np.prod(r), C])
        z    = tf.placeholder(tf.float32, [None, h,             w,            8])
        condMean = tf.placeholder(tf.float32, [None, h*np.prod(r), w*np.prod(r), C])
        condStdDiv  = tf.placeholder(tf.float32, [None, h*np.prod(r), w*np.prod(r), C])

        model = condSR_NETWORK(x_LR, x_HR, condMean, condStdDiv, z=z, 
                               r=r, rep_size=rep_size, status='training', alpha=alpha)

        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        g_train_op = optimizer.minimize(model.g_loss, var_list=model.g_variables)
        d_train_op = optimizer.minimize(model.d_loss, var_list=model.d_variables)

        g_saver = tf.train.Saver(var_list=model.g_variables, max_to_keep=100000)
        gd_saver = tf.train.Saver(var_list=(model.g_variables+model.d_variables), max_to_keep=10000)

        init = tf.global_variables_initializer()
        print('Done.')
        
        print('Building data pipeline ...', end=' ')
        ds_train = tf.data.TFRecordDataset(data_path)
        ds_train = ds_train.map(lambda xx: self._parse_train_(xx, self.mu_sig)).shuffle(100).batch(batch_size)

        iterator = tf.data.Iterator.from_structure(ds_train.output_types,
                                                   ds_train.output_shapes)
        idx, LR_out, HR_out, condMean_out, condStdDiv_out = iterator.get_next()

        init_iter_train = iterator.make_initializer(ds_train)
        print('Done.')
        
        with tf.Session() as sess:
            print('Training network ...')

            sess.run(init)

            if model_path is not None:
                print('Loading previously trained network...', end=' ')
                if 'gan-all' in model_path:
                    # Load both pretrained generator and discriminator networks
                    gd_saver.restore(sess, model_path)
                else:
                    # Load only pretrained generator network, start discriminator training from scratch
                    g_saver.restore(sess, model_path)
                print('Done.')

            # Start training
            iters = 0
            for epoch in range(self.epoch_shift+1, self.epoch_shift+self.N_epochs+1):
                print('Epoch: '+str(epoch))
                start_time = time()

                model_losses = [model.g_loss, model.d_loss, model.adv_perf]

                # Loop through training data
                sess.run(init_iter_train)
                try:
                    epoch_g_loss, epoch_d_loss, N = 0, 0, 0
                    while True:
                        batch_idx, batch_LR, batch_HR, batch_condMean, batch_condStdDiv = sess.run([idx, LR_out, HR_out, condMean_out, condStdDiv_out])
                        batch_LR, batch_HR = np.repeat(batch_LR, rep_size, axis=0), np.repeat(batch_HR, rep_size, axis=0)
                        N_batch = batch_LR.shape[0]
                        feed_dict = {x_HR:batch_HR, x_LR:batch_LR, condMean:batch_condMean, condStdDiv:batch_condStdDiv}

                        batch_z = np.random.uniform(low=-1.0, high=1.0, size=[N_batch, h, w, 8])
                        feed_dict[z] = batch_z

                        # Initial training of the discriminator and generator
                        sess.run(d_train_op, feed_dict=feed_dict)
                        sess.run(g_train_op, feed_dict=feed_dict)
                        
                        #calculate current discriminator losses
                        gl, dl, p = sess.run(model_losses, feed_dict=feed_dict)
                        
                        gen_count = 1
                        while (dl < 0.460) and gen_count < 10:
                            batch_z = np.random.uniform(low=-1.0, high=1.0, size=[N_batch, h, w, 8])
                            feed_dict[z] = batch_z

                            sess.run(g_train_op, feed_dict=feed_dict)
                            gl, dl, p = sess.run(model_losses, feed_dict=feed_dict)
                            gen_count += 1

                        dis_count = 1
                        while (dl >= 0.6) and dis_count < 10:
                            batch_z = np.random.uniform(low=-1.0, high=1.0, size=[N_batch, h, w, 8])
                            feed_dict[z] = batch_z

                            sess.run(d_train_op, feed_dict=feed_dict)
                            gl, dl, p = sess.run(model_losses, feed_dict=feed_dict)
                            dis_count += 1

                        epoch_g_loss += gl*N_batch
                        epoch_d_loss += dl*N_batch
                        N += N_batch
                        
                        iters += 1
                        if (iters % self.print_every) == 0:
                            gl, dl, p = sess.run(model_losses, feed_dict=feed_dict)
                            g_cl, g_al, g_dl = sess.run([model.g_con_loss, model.g_adv_loss, model.g_div_loss], feed_dict=feed_dict)

                            print('Number of G steps=%d, Number of D steps=%d' %(gen_count, dis_count))
                            print('G loss=%.5f, Content loss=%.5f, Adversarial loss=%.5f, Diversity loss=%.5f' %(gl, np.mean(g_cl), np.mean(g_al), np.mean(g_dl)))
                            print('D loss=%.5f' %(dl))
                            print('TP=%.5f, TN=%.5f, FP=%.5f, FN=%.5f' %(p[0], p[1], p[2], p[3]))
                            print('', flush=True)

                except tf.errors.OutOfRangeError:
                    pass
                
                if (epoch % self.save_every) == 0:
                    g_model_dr  = '/'.join([self.model_name, 'gan{0:05d}'.format(epoch), 'gan'])
                    gd_model_dr = '/'.join([self.model_name, 'gan-all{0:05d}'.format(epoch), 'gan'])
                    if not os.path.exists(self.model_name):
                        os.makedirs(self.model_name)
                    g_saver.save(sess,  g_model_dr)
                    gd_saver.save(sess, gd_model_dr)

                g_loss = epoch_g_loss/N
                d_loss = epoch_d_loss/N

                print('Epoch generator training loss=%.5f, discriminator training loss=%.5f' %(g_loss, d_loss))
                print('Epoch took %.2f seconds\n' %(time() - start_time), flush=True)

            g_model_dr  = '/'.join([self.model_name, 'gan', 'gan'])
            gd_model_dr = '/'.join([self.model_name, 'gan-all', 'gan'])
            if not os.path.exists(self.model_name):
                os.makedirs(self.model_name)
            g_saver.save(sess,  g_model_dr)
            gd_saver.save(sess, gd_model_dr)

        print('Done.')

        return g_model_dr

    def test(self, r, data_path, model_path, batch_size=10, rep_size=10, plot_data=False):
        '''
            This method loads a previously trained model and runs it on test data

            inputs:
                r          - (int array) should be array of prime factorization of amount of super-resolution to perform
                data_path  - (string) path of test data file to load in
                model_path - (string) path of model to load in
                batch_size - (int) number of images to grab per batch. decrease if running out of memory
                rep_size   - (int) number of distinct realizations to generate from each LR field
                plot_data  - (bool) flag for whether or not to plot LR and SR images
        '''

        tf.reset_default_graph()
        
        assert self.mu_sig is not None, 'Value for mu_sig must be set first.'
        
        self.set_LR_data_shape(data_path)
        h, w, C = self.LR_data_shape

        print('Initializing network ...', end=' ')
        x_LR = tf.placeholder(tf.float32, [None, h, w, C])
        z    = tf.placeholder(tf.float32, [None, h, w, 8])

        model = condSR_NETWORK(x_LR, z=z, r=r, rep_size=rep_size, status='testing')

        g_saver = tf.train.Saver(var_list=model.g_variables, max_to_keep=10000)
        init = tf.global_variables_initializer()
        print('Done.')

        print('Building data pipeline ...', end=' ')
        ds_test = tf.data.TFRecordDataset(data_path)
        ds_test = ds_test.map(lambda xx: self._parse_test_(xx, self.mu_sig)).batch(batch_size)

        iterator = tf.data.Iterator.from_structure(ds_test.output_types,
                                                   ds_test.output_shapes)
        idx, LR_out = iterator.get_next()

        init_iter_test = iterator.make_initializer(ds_test)
        print('Done.')

        with tf.Session() as sess:
            print('Loading saved network ...', end=' ')
            sess.run(init)
            g_saver.restore(sess, model_path)
            print('Done.')

            print('Running test data ...')

            # Loop through test data
            sess.run(init_iter_test)
            try:
                LR_data_out, SR_data_out = None, None
                while True:
                    batch_idx, batch_LR = sess.run([idx, LR_out])
                    N_batch = batch_LR.shape[0]

                    batch_LR = np.repeat(batch_LR, rep_size, axis=0)
                    batch_z = np.random.uniform(low=-1.0, high=1.0, size=[N_batch*rep_size, h, w, 8])

                    feed_dict = {x_LR:batch_LR, z:batch_z}

                    batch_SR = sess.run(model.x_SR, feed_dict=feed_dict)

                    batch_LR = batch_LR.reshape((N_batch, rep_size, h, w, C))
                    batch_SR = batch_SR.reshape((N_batch, rep_size, h*np.prod(r), w*np.prod(r), C))

                    batch_LR = self.mu_sig[1]*batch_LR[:, 0, ...] + self.mu_sig[0]
                    batch_SR = self.mu_sig[1]*batch_SR + self.mu_sig[0]

                    if plot_data:
                        img_path = '/'.join([self.data_out_path, 'imgs'])
                        if not os.path.exists(img_path):
                            os.makedirs(img_path)
                        plot_SR_data(batch_idx, batch_LR, batch_SR, img_path)

                    if SR_data_out is None:
                        LR_data_out, SR_data_out = batch_LR, batch_SR
                        condMean_out, condStdDev_out = np.mean(batch_SR, axis=1), np.std(batch_SR, axis=1)
                    else:
                        LR_data_out = np.concatenate((LR_data_out, batch_LR), axis=0)
                        SR_data_out = np.concatenate((SR_data_out, batch_SR), axis=0)

            except tf.errors.OutOfRangeError:
                pass

            if not os.path.exists(self.data_out_path):
                os.makedirs(self.data_out_path)
            np.save(self.data_out_path+'/test_LR.npy', LR_data_out)
            np.save(self.data_out_path+'/test_SR.npy', SR_data_out)

        print('Done.')

    def _parse_train_(self, serialized_example, mu_sig=None):
        feature = {
          'index'    : tf.FixedLenFeature([], tf.int64),
          'data_LR'  : tf.FixedLenFeature([], tf.string),
          'h_LR'     : tf.FixedLenFeature([], tf.int64),
          'w_LR'     : tf.FixedLenFeature([], tf.int64),
          'data_HR'  : tf.FixedLenFeature([], tf.string),
          'h_HR'     : tf.FixedLenFeature([], tf.int64),
          'w_HR'     : tf.FixedLenFeature([], tf.int64),
          'c'        : tf.FixedLenFeature([], tf.int64),
          'mean'     : tf.FixedLenFeature([], tf.string),
          'std'      : tf.FixedLenFeature([], tf.string)}

        example = tf.parse_single_example(serialized_example, feature)

        idx = example['index']

        h_LR, w_LR = example['h_LR'], example['w_LR']
        h_HR, w_HR = example['h_HR'], example['w_HR']

        c = example['c']

        condMean = tf.decode_raw(example['mean'], tf.float64)
        condMean = tf.reshape(condMean, (h_HR, w_HR, c))

        condStdDiv = tf.decode_raw(example['std'], tf.float64)
        condStdDiv = tf.reshape(condStdDiv, (h_HR, w_HR, c))

        data_LR = tf.decode_raw(example['data_LR'], tf.float64)
        data_HR = tf.decode_raw(example['data_HR'], tf.float64)

        data_LR = tf.reshape(data_LR, (h_LR, w_LR, c))
        data_HR = tf.reshape(data_HR, (h_HR, w_HR, c))

        if mu_sig is not None:
            data_LR = (data_LR - mu_sig[0])/mu_sig[1]
            data_HR = (data_HR - mu_sig[0])/mu_sig[1]
            condMean = condMean/mu_sig[1]
            condStdDiv = condStdDiv/mu_sig[1]

        return idx, data_LR, data_HR, condMean, condStdDiv

    def _parse_test_(self, serialized_example, mu_sig=None):
        feature = {
          'index'    : tf.FixedLenFeature([], tf.int64),
          'data_LR'  : tf.FixedLenFeature([], tf.string),
          'h_LR'     : tf.FixedLenFeature([], tf.int64),
          'w_LR'     : tf.FixedLenFeature([], tf.int64),
          'c'        : tf.FixedLenFeature([], tf.int64)}

        example = tf.parse_single_example(serialized_example, feature)

        idx = example['index']

        h_LR, w_LR = example['h_LR'], example['w_LR']

        c = example['c']

        data_LR = tf.decode_raw(example['data_LR'], tf.float64)

        data_LR = tf.reshape(data_LR, (h_LR, w_LR, c))

        if mu_sig is not None:
            data_LR = (data_LR - mu_sig[0])/mu_sig[1]

        return idx, data_LR

    def set_mu_sig(self, data_path, batch_size):
        '''
            Compute mu, sigma for all channels of data
            inputs:
                data_path - (string) should be the path to the TRAINING DATA since mu and sigma are
                            calculated based on the trainind data regardless of if pretraining,
                            training, or testing.
                batch_size - number of samples to grab each interation. will be passed in directly
                             from pretrain, train, or test method by default.
            outputs:
                sets self.mu_sig
        '''
        print('Loading data ...', end=' ')
        dataset = tf.data.TFRecordDataset(data_path)
        dataset = dataset.map(self._parse_train_).batch(batch_size)

        iterator = dataset.make_one_shot_iterator()
        _, _, HR_out, _, _ = iterator.get_next()

        with tf.Session() as sess:
            N, mu, sigma = 0, 0, 0
            try:
                while True:
                    data_HR = sess.run(HR_out)

                    N_batch, h, w, c = data_HR.shape
                    N_new = N + N_batch

                    mu_batch = np.mean(data_HR, axis=(0, 1, 2))
                    sigma_batch = np.var(data_HR, axis=(0, 1, 2))

                    sigma = (N/N_new)*sigma + (N_batch/N_new)*sigma_batch + (N*N_batch/N_new**2)*(mu - mu_batch)**2
                    mu = (N/N_new)*mu + (N_batch/N_new)*mu_batch

                    N = N_new

            except tf.errors.OutOfRangeError:
                pass

        self.mu_sig = [mu, np.sqrt(sigma)]

        print('Done.')

    def set_LR_data_shape(self, data_path):
        '''
            Compute mu, sigma for all channels of data
            inputs:
                data_path - (string) should be the path to the TRAINING DATA since mu and sigma are
                            calculated based on the trainind data regardless of if pretraining,
                            training, or testing.
            outputs:
                sets self.LR_data_shape
        '''
        print('Loading data ...', end=' ')
        dataset = tf.data.TFRecordDataset(data_path)
        dataset = dataset.map(self._parse_test_).batch(1)

        iterator = dataset.make_one_shot_iterator()
        _, LR_out = iterator.get_next()

        with tf.Session() as sess:
            data_LR = sess.run(LR_out)
        
        self.LR_data_shape = data_LR.shape[1:]

        print('Done.')
