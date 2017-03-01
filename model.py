from __future__ import division
import os
import time
import math
import numpy as np
from operator import mul
from functools import reduce
import tensorflow as tf

from ops import *
from utils import *


class deepGAN(object):
    def __init__(self, sess, batch_size=64, sample_num=64, z_dim=100,
                 gf_dim=64, df_dim=64, gfc_dim=1024, dfc_dim=1024, col_dim=1,
                 dataset_name='default', input_fname_pattern='*.jpg', output_depth=3,
                 checkpoint_dir=None, sample_dir=None):
        """

        Args:
            sess: TensorFlow session
            batch_size: The size of batch. Should be specified before training.
            y_dim: (optional) Dimension of dim for y. [None]
            z_dim: (optional) Dimension of dim for Z. [100]
            gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
            df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
            gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
            dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
            col_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
        """
        self.sess = sess
        self.batch_size = batch_size
        self.sample_num = sample_num

        self.z_dim = z_dim

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        self.col_dim = col_dim
        self.output_depth = output_depth

        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')

        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')

        self._setup_placeholder()

        self.dataset_name = dataset_name
        self.input_fname_pattern = input_fname_pattern
        self.checkpoint_dir = checkpoint_dir
        self.build_model()

    def build_model(self):
        inputs = self.inputs
        sample_inputs = self.sample_inputs

        self.G = self.generator(self.z, self.cost, name='generator')
        self.D, self.D_logits = self.discriminator(inputs,
                                                   self.cost, reuse=False)

        self.sampler = self.sampler(self.z, self.cost, name="generator")
        self.D_, self.D_logits_ = self.discriminator(self.G,
                                                     self.cost, reuse=True)

        self.d_sum = tf.summary.histogram("d", self.D)
        self.d__sum = tf.summary.histogram("d_", self.D_)
        self.G_sum = tf.summary.image("G", self.G)

        self.d_loss_real = tf.reduce_mean(self.D_logits)
        self.d_loss_fake = tf.reduce_mean(self.D_logits_)

        self.d_loss = tf.reduce_mean(self.d_loss_real - self.d_loss_fake)
        self.g_loss = self.d_loss_fake

        self.d_loss_real_sum = tf.summary.scalar("d_loss_real",
                                                 self.d_loss_real)
        self.d_loss_fake_sum = tf.summary.scalar("d_loss_fake",
                                                 self.d_loss_fake)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'generator_' in var.name]

        self.saver = tf.train.Saver()

    def train(self, config):
        self.data_X, self.data_y = self.load_mnist()
        self.data_X = self.data_X[:1].astype(np.float16)
        self.data_y = self.data_y[:1].astype(np.float16)

        print("generating init pars: ")
        #data_pars = np.float32(
        #    np.random.normal(0, 0.01, (self.sample_num, int(self.cuts[-1]))))
        #padding = np.zeros((self.sample_num, self.output_height
        #                    * self.output_width * self.output_depth
        #                    - int(self.cuts[-1])), dtype=np.float32)
        #data_pars = np.concatenate((data_pars, padding), axis=1)
        print("pars generated!")
        data_cost = np.float16(np.random.random(size=(self.sample_num, 1)))

        print("define optimizer")
        d_optim = tf.train.RMSPropOptimizer(config.learning_rate,
                                         decay=0.5).minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.RMSPropOptimizer(config.learning_rate,
                                         decay=0.5).minimize(self.g_loss, var_list=self.g_vars)
        print("optimizer defined")

        print("initializing global vars...")
        tf.global_variables_initializer().run()
        print("variables initialized")

        self.g_sum = tf.summary.merge([
            self.z_sum, self.d__sum, self.G_sum,
            self.d_loss_fake_sum, self.g_loss_sum])
        self.d_sum = tf.summary.merge([
            self.z_sum, self.d_sum,
            self.d_loss_real_sum,
            self.d_loss_sum])

        self.writer = SummaryWriter("./logs", self.sess.graph)


        counter = 1
        start_time = time.time()

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        for cycle in xrange(config.cycle):
            # use T net with generated pars to calculate true loss,
            # and update new (pars, cost) pair
            data_cost[-self.sample_num:, ] = self.evaluator(
                self.sess, data_pars[-self.sample_num:, ])

            for epoch in xrange(config.epoch):
                batch_idxs = min(len(data_pars), config.train_size) // config.batch_size
                for idx in xrange(0, batch_idxs):
                    batch_pars = data_pars[idx * config.batch_size:(idx + 1) * config.batch_size]
                    batch_cost = data_cost[idx * config.batch_size:(idx + 1) * config.batch_size]
                    batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]).astype(np.float16)

                    # Update D network
                    _, summary_str = self.sess.run([d_optim, d_sum],
                                                    feed_dict={
                                                        self.inputs: batch_pars,
                                                        self.z: batch_z,
                                                        self.cost: batch_cost
                                                    })
                    self.writer.add_summary(summary_str, counter)
                    # Update G network
                    _, summary_str = self.sess.run([g_optim, g_sum], feed_dict={
                        self.z: batch_z,
                        self.cost: batch_cost,
                    })
                    self.writer.add_summary(summary_str, counter)

                    counter += 1
                    exit()
                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" % (epoch, idx, batch_idxs, time.time() - start_time, errD_fake + errD_real, errG))

            # randomly generate pars with expected loss
            max = data_cost.max()
            if data_cost.max - 0.01 < 0:
                min = np.random.uniform(0, data_cost.max(), 1)
            else:
                min = data_cost.max() - 0.01
            expected_cost = np.random.uniform(min, max, (self.sample_num, 1))
            sample_z = np.random.uniform(-1, 1, size=(self.sample_num, self.z_dim))
            generated_pars = self.sess.run(sampler,
                                           feed_dict={
                                               self.z: sample_z,
                                               self.cost: expected_cost
                                           })
            generated_pars = generated_pars[:, int(self.cuts[-1]):] = 0
            data_pars = np.concatenate((data_pars, generated_pars))
            data_cost = np.concatenate((data_cost, expected_cost))

    def generator(self, z, cost, name='generator', reuse=False):
            with tf.variable_scope(name) as scope:
                if reuse:
                    scope.reuse_variables()

                s_h, s_w = self.output_height, self.output_width
                s_h2, s_h4 = math.ceil(s_h / 2), math.ceil(s_h / 4)
                s_w2, s_w4 = math.ceil(s_w / 2), math.ceil(s_w / 4)

                cost_reshaped = tf.reshape(cost, [self.batch_size, 1, 1, 1])
                z = concat([z, cost], 1)

                h0 = tf.nn.relu(
                    self.g_bn0(linear(z, self.gfc_dim, name+'_h0_lin')))
                h0 = concat([h0, cost], 1)

                h1 = tf.nn.relu(self.g_bn1(
                    linear(h0, self.gf_dim * 2 * s_h4 * s_w4, name+'_h1_lin')))
                h1 = tf.reshape(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2])
                print("h1 & cost_reshaped dtype: " + str(h1.dtype) + str(cost_reshaped.dtype))
                h1 = conv_cond_concat(h1, cost_reshaped)

                h2 = tf.nn.relu(self.g_bn2(deconv2d(h1,
                                                    [self.batch_size, s_h2, s_w2, self.gf_dim * 2], name=name+'_h2')))
                h2 = conv_cond_concat(h2, cost_reshaped)

                return tf.nn.sigmoid(
                    deconv2d(h2, [self.batch_size, s_h, s_w, self.output_depth], name=name+'_h3'))

    def discriminator(self, input, cost=None, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            cost_reshaped = tf.reshape(cost, [self.batch_size, 1, 1, 1])
            x = conv_cond_concat(input, cost_reshaped)

            h0 = lrelu(conv2d(x, self.output_depth + 1, name='d_h0_conv'))
            h0 = conv_cond_concat(h0, cost_reshaped)

            h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim + 1, name='d_h1_conv')))
            h1 = tf.reshape(h1, [self.batch_size, -1])
            h1 = concat([h1, cost], 1)

            h2 = lrelu(self.d_bn2(linear(h1, self.dfc_dim, 'd_h2_lin')))
            h2 = concat([h2, cost], 1)

            h3 = linear(h2, 1, 'd_h3_lin')

            return tf.nn.sigmoid(h3), h3

    def sampler(self, z, cost=None, name="generator", reuse=True):
        return self.generator(z, cost, name, reuse)

    def transmitter(self, x):
            x = tf.reshape(x, shape=[-1, self.input_height, self.input_width, 1])
            h0 = lrelu(conv2d_t(x, self.weights['w_t_conv0'],
                                self.biases['b_t_conv0']))
            h1 = lrelu(conv2d_t(h0, self.weights['w_t_conv1'],
                                self.biases['b_t_conv1']))
            h2 = lrelu(conv2d_t(h1, self.weights['w_t_conv2'],
                                self.biases['b_t_conv2']))
            h3 = lrelu(conv2d_t(h2, self.weights['w_t_conv3'],
                                self.biases['b_t_conv3']))
            h4 = tf.matmul(tf.reshape(h3, [len(self.data_X), -1]),
                           self.weights['w_t_fc']) + self.biases['b_t_fc']

            return(tf.nn.sigmoid(h4), h4)

    def evaluator(self, sess, pars):
        for par in pars:
            T, T_logits = sess.run(self.transmitter(self.data_X), feed_dict={
                self.weights['w_t_conv0']:
                np.reshape(par[:self.cuts[0]],
                           self.weights['w_t_conv0'].get_shape()),
                self.weights['w_t_conv1']:
                np.reshape(par[self.cuts[0]:self.cuts[1]],
                           self.weights['w_t_conv1'].get_shape()),
                self.weights['w_t_conv2']:
                np.reshape(par[self.cuts[1]:self.cuts[2]],
                           self.weights['w_t_conv2'].get_shape()),
                self.weights['w_t_conv3']:
                np.reshape(par[self.cuts[2]:self.cuts[3]],
                           self.weights['w_t_conv3'].get_shape()),
                self.weights['w_t_fc']:
                np.reshape(par[self.cuts[3]:self.cuts[4]],
                           self.weights['w_t_fc'].get_shape()),
                self.biases['b_t_conv0']:
                np.reshape(par[self.cuts[4]:self.cuts[5]],
                           self.biases['b_t_conv0'].get_shape()),
                self.biases['b_t_conv1']:
                np.reshape(par[self.cuts[5]:self.cuts[6]],
                           self.biases['b_t_conv1'].get_shape()),
                self.biases['b_t_conv2']:
                np.reshape(par[self.cuts[6]:self.cuts[7]],
                           self.biases['b_t_conv2'].get_shape()),
                self.biases['b_t_conv3']:
                np.reshape(par[self.cuts[7]:self.cuts[8]],
                           self.biases['b_t_conv3'].get_shape()),
                self.biases['b_t_fc']:
                np.reshape(par[self.cuts[8]:self.cuts[9]],
                           self.biases['b_t_fc'].get_shape())
            })
            cost = 0  # use the evaluator to calculate the loss

        return(cost)

    def _setup_placeholder(self, k_h=5, k_w=5):
        with tf.variable_scope("transmitter"):
            self.weights = {
                'w_t_conv0': tf.placeholder(
                    tf.float16,
                    [k_h, k_w, self.col_dim, self.df_dim]),
                'w_t_conv1': tf.placeholder(
                    tf.float16,
                    [k_h, k_w, self.df_dim, self.df_dim * 2]),
                'w_t_conv2': tf.placeholder(
                    tf.float16,
                    [k_h, k_w, self.df_dim * 2, self.df_dim * 4]),
                'w_t_conv3': tf.placeholder(
                    tf.float16,
                    [k_h, k_w, self.df_dim * 4, self.df_dim * 8]),
                'w_t_fc': tf.placeholder(tf.float16, [2048, self.dfc_dim])
            }
            self.biases = {
                'b_t_conv0': tf.placeholder(tf.float16, [self.df_dim]),
                'b_t_conv1': tf.placeholder(tf.float16, [self.df_dim * 2]),
                'b_t_conv2': tf.placeholder(tf.float16, [self.df_dim * 4]),
                'b_t_conv3': tf.placeholder(tf.float16, [self.df_dim * 8]),
                'b_t_fc': tf.placeholder(tf.float16, [self.dfc_dim])
            }
        w_cuts = [reduce(mul, self.weights[name].get_shape())
                  for name in ['w_t_conv0', 'w_t_conv1',
                               'w_t_conv2', 'w_t_conv3', 'w_t_fc']]
        b_cuts = [reduce(mul, self.biases[name].get_shape())
                  for name in ['b_t_conv0', 'b_t_conv1',
                               'b_t_conv2', 'b_t_conv3', 'b_t_fc']]
        self.cuts = np.cumsum([w_cuts, b_cuts])
        self.output_height = math.ceil(math.sqrt(float(int(self.cuts[-1])) / int(self.output_depth)))
        if int(self.cuts[-1]) > self.output_height * (self.output_height - 1) * self.output_depth:
            self.output_width = math.ceil(math.sqrt(float(int(self.cuts[-1])) / int(self.output_depth)))
        else:
            self.output_width = math.floor(math.sqrt(float(int(self.cuts[-1])) / int(self.output_depth)))
        input_dims = [self.output_height, self.output_width, self.output_depth]

        self.cost = tf.placeholder(tf.float16, [self.batch_size, 1], name='cost')
        self.z = tf.placeholder(tf.float16, [None, self.z_dim], name='z')
        self.cost_sum = tf.summary.scalar("cost", self.cost)
        self.z_sum = tf.summary.histogram("z", self.z)

        self.inputs = tf.placeholder(
            tf.float16, [self.batch_size] +input_dims, name='real_images')
        self.sample_inputs = tf.placeholder(
            tf.float16, [self.sample_num] + input_dims, name='sample_inputs')

    def load_mnist(self):
        data_dir = os.path.join("./data", self.dataset_name)

        fd = open(os.path.join(data_dir, 'train-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float)

        fd = open(os.path.join(data_dir, 'train-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trY = loaded[8:].reshape((60000)).astype(np.float)

        fd = open(os.path.join(data_dir, 't10k-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teX = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)

        fd = open(os.path.join(data_dir, 't10k-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teY = loaded[8:].reshape((10000)).astype(np.float)

        trY = np.asarray(trY)
        teY = np.asarray(teY)

        X = np.concatenate((trX, teX), axis=0)
        y = np.concatenate((trY, teY), axis=0).astype(np.int)

        seed = 547
        np.random.seed(seed)
        np.random.shuffle(X)
        np.random.seed(seed)
        np.random.shuffle(y)

        y_vec = np.zeros((len(y), 10), dtype=np.float)
        for i, label in enumerate(y):
            y_vec[i, y[i]] = 1.0

        return X / 255., y_vec

    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
            self.dataset_name, self.batch_size,
            self.output_height, self.output_width)

    def save(self, checkpoint_dir, step):
        model_name = "deep-GAN.model-"
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            print(" [*] Success to read {}".format(ckpt_name))
            return True
        else:
            print(" [*] Failed to find a checkpoint")
            return False
