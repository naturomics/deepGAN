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
    def __init__(self, sess, batch_size=16, z_dim=100, conv='conv2d',
                 gf_dim=16, df_dim=16, tf_dim=16, tfc_dim=512, gfc_dim=512, dfc_dim=512, col_dim=1,
                 dataset_name='default', input_fname_pattern='*.jpg', output_channel=3,
                 T_input_height=28, T_input_width=28, n_classes=10,
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
        self.n_classes = n_classes
        self.T_input_height = T_input_height
        self.T_input_width = T_input_width
        self.z_dim = z_dim
        self.multiple = 8
        self.cycles = 200

        self.gf_dim = gf_dim
        self.df_dim = df_dim
        self.tf_dim = tf_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim
        self.tfc_dim = tfc_dim

        self.col_dim = col_dim
        self.output_channel = output_channel
        self.conv = conv

        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')

        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')
        self.g_bn2_1 = batch_norm(name='g_bn2_1')
        self.g_bn2_2 = batch_norm(name='g_bn2_2')

        self.dataset_name = dataset_name
        self.input_fname_pattern = input_fname_pattern
        self.checkpoint_dir = checkpoint_dir
        self.build_model()

    def build_model(self):
        self._setup_placeholder()
        inputs = self.inputs
        sample_inputs = self.sample_inputs

        print("\ndefining generator...")
        self.G = self.generator(name='g')
        print("\n\ndefining discriminator...")
        print(self.inputs)
        self.D, self.D_logits = self.discriminator()
        print("G & D defined!\n")

        print("reusing G & D...")
        self.sampler = self.generator(name='g', reuse=True)
        self.D_, self.D_logits_ = self.discriminator(reuse=True)
        print("reused !\n")

        self.D_sum = tf.summary.histogram("D", self.D)
        self.G_sum = tf.summary.image("G", self.G)
        self.d__sum = tf.summary.histogram("D_", self.D_)

        self.d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.D_logits, labels=tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.D_logits_, labels=tf.zeros_like(self.D_)))

        # self.d_loss = tf.reduce_mean(self.D_logits - self.D_logits_)
        self.d_loss = self.d_loss_real + self.d_loss_fake
        # self.g_loss = self.d_loss_fake
        self.g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.D_logits_, labels=tf.ones_like(self.D_)))

        self.d_loss_real_sum = tf.summary.scalar("d_loss_real",
                                                 self.d_loss_real)
        self.d_loss_fake_sum = tf.summary.scalar("d_loss_fake",
                                                 self.d_loss_fake)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver(t_vars)

    def train(self, config):
        self.data_X, self.data_y = self.load_mnist()
        self.data_X = tf.constant(self.data_X[:30000], tf.float32)
        self.data_y = tf.constant(self.data_y[:30000], tf.float32)

        print("\ngenerating init pars: ")
        #data_pars = np.float32(
            #np.random.normal(0.001, 0.05, (self.batch_size*self.multiple, int(self.cuts[-1]))))
        data_pars = tf.constant(tf.random_normal(
            self.batch_size * self.multiple, int(self.cuts[-1])))
        padding = np.zeros((self.batch_size*self.multiple, self.output_height
                            * self.output_width * self.output_channel
                            - int(self.cuts[-1])), dtype=np.float32)
        data_pars = np.concatenate((data_pars, padding), axis=1)
        print("pars generated! pars dims: "+ str(data_pars.shape))
        data_cost = np.float32(np.random.random(size=(self.batch_size*self.multiple, 1)))

        print("\ndefine optimizer")
        d_optim = tf.train.RMSPropOptimizer(config.learning_rate,
                                         decay=0.5).minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.RMSPropOptimizer(config.learning_rate,
                                         decay=0.5).minimize(self.g_loss, var_list=self.g_vars)
        print("optimizer defined")

        print("\ninitializing global vars...")
        tf.global_variables_initializer().run()
        print("variables initialized")

        self.g_sum = tf.summary.merge([
            self.z_sum, self.G_sum, self.d__sum,
            self.d_loss_fake_sum, self.g_loss_sum, self.cost_sum])
        self.d_sum = tf.summary.merge([
            self.z_sum, self.D_sum,
            self.d_loss_real_sum,
            self.d_loss_sum, self.cost_sum])

        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)


        counter = 1
        start_time = time.time()

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        # use T net with generated pars to calculate true loss,
        # and update new (pars, cost) pair
        print("initializing cost...")
        data_cost[:, ] = self.evaluator(data_pars[:, ])
        for cycle in xrange(config.cycle):

            for epoch in xrange(config.epoch):
                batch_idxs = min(len(data_pars), config.train_size) // self.batch_size
                for idx in xrange(0, batch_idxs):
                    batch_pars = data_pars[idx * self.batch_size:(idx + 1) * self.batch_size]
                    self.batch_pars = np.reshape(batch_pars, ([self.batch_size] + self.input_dims))
                    batch_cost = data_cost[idx * self.batch_size:(idx + 1) * self.batch_size]
                    batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)

                    # Update D network
                    print("\nupdating D network...")
                    _, summary_str = self.sess.run([d_optim, self.d_sum],
                                                    feed_dict={
                                                        self.inputs: self.batch_pars,
                                                        self.z: batch_z,
                                                        self.cost: batch_cost
                                                    })
                    self.writer.add_summary(summary_str, counter)

                    # Update G network
                    print("updating G network...")
                    _, summary_str = self.sess.run([g_optim, self.g_sum], feed_dict={
                        self.z: batch_z,
                        self.cost: batch_cost,
                    })
                    self.writer.add_summary(summary_str, counter)

                    errD = self.d_loss_real.eval({self.inputs: self.batch_pars,self.cost:batch_cost})
                    errG = self.d_loss_fake.eval({self.z: batch_z,self.cost:batch_cost})

                    counter += 1
                    print("\nCycle: [%2d] Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" % (cycle, epoch, idx, batch_idxs, time.time() - start_time, errD, errG))

            if np.mod(cycle, 100) == 10:
                self.save(config.checkpoint_dir, counter)
            # randomly generate pars with expected loss
            max_val = data_cost.min()
            if (max_val - 0.001) < 0:
                min_val = np.random.uniform(0, max_val+1e-10, 1)
            else:
                min_val = max_val - 0.001
            expected_cost = tf.constant(tf.random_uniform((self.batch_size, 1), min_val, max_val))
            sample_z = tf.constant(tf.random_uniform((self.batch_size, self.z_dim), -1, 1))
            print("\ngenerating expected parameters...")
            generated_pars = self.sess.run(self.sampler,
                                           feed_dict={
                                               self.z: sample_z,
                                               self.cost: expected_cost,
                                           })
            save_images(generated_pars, [self.batch_size//2, self.batch_size//2], './{}/autoModeler/train_{:04d}.png'.format(config.sample_dir, cycle))
            generated_pars = np.reshape(generated_pars, (self.batch_size, -1))
            generated_pars[:, int(self.cuts[-1]):] = 0
            np.savetxt('generated_pars.csv', generated_pars, delimiter=',')
            if cycle <= self.cycles:
                #data_pars = np.concatenate((data_pars, generated_pars))
                data_pars = tf.concat([data_pars, generated_pars], 0)
                #data_cost = np.concatenate((data_cost, self.evaluator(data_pars[-self.batch_size:, ])))
                data_cost = tf.concat([data_cost, self.evaluator(data_pars[-self.batch_size:, ])], 0)
            else:
                start = (cycle + self.multiple - 1) % self.cycles * self.batch_size
                end = start + self.batch_size
                data_pars[start:end, ] = generated_pars
                data_cost[start:end, ] = self.evaluator(data_pars[start:end, ])

            print("\ncycle [%2d] done" % cycle)

    def generator(self, name='generator', reuse=False):
            with tf.variable_scope(name) as scope:
                z = self.z
                cost = self.cost
                if reuse:
                    scope.reuse_variables()

                if self.conv == 'conv2d':
                    s_h, s_w = self.output_height, self.output_width
                    s_h2, s_h4, s_h8, s_h16 = math.ceil(s_h / 2), math.ceil(s_h / 4), math.ceil(s_h / 8), math.ceil(s_h / 16)
                    s_w2, s_w4, s_w8, s_w16 = math.ceil(s_w / 2), math.ceil(s_w / 4), math.ceil(s_w / 8), math.ceil(s_w / 16)


                    cost_reshaped = tf.reshape(cost, [self.batch_size, 1, 1, 1])
                    z = concat([z, cost], 1)
                    self.g_z = z

                    #h0 = tf.nn.relu(
                        #self.g_bn0(linear(z, self.gfc_dim, name+'_h0_lin')))
                    self.g_h0, self.w, self.bias = linear(z, self.gfc_dim, name+'_h0_lin', with_w=True)
                    h0 = tf.nn.relu(self.g_bn0(self.g_h0))
                    h0 = concat([h0, cost], 1)

                    h1 = tf.nn.relu(self.g_bn1(
                        linear(h0, self.gf_dim * 2 * s_h16 * s_w16, name+'_h1_lin')))
                    h1 = tf.reshape(h1, [self.batch_size, s_h16, s_w16, self.gf_dim * 2])
                    h1 = conv_cond_concat(h1, cost_reshaped, conv=self.conv)

                    h2_1 = tf.nn.relu(self.g_bn2_1(deconv2d(h1,
                                                        [self.batch_size, s_h8, s_w8, self.gf_dim * 2], name=name+'_h2_1')))
                    h2_1 = conv_cond_concat(h2_1, cost_reshaped, conv=self.conv)

                    h2_2 = tf.nn.relu(self.g_bn2_2(deconv2d(h2_1,
                                                        [self.batch_size, s_h4, s_w4, self.gf_dim * 2], name=name+'_h2_2')))
                    h2_2 = conv_cond_concat(h2_2, cost_reshaped, conv=self.conv)

                    h2 = tf.nn.relu(self.g_bn2(deconv2d(h2_2,
                                                        [self.batch_size, s_h2, s_w2, self.gf_dim * 2], name=name+'_h2')))
                    h2 = conv_cond_concat(h2, cost_reshaped, conv=self.conv)
                    self.g_h2 = h2

                    return tf.nn.sigmoid(
                        deconv2d(h2, [self.batch_size, s_h, s_w, self.output_channel], name=name+'_h3'))
                elif self.conv == 'conv3d':
                    s_h, s_w, s_d = self.output_height, self.output_width, self.output_depth
                    s_h2, s_h4 = math.ceil(s_h / 2), math.ceil(s_h / 4)
                    s_w2, s_w4 = math.ceil(s_w / 2), math.ceil(s_w / 4)
                    s_d2, s_d4 = math.ceil(s_d / 2), math.ceil(s_d / 4)

                    cost_reshaped = tf.reshape(cost, [self.batch_size, 1, 1, 1, 1])
                    z = concat([z, cost], 1)

                    h0 = tf.nn.relu(
                        self.g_bn0(linear(z, self.gfc_dim, name+'_h0_lin')))
                    h0 = concat([h0, cost], 1)

                    h1 = tf.nn.relu(self.g_bn1(
                        linear(h0, self.gf_dim * 2 * s_h4 * s_w4 * s_d4, name+'_h1_lin')))
                    h1 = tf.reshape(h1, [self.batch_size, s_h4, s_w4, s_h4, self.gf_dim * 2])
                    print("h1 & cost_reshaped dtype(conv3d): " + str(h1.dtype) + str(cost_reshaped.dtype))
                    h1 = conv_cond_concat(h1, cost_reshaped)

                    h2 = tf.nn.relu(self.g_bn2(deconv3d(h1,
                                                        [self.batch_size, s_h2, s_w2, s_d2, self.gf_dim * 2], name=name+'_h2')))
                    h2 = conv_cond_concat(h2, cost_reshaped)

                    return tf.nn.sigmoid(
                        deconv3d(h2, [self.batch_size, s_h, s_w, s_d, self.output_channel], name=name + '_h3'))

    def discriminator(self, reuse=False):
        #print(image)
        #print(isinstance(image, (tf.Tensor)))
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()
                image = self.G
            else:
                image = self.inputs
                self.input = image

            if self.conv == 'conv2d':
                cost_reshaped = tf.reshape(self.cost, [self.batch_size, 1, 1, 1])
                self.cost_reshaped = cost_reshaped
                self.image = image
                x = conv_cond_concat(image, cost_reshaped, conv=self.conv)
                # print("x dims(concated): " + str(x.shape))
                #print("input: " + str(image))
                self.x = x

                h0, self.d_h0w, self.d_h0b = conv2d(x, self.output_channel + 1, name='d_h0_conv', with_w=True)
                # print("\nh0 dims(conv2ded): " + str(h0.get_shape()))
                h0 = conv_cond_concat(lrelu(h0), cost_reshaped, conv=self.conv)
                #print("h0 dims(concated): " + str(h0.get_shape()))
                self.h0 = h0

                #h0_1, self.d_h01w, self.d_h01b = conv2d(h0, self.output_channel + 1, name='d_h0_1_conv', with_w=True)
                #h0_1 = conv_cond_concat(lrelu(h0_1), cost_reshaped, conv=self.conv)

                #h0_2, self.d_h02w, self.d_h02b = conv2d(h0_1, self.output_channel + 1, name='d_h0_2_conv', with_w=True)
                #h0_2 = conv_cond_concat(lrelu(h0_2), cost_reshaped, conv=self.conv)

                h1, self.d_h1w, self.d_h1b = conv2d(h0, self.df_dim + 1, name='d_h1_conv', with_w=True)
                #print("\nh1 dims(conv2ded): " + str(h1.get_shape()))
                h1 = tf.reshape(lrelu(self.d_bn1(h1)), [self.batch_size, -1])
                #print("\nh1 dims(reshaped): " + str(h1.get_shape()))
                h1 = concat([h1, self.cost], 1)
                #print("h1 dims(concated): " + str(h1.get_shape()))

                h2, self.d_h2w, self.d_h2b = linear(h1, self.dfc_dim, 'd_h2_lin', with_w=True)
                #print("\nh2 dims(lin): " + str(h2.get_shape()))
                h2 = concat([lrelu(self.d_bn2(h2)), self.cost], 1)
                #print("h2 dims(concated): " + str(h2.get_shape()))

                h3, self.d_h3w, self.d_h3b = linear(h2, 1, 'd_h3_lin', with_w=True)
                #print("\nh3 dims(lin): " + str(h3.get_shape()))
                #print("\nh3: " + str(h3))

                return tf.nn.sigmoid(h3), h3
            elif self.conv == 'conv3d':
                cost_reshaped = tf.reshape(self.cost, [self.batch_size, 1, 1, 1, 1])
                x = conv_cond_concat(input, cost_reshaped)

                h0 = lrelu(conv3d(x, self.output_channel + 1, name='d_h0_conv'))
                h0 = conv_cond_concat(h0, cost_reshaped)

                h1 = lrelu(self.d_bn1(conv3d(h0, self.df_dim + 1, name='d_h1_conv')))
                h1 = tf.reshape(h1, [self.batch_size, -1])
                h1 = concat([h1, self.cost], 1)

                h2 = lrelu(self.d_bn2(linear(h1, self.dfc_dim, 'd_h2_lin')))
                h2 = concat([h2, self.cost], 1)

                h3 = linear(h2, 1, 'd_h3_lin')

                return tf.nn.sigmoid(h3), h3

    def sampler(self, name="generator", reuse=True):
        print("\nrunning sampler...")
        return self.generator(name=name, reuse=reuse)

    def transmitter(self, x):
            x = tf.reshape(x, shape=[-1, self.T_input_height, self.T_input_width, 1])
            h0 = lrelu(conv2d_t(x, self.weights['w_t_conv0'],
                                self.biases['b_t_conv0']))
            h1 = lrelu(conv2d_t(h0, self.weights['w_t_conv1'],
                                self.biases['b_t_conv1']))
            h2 = lrelu(conv2d_t(h1, self.weights['w_t_conv2'],
                                self.biases['b_t_conv2']))
            h3 = lrelu(conv2d_t(h2, self.weights['w_t_conv3'],
                                self.biases['b_t_conv3']))
            h4 = tf.matmul(tf.reshape(h3, [self.data_X.get_shape().as_list()[0], -1]),
                           self.weights['w_t_fc']) + self.biases['b_t_fc']
            h5 = tf.matmul(tf.reshape(h4, [self.data_X.get_shape().as_list()[0], -1]),
                           self.weights['w_t_out']) + self.biases['b_t_out']

            return(tf.nn.softmax(h5), h5)

    def evaluator(self, pars):
        cost = []
        for par in pars:
            #print("parameter :")
            #print(par)
            T, T_logits = self.sess.run(self.transmitter(self.data_X), feed_dict={
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
                self.weights['w_t_out']:
                np.reshape(par[self.cuts[4]:self.cuts[5]],
                           self.weights['w_t_out'].get_shape()),
                self.biases['b_t_conv0']:
                np.reshape(par[self.cuts[5]:self.cuts[6]],
                           self.biases['b_t_conv0'].get_shape()),
                self.biases['b_t_conv1']:
                np.reshape(par[self.cuts[6]:self.cuts[7]],
                           self.biases['b_t_conv1'].get_shape()),
                self.biases['b_t_conv2']:
                np.reshape(par[self.cuts[7]:self.cuts[8]],
                           self.biases['b_t_conv2'].get_shape()),
                self.biases['b_t_conv3']:
                np.reshape(par[self.cuts[8]:self.cuts[9]],
                           self.biases['b_t_conv3'].get_shape()),
                self.biases['b_t_fc']:
                np.reshape(par[self.cuts[9]:self.cuts[10]],
                           self.biases['b_t_fc'].get_shape()),
                self.biases['b_t_out']:
                np.reshape(par[self.cuts[10]:self.cuts[11]],
                           self.biases['b_t_out'].get_shape())
            })
            # save_images(T, [28, 28], "./rowData1.png")
            # use the evaluator to calculate the loss
            # print(T)
            loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=T_logits, labels=tf.ones_like(self.data_y)))
            #np.savetxt('labels_false.csv', T, delimiter="\t")
            #np.savetxt('labels_true.csv', self.data_y, delimiter="\t")
            correct_prediction = tf.equal(tf.argmax(T, 1), tf.argmax(self.data_y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            loss = 1 - self.sess.run(accuracy)
            #nploss = 1 - np.sum(np.all(T == self.data_y,1))/10000
            print("evaluator loss: " + str(loss))
            #print("np loss: " + str(nploss))
            cost.append(loss)
        cost = np.reshape(cost, (-1, 1))

        return(cost)

    def _setup_placeholder(self, k_h=5, k_w=5):
        with tf.variable_scope("transmitter"):
            self.weights = {
                'w_t_conv0': tf.placeholder(
                    tf.float32,
                    [k_h, k_w, self.col_dim, self.tf_dim]),
                'w_t_conv1': tf.placeholder(
                    tf.float32,
                    [k_h, k_w, self.df_dim, self.tf_dim * 2]),
                'w_t_conv2': tf.placeholder(
                    tf.float32,
                    [k_h, k_w, self.df_dim * 2, self.tf_dim * 4]),
                'w_t_conv3': tf.placeholder(
                    tf.float32,
                    [k_h, k_w, self.df_dim * 4, self.tf_dim * 8]),
                'w_t_fc': tf.placeholder(tf.float32, [self.tfc_dim, self.tfc_dim]),
                'w_t_out': tf.placeholder(tf.float32, [self.tfc_dim, self.n_classes])
            }
            self.biases = {
                'b_t_conv0': tf.placeholder(tf.float32, [self.tf_dim]),
                'b_t_conv1': tf.placeholder(tf.float32, [self.tf_dim * 2]),
                'b_t_conv2': tf.placeholder(tf.float32, [self.tf_dim * 4]),
                'b_t_conv3': tf.placeholder(tf.float32, [self.tf_dim * 8]),
                'b_t_fc': tf.placeholder(tf.float32, [self.tfc_dim]),
                'b_t_out': tf.placeholder(tf.float32, [self.n_classes])
            }
        w_cuts = [reduce(mul, self.weights[name].get_shape())
                  for name in ['w_t_conv0', 'w_t_conv1',
                               'w_t_conv2', 'w_t_conv3', 'w_t_fc', 'w_t_out']]
        b_cuts = [reduce(mul, self.biases[name].get_shape())
                  for name in ['b_t_conv0', 'b_t_conv1',
                               'b_t_conv2', 'b_t_conv3', 'b_t_fc', 'b_t_out']]
        self.cuts = np.cumsum([w_cuts, b_cuts])
        if self.conv == 'conv2d':
            self.output_height = math.ceil(math.sqrt(float(int(self.cuts[-1])) / int(self.output_channel)))
            if int(self.cuts[-1]) > self.output_height * (self.output_height - 1) * self.output_channel:
                self.output_width = math.ceil(math.sqrt(float(int(self.cuts[-1])) / int(self.output_channel)))
            else:
                self.output_width = math.floor(math.sqrt(float(int(self.cuts[-1])) / int(self.output_channel)))
            self.input_dims = [self.output_height, self.output_width, self.output_channel]
        elif self.conv == 'conv3d':
            self.output_height = math.ceil(int(self.cuts[-1]) ** (1/3))
            self.output_width = self.output_height
            self.output_depth = self.output_height
            self.input_dims = [self.output_height, self.output_width, self.output_depth, self.output_channel]

        self.cost = tf.placeholder(tf.float32, [self.batch_size, 1], name='cost')
        self.z = tf.placeholder(tf.float32, [self.batch_size, self.z_dim], name='z')
        self.cost_sum = tf.summary.histogram("cost", self.cost)
        self.z_sum = tf.summary.histogram("z", self.z)

        self.inputs = tf.placeholder(
            tf.float32, [self.batch_size] +self.input_dims, name='real_images')
        self.sample_inputs = tf.placeholder(
            tf.float32, [self.batch_size] + self.input_dims, name='sample_inputs')

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
        model_name = "deep-GAN.model"
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
