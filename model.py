from __future__ import division
import os
import time
import numpy as np
import tensorflow as tf

from ops import *
from utils import *


class deepGAN(object):
    def __init__(self, sess, input_height=108, input_width=108, is_crop=True,
                 batch_size=64, sample_num=64, output_height=64, output_width=64,
                 y_dim=None, z_dim=100, gf_dim=64, df_dim=64,
                 gfc_dim=1024, dfc_dim=1024, c_dim=3, dataset_name='default',
                 input_fname_pattern='*.jpg', checkpoint_dir=None, sample_dir=None):
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
            c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
        """
        self.sess = sess
        self.is_crop = is_crop
        self.is_grayscale = (c_dim == 1)
        self.batch_size = batch_size
        self.sample_num = sample_num

        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width

        self.y_dim = y_dim
        self.z_dim = z_dim

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        self.c_dim = c_dim

        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')

        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')

        self.dataset_name = dataset_name
        self.input_fname_pattern = input_fname_pattern
        self.checkpoint_dir = checkpoint_dir
        self.build_model()

    def generator(self, z, y, name='generator'):
            with tf.variable_scope(name) as scope:
                s_h, s_w = self.output_height, self.output_width
                s_h2, s_h4 = int(s_h / 2), int(s_h / 4)
                s_w2, s_w4 = int(s_w / 2), int(s_w / 4)

                # yb = tf.expand_dims(tf.expand_dims(y, 1),2)
                yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
                z = concat([z, y], 1)

                h0 = tf.nn.relu(
                    self.g_bn0(linear(z, self.gfc_dim, name+'_h0_lin')))
                h0 = concat([h0, y], 1)

                h1 = tf.nn.relu(self.g_bn1(
                    linear(h0, self.gf_dim * 2 * s_h4 * s_w4, name+'_h1_lin')))
                h1 = tf.reshape(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2])
                h1 = conv_cond_concat(h1, yb)

                h2 = tf.nn.relu(self.g_bn2(deconv2d(h1,
                                                    [self.batch_size, s_h2, s_w2, self.gf_dim * 2], name=name+'_h2')))
                h2 = conv_cond_concat(h2, yb)

                return tf.nn.sigmoid(
                    deconv2d(h2, [self.batch_size, s_h, s_w, self.c_dim], name=name+'_h3'))

    def build_model(self):
        if self.y_dim:
            self.y = tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='y')

        if self.is_crop:
            image_dims = [self.output_height, self.output_width, self.c_dim]
        else:
            image_dims = [self.input_height, self.input_width, self.c_dim]

        self.inputs = tf.placeholder(
            tf.float32, [self.batch_size] + image_dims, name='real_images')
        self.sample_inputs = tf.placeholder(
            tf.float32, [self.sample_num] + image_dims, name='sample_inputs')

        inputs = self.inputs
        sample_inputs = self.sample_inputs

        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
        self.z_sum = histogram_summary("z", self.z)

        self.G1 = self.generator(self.z, self.y, name='G1')
        self.G2 = self.generator(self.z, self.y, name='G2')
        self.D, self.D_logits = self.discriminator(inputs, self.y, reuse=False)

        self.g1_sampler = self.sampler(self.z, self.y, name="G1")
        self.g2_sampler = self.sampler(self.z, self.y, name="G2")
        self.D1_, self.D1_logits_ = self.discriminator(self.G1, self.y, reuse=True)
        self.D2_, self.D2_logits_ = self.discriminator(self.G2, self.y, reuse=True)

        self.d_sum = histogram_summary("d", self.D)
        self.d1__sum = histogram_summary("d1_", self.D1_)
        self.d2__sum = histogram_summary("d2_", self.D2_)
        self.G1_sum = image_summary("G1", self.G1)
        self.G2_sum = image_summary("G2", self.G2)

        self.d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.D_logits, labels=tf.ones_like(self.D)))
        self.d_loss_g1asReal = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.D1_logits_, labels=tf.ones_like(self.D1_)))
        self.d_loss_g1asFake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.D1_logits_, labels=tf.zeros_like(self.D1_)))
        self.d_loss_g2asReal = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.D2_logits_, labels=tf.ones_like(self.D2_)))
        self.d_loss_g2asFake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.D2_logits_, labels=tf.zeros_like(self.D2_)))

        self.g1_loss = self.d_loss_g1asReal
        self.g2_loss = self.d_loss_g2asReal

        self.d1_loss = self.d_loss_real + self.d_loss_g1asFake + self.d_loss_g2asReal
        self.d2_loss = self.d_loss_real + self.d_loss_g1asReal + self.d_loss_g2asFake

        self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
        self.d_loss_g1asFake_sum = scalar_summary("d_loss_g1asFake", self.d_loss_g1asFake)
        self.d_loss_g2asFake_sum = scalar_summary("d_loss_g2asFake", self.d_loss_g2asFake)
        self.d1_loss_sum = scalar_summary("d1_loss", self.d1_loss)
        self.d2_loss_sum = scalar_summary("d2_loss", self.d2_loss)
        self.g1_loss_sum = scalar_summary("g1_loss", self.g1_loss)
        self.g2_loss_sum = scalar_summary("g2_loss", self.g2_loss)

        t_vars = tf.trainable_variables()

        self.d1_vars = [var for var in t_vars if 'd_' in var.name]
        self.g1_vars = [var for var in t_vars if 'G1_' in var.name]
        self.d2_vars = [var for var in t_vars if 'd_' in var.name]
        self.g2_vars = [var for var in t_vars if 'G2_' in var.name]

        self.saver = tf.train.Saver()

    def train(self, config):
        data_X, data_y = self.load_mnist()

        d1_optim = tf.train.AdamOptimizer(config.learning_rate,
                                          beta1=config.beta1).minimize(self.d1_loss, var_list=self.d1_vars)
        d2_optim = tf.train.AdamOptimizer(config.learning_rate,
                                          beta1=config.beta1).minimize(self.d2_loss, var_list=self.d2_vars)
        g1_optim = tf.train.AdamOptimizer(config.learning_rate,
                                          beta1=config.beta1).minimize(self.g1_loss, var_list=self.g1_vars)
        g2_optim = tf.train.AdamOptimizer(config.learning_rate,
                                          beta1=config.beta1).minimize(self.g2_loss, var_list=self.g2_vars)

        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()

        self.g1_sum = tf.summary.merge([
            self.z_sum, self.d1__sum, self.G1_sum,
            self.d_loss_g1asFake_sum, self.g1_loss_sum])
        self.d1_sum = tf.summary.merge([
            self.z_sum, self.d_sum,
            self.d_loss_real_sum,
            self.d1_loss_sum])
        self.g2_sum = tf.summary.merge([
            self.z_sum, self.d2__sum, self.G2_sum,
            self.d_loss_g2asFake_sum, self.g2_loss_sum])
        self.d2_sum = tf.summary.merge([
            self.z_sum, self.d_sum,
            self.d_loss_real_sum,
            self.d2_loss_sum])

        self.theta = 0.4

        self.writer = SummaryWriter("./logs/theta" + str(self.theta).replace('.', ''), self.sess.graph)

        sample_z = np.random.uniform(-1, 1, size=(self.sample_num, self.z_dim))

        sample_inputs = data_X[0:self.sample_num]
        sample_labels = data_y[0:self.sample_num]

        counter = 1
        start_time = time.time()

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        generators = [[d1_optim, g1_optim, self.d1_loss,
                       self.g1_loss, self.d1_sum,
                       self.g1_sum, self.g1_sampler],
                      [d2_optim, g2_optim, self.d2_loss,
                       self.g2_loss, self.d2_sum,
                       self.g2_sum, self.g2_sampler]]

        avg_errG = [100, 100]

        for epoch in xrange(config.epoch):
            batch_idxs = min(len(data_X), config.train_size) // config.batch_size
            for idx in xrange(0, batch_idxs):
                batch_images = data_X[idx * config.batch_size:(idx + 1) * config.batch_size]
                batch_labels = data_y[idx * config.batch_size:(idx + 1) * config.batch_size]
                batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]).astype(np.float32)

                ## the key algorithms for deep GAN ##
                generator = 1
                for d_optim, g_optim, d_loss, g_loss, d_sum, g_sum, sampler in generators:
                    # make the better generator as a guide of the worse
                    # generator, until the bad guy perform better than the former
                    err_g = g_loss.eval({
                        self.z: batch_z,
                        self.y: batch_labels
                    })
                    avg_errG[generator-1] = self.theta * avg_errG[generator-1] + (1 - self.theta) * err_g
                    if avg_errG[generator-1] < avg_errG[generator%2]:
                        # errG = err_g
                        print("skip for generator {} in the epoch {}, {} batch".format(generator, epoch, idx))
                        generator += 1
                        continue

                    # Update D network
                    _, summary_str = self.sess.run([d_optim, d_sum],
                                                   feed_dict={
                                                       self.inputs: batch_images,
                                                       self.z: batch_z,
                                                       self.y: batch_labels
                                                   })
                    self.writer.add_summary(summary_str, counter)

                    # Update G network
                    _, summary_str = self.sess.run([g_optim, g_sum], feed_dict={
                        self.z: batch_z,
                        self.y: batch_labels,
                    })
                    self.writer.add_summary(summary_str, counter)

                    # Run g_optim twice
                    _, summary_str = self.sess.run([g_optim, g_sum], feed_dict={
                        self.z: batch_z,
                        self.y: batch_labels,
                    })
                    self.writer.add_summary(summary_str, counter)

#                    errD_g1asFake = self.d_loss_g1asFake.eval({
#                        self.z: batch_z,
#                        self.y: batch_labels
#                    })
#                    errD_g2asFake = self.d_loss_g2asFake.eval({
#                        self.z: batch_z,
#                        self.y: batch_labels
#                    })
                    errD = d_loss.eval({
                        self.inputs: batch_images,
                        self.z: batch_z,
                        self.y: batch_labels
                    })

                    counter += 1
                    print("Epoch(generator %1d): [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f, errG1:%.8f, errG2:%.8f"
                          % (generator, epoch, idx, batch_idxs,
                             time.time() - start_time, errD, err_g, avg_errG[0], avg_errG[1]))

                    if np.mod(counter, 100) == 1:
                        samples, d_loss, g_loss = self.sess.run(
                            [sampler, d_loss, g_loss],
                            feed_dict={
                                self.z: sample_z,
                                self.inputs: sample_inputs,
                                self.y: sample_labels,
                            }
                        )
                        save_images(samples, [8, 8],
                                    './{}/train_generator{:01d}_{:02d}_{:04d}.png'.format(
                                        config.sample_dir, generator, epoch, idx
                                    ))
                        print("[Sampled] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))

                    if np.mod(counter, 500) == 2:
                        self.save(config.checkpoint_dir, counter)

                    generator += 1

    def discriminator(self, image, y=None, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
            x = conv_cond_concat(image, yb)

            h0 = lrelu(conv2d(x, self.c_dim + self.y_dim, name='d_h0_conv'))
            h0 = conv_cond_concat(h0, yb)

            h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim + self.y_dim, name='d_h1_conv')))
            h1 = tf.reshape(h1, [self.batch_size, -1])
            h1 = concat([h1, y], 1)

            h2 = lrelu(self.d_bn2(linear(h1, self.dfc_dim, 'd_h2_lin')))
            h2 = concat([h2, y], 1)

            h3 = linear(h2, 1, 'd_h3_lin')

            return tf.nn.sigmoid(h3), h3

    def sampler(self, z, y=None, name="generator"):
        with tf.variable_scope(name) as scope:
            scope.reuse_variables()

            s_h, s_w = self.output_height, self.output_width
            s_h2, s_h4 = int(s_h / 2), int(s_h / 4)
            s_w2, s_w4 = int(s_w / 2), int(s_w / 4)

            yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
            z = concat([z, y], 1)

            h0 = tf.nn.relu(self.g_bn0(linear(z, self.gfc_dim, name+'_h0_lin')))
            h0 = concat([h0, y], 1)

            h1 = tf.nn.relu(self.g_bn1(linear(h0, self.gf_dim * 2 * s_h4 * s_w4, name+'_h1_lin'), train=False))
            h1 = tf.reshape(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2])
            h1 = conv_cond_concat(h1, yb)

            h2 = tf.nn.relu(self.g_bn2(deconv2d(h1, [self.batch_size, s_h2, s_w2, self.gf_dim * 2], name=name+'_h2'), train=False))
            h2 = conv_cond_concat(h2, yb)

            return tf.nn.sigmoid(deconv2d(h2, [self.batch_size, s_h, s_w, self.c_dim], name=name+'_h3'))

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

        y_vec = np.zeros((len(y), self.y_dim), dtype=np.float)
        for i, label in enumerate(y):
            y_vec[i, y[i]] = 1.0

        return X / 255., y_vec

    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
            self.dataset_name, self.batch_size,
            self.output_height, self.output_width)

    def save(self, checkpoint_dir, step):
        model_name = "deep-GAN.model-theta" + str(self.theta).replace('.', '')
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
