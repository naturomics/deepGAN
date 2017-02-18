from __future__ import division
import os
import time
import numpy as np
import tensorflow as tf

from ops import *
from utils import *


class deepGAN(object):
    def __init__():
        pass

    class generator:
        def __init__(self, z, y, name='generator'):
            pass

    def build_model(self):
        self.y = tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='y')
        image_dims = [self.input_height, self.input_width, self.c_dim]
        self.inputs = tf.placeholder(
        self.sample_inputs = tf.placeholder(
            tf.float32, [self.sample_num] + image_dims, name='sample_inputs')

        #inputs = self.inputs
        #sample_inputs = self.sample_inputs

        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
        self.z_sum = histogram_summary("z", self.z)

        self.G1 = deepGAN.generator(self.z, self.y, name='G1')
        self.G2 = deepGAN.generator(self.z, self.y, name='G2')
        self.D, self.D_logits = self.discriminator(input, self.y, reuser=False)

        self.sampler = self.sampler(self.z, self.y)
        self.D1_, self.D1_logits_ =  self.discriminator(self.G1, self.y, reuse=True)
        self.D2_, self.D2_logits_ =  self.discriminator(self.G2, self.y, reuse=True)

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

        self.d_loss_real_sum =  scalar_summary("d_loss_real",  self.d_loss_real)
        self.d_loss_g1asFake_sum =  scalar_summary("d_loss_g1asFake",  self.d_loss_g1asFake)
        self.d_loss_g2asFake_sum =  scalar_summary("d_loss_g2asFake",  self.d_loss_g2asFake)
        self.d1_loss_sum =  scalar_summary("d1_loss",  self.d1_loss)
        self.d2_loss_sum =  scalar_summary("d2_loss",  self.d2_loss)
        self.g1_loss_sum =  scalar_summary("g1_loss",  self.g1_loss)
        self.g2_loss_sum =  scalar_summary("g2_loss",  self.g2_loss)

        t_vars = tf.trainable_variables()

        self.d1_vars = [var for var in t_vars if 'd1_' in var.name]
        self.g1_vars = [var for var in t_vars if 'g1_' in var.name]
        self.d2_vars = [var for var in t_vars if 'd2_' in var.name]
        self.g2_vars = [var for var in t_vars if 'g2_' in var.name]

        self.saver = tf.train.Saver()

    def train(self, config):
        data_X, data_y = self.load_mnist()

        d1_optim = tf.train.AdamOptimizer(config.learning_rate,
                                          betal=config.betal).minimize(self.d1_loss, var_list=self.d1_vars)
        d2_optim = tf.train.AdamOptimizer(config.learning_rate,
                                          betal=config.betal).minimize(self.d2_loss, var_list=self.d2_vars)
        g1_optim = tf.train.AdamOptimizer(config.learning_rate,
                                          betal=config.betal).minimize(self.g1_loss, var_list=self.g1_vars)
        g2_optim = tf.train.AdamOptimizer(config.learning_rate,
                                          betal=config.betal).minimize(self.g2_loss, var_list=self.g2_vars)

        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()

        self.g1_sum = tf.summary.merge([
            self.z_sum, self.d1__sum,self.G1_sum,
            self.d_loss_g1asFake_sum, g1_loss_sum])
        self.d1_sum = tf.summary.merge([
            self.z_sum, self.d_sum,
            self.d_loss_real_sum,
            self.d1_loss_sum])
        self.g2_sum = tf.summary.merge([
            self.z_sum, self.d2__sum,self.G2_sum,
            self.d_loss_g2asFake_sum, g2_loss_sum])
        self.d2_sum = tf.summary.merge([
            self.z_sum, self.d_sum,
            self.d_loss_real_sum,
            self.d2_loss_sum])
        self.writer = SummaryWriter("./logs", self.sess.graph)

        sample_z = np.random.uniform(-1, 1, size=(self.sample_num, self.z_dim))

        sample_inputs = data_X[0:self.sample_num]
        sample_inputs = data_X[0:self.sample_num]

        counter = 1
        start_time = time.time()

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        optims = [[d1_optim, g1_optim, d1_loss, g1_loss], [d2_optim, g2_optim, d2_loss, g2_loss]]
        for epoch in xrange(config.epoch):
            batch_idxs = min(len(data_x), config.train_size) // config.batch_size
            for idx in xrange(0, batch_idxs):
                batch_images = data_X[idx*config.batch_size:(idx+1)*config.batch_size]
                batch_labels = data_y[idx*config.batch_size:(idx+1)*config.batch_size]
                batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]).astype(np.float32)

                ## the key algorithms for deep GAN ##
                generator = 1
                for d_optim, g_optim, d_loss, g_loss in optims:
                    # Update D network
                    _, summary_str = self.sess.run(d_optim, feed_dict={
                        self.inputs: batch_images,
                        self.z: batch_z,
                        self.y: batch_labels,
                    })
                    self.writer.add_summary(summary_str, counter)

                    # Update G network
                    _, summary_str = self.sess.run(g_optim, feed_dict={
                        self.z: batch_z,
                        self.y: batch_labels,
                    })
                    self.writer.add_summary(summary_str, counter)

                    # Run g_optim twice
                    _, summary_str = self.sess.run(g_optim, feed_dict={
                        self.z: batch_z,
                        self.y: batch_labels,
                    })
                    self.writer.add_summary(summary_str, counter)

                    errD_fake = self.d_loss_fake.eval({
                        self.z: batch_z,
                        self.y: batch_labels
                    })
                    errD_real = self.d_loss_real.eval({
                        self.inputs: batch_images,
                        self.y: batch_labels
                    })
                    errG = self.g_loss.eval({
                        self.z: batch_z,
                        self.y: batch_labels
                    })

                    counter += 1
                    print("Epoch(generator %1d): [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f"
                        % (generator, epoch, idx, batch_idxs,
                            time.time() - start_time, errD_fake + errD_real, errG))
                    generator += 1

                    if np.mod(counter, 100) == 1:
                        samples , d_loss, g_loss = self.sess.run(
                            [self.sampler, d_loss, g_loss],
                            feed_dict={
                                self.z: sample_z,
                                self.inputs: sample_inputs,
                                self.y: sample_labels,
                            }
                        )
                        save_images(samples, [8, 8],
                                    './{}/train_g{:02d}_{:02d}_{:04d}.png'.format(
                                        config.sample_dir, generator, epoch, idx
                                    ))
                        print("[Sampled] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))

                    if np.mod(counter, 500) == 2:
                        self.save(config.checkpoint_dir,counter)

        def discriminator(self,):
            with tf.variable_scope("discriminator") as scpoe:
                pass
