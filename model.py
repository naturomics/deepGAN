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
        image_dims = [self.input_height, self.input_width, self.c_dim]
        self.inputs = tf.placeholder(
            tf.float32, [self.batch_size] + image_dims, name='real_images')

        self.G1 = generator(self.z, self.y, name='G1')
        self.G2 = generator(self.z, self.y, name='G2')
        self.D, self.D_logits = self.discriminator(input, self.y, reuser=False)

        self.samples = self.sampler(self.z, self.y)
        self.D_, self.D_logits_ =  self.discriminator(self.G, self.y, reuse=True)

    def train(self, config):
        data_X, data_y = self.load_mnist()

        d_optim = tf.train.AdamOptimizer(config.learning_rate, betal=config.betal).minimize(self.d_loss, var_list=self.d_vars)
        g1_optim = tf.train.AdamOptimizer(config.learning_rate, betal=config.betal).minimize(self.g1_loss, var_list=self.g1_vars)
        g2_optim = tf.train.AdamOptimizer(config.learning_rate, betal=config.betal).minimize(self.g2_loss, var_list=self.g2_vars)

        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()

        sample_z = np.random.uniform(-1, 1, size=(self.sample_num, self.z_dim))

        sample_inputs = data_X[0:self.sample_num]
        sample_inputs = data_X[0:self.sample_num]

        counter = 1
        start_time = time.time()

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        for epoch in xrange(config.epoch):
            batch_idxs = min(len(data_x), config.train_size) // config.batch_size
            for idx in xrange(0, batch_idxs):
                batch_images = data_X[idx*config.batch_size:(idx+1)*config.batch_size]
                batch_labels = data_y[idx*config.batch_size:(idx+1)*config.batch_size]
                batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]).astype(np.float32)


        def discriminator(self,):
            with tf.variable_scope("discriminator") as scpoe:
                pass
