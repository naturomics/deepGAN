import math
import numpy as np
import tensorflow as tf

from tensorflow.python.framework import ops

from utils import *


if "concat_v2" in dir(tf):
  def concat(tensors, axis, *args, **kwargs):
    return tf.concat_v2(tensors, axis, *args, **kwargs)
else:
  def concat(tensors, axis, *args, **kwargs):
    return tf.concat(tensors, axis, *args, **kwargs)

class batch_norm(object):
  def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
    with tf.variable_scope(name):
      self.epsilon  = epsilon
      self.momentum = momentum
      self.name = name

  def __call__(self, x, train=True):
    return tf.contrib.layers.batch_norm(x,
                      decay=self.momentum,
                      updates_collections=None,
                      epsilon=self.epsilon,
                      scale=True,
                      is_training=train,
                      scope=self.name)

def conv_cond_concat(x, y, conv="conv3d"):
  """Concatenate conditioning vector on feature map axis."""
  x_shapes = x.get_shape()
  y_shapes = y.get_shape()
  print(x_shapes)
  print(y_shapes)
  if conv == "conv2d":
      return concat([
          x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]], dtype=tf.float16)], 3)
  elif conv == "conv3d":
      return concat([
          x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], x_shapes[3], y_shapes[4]],dtype=tf.float16)], 4)

def conv2d(input_, output_dim,
       k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
       name="conv2d"):
  with tf.variable_scope(name):
    w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim], dtype=tf.float16,
              initializer=tf.truncated_normal_initializer(stddev=stddev))
    print('w dims in conv2d: ' + str(w.get_shape()))
    conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

    biases = tf.get_variable('biases', [output_dim], dtype=tf.float16, initializer=tf.constant_initializer(0.0))
    print('biases dims in conv2d: ' + str(biases.get_shape()))
    conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

    return conv

def conv3d(input_, output_dim, k_h=3, k_w=3, k_d=3,
           d_h=2, d_w=2, d_d=2, stddev=0.02, name="conv3d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, k_d, input_.get_shape()[-1], output_dim],
                            dtype=tf.float16, initializer=tf.truncated_normal_initializer(stddev=stddev))
        print('w dims in conv3d: ' + str(w.get_shape()))
        conv = tf.nn.conv3d(input_, w, strides=[1, d_h, d_w, d_d, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim], dtype=tf.float16, initializer=tf.constant_initializer(0.0))
        return tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

def conv2d_t(input_, weight, bias, d_h=2, d_w=2):
    conv = tf.nn.conv2d(input_, weight,
                        strides=[1, d_h, d_w, 1], padding='SAME')
    conv = tf.reshape(tf.nn.bias_add(conv, bias), conv.get_shape())

    return(conv)

def deconv2d(input_, output_shape,
       k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
       name="deconv2d", with_w=False):
  with tf.variable_scope(name):
    # filter : [height, width, output_channels, in_channels]
    w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]], dtype=tf.float16,
              initializer=tf.random_normal_initializer(stddev=stddev))

    print('w dims in deconv2d: ' + str(w.get_shape()))
    try:
      deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                strides=[1, d_h, d_w, 1])

    # Support for verisons of TensorFlow before 0.7.0
    except AttributeError:
      deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                strides=[1, d_h, d_w, 1])

    biases = tf.get_variable('biases', [output_shape[-1]], dtype=tf.float16, initializer=tf.constant_initializer(0.0))
    print('biases dims in deconv2d: ' + str(biases.get_shape()))
    deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

    if with_w:
      return deconv, w, biases
    else:
      return deconv

def deconv3d(input_, output_shape,
       k_h=3, k_w=3, k_d=3, d_h=2, d_w=2, d_d=2, stddev=0.02,
       name="deconv3d", with_w=False):
  with tf.variable_scope(name):
    # filter : [height, width, depth, output_channels, in_channels]
    w = tf.get_variable('w', [k_h, k_w, k_d, output_shape[-1], input_.get_shape()[-1]], dtype=tf.float16,
              initializer=tf.random_normal_initializer(stddev=stddev))

    deconv = tf.nn.conv3d_transpose(input_, w, output_shape=output_shape,
                                    strides=[1, d_h, d_w, d_d, 1])

    # Support for verisons of TensorFlow before 0.7.0

    biases = tf.get_variable('biases', [output_shape[-1]], dtype=tf.float16, initializer=tf.constant_initializer(0.0))
    deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

    if with_w:
      return deconv, w, biases
    else:
      return deconv

def lrelu(x, leak=0.2, name="lrelu"):
  return tf.maximum(x, leak*x)

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
  shape = input_.get_shape().as_list()

  with tf.variable_scope(scope or "Linear"):
    matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float16,
                 tf.random_normal_initializer(stddev=stddev))
    print("matrix in linear :" + str(matrix.get_shape()))
    bias = tf.get_variable("bias", [output_size], tf.float16,
      initializer=tf.constant_initializer(bias_start))
    print("bias in linear :" + str(bias.get_shape()))
    if with_w:
      return tf.matmul(input_, matrix) + bias, matrix, bias
    else:
      return tf.matmul(input_, matrix) + bias
