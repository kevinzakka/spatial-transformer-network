import numpy as np
import tensorflow as tf

def init_weights(name, shape):
	"""
	Handy helper function for initializing the weights of a layer.

	Performs He. et al. initilization as described in [1].

	References
	----------
	[1] - https://arxiv.org/abs/1502.01852
	"""
	init = tf.contrib.layers.variance_scaling_initializer()
	W = tf.get_variable(name, shape, tf.float32, init)
	return W

def init_bias(name, shape, trans=False):
	"""
	Handy helper function for initializing the biases of a layer.

	Performs zero bias initialization.
	"""
	init = tf.zeros_initializer
	b = tf.get_variable(name, shape, tf.float32, init)

	if trans:
		x = np.array([[1., 0, 0], [0, 1., 0]])
		x = x.astype('float32').flatten()
		b = tf.Variable(initial_value=x)

	return b

def Conv2D(input_tensor, input_shape, filter_size, num_filters, strides=1, name=None):
	"""
	Handy helper function for convnets.

	Performs 2D convolution with a default stride of 1. The kernel has shape
	filter_size x filter_size with num_filters output filters.
	"""
	shape = [filter_size, filter_size, input_shape, num_filters]

	# initialize weights and biases of the convolution
	W = init_weights(name=name+'_W' , shape=shape)
	b = init_bias(name=name+'_b', shape=shape[-1])

	conv = tf.nn.conv2d(input_tensor, W, strides=[1, strides, strides, 1], padding='SAME', name=name)
	conv = tf.nn.bias_add(conv, b)
	return conv

def MaxPooling2D(input_tensor, k=2, use_relu=False, name=None):
	"""
	Handy wrapper function for convolutional networks.

	Performs 2D max pool with a default stride of 2.
	"""
	pool = tf.nn.max_pool(input_tensor, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME', name=name)

	if use_relu:
		pool = tf.nn.relu(pool)

	return pool

def BatchNormalization(input_tensor, phase, use_relu=False, name=None):
	"""
	Handy wrapper function for convolutional networks.

	Performs batch normalization on the input tensor.
	"""
	normed = tf.contrib.layers.batch_norm(input_tensor, center=True, scale=True, is_training=phase, scope=name)
	
	if use_relu:
		normed = tf.nn.relu(normed)
		
	return normed

def Flatten(layer):
	"""
	Handy function for flattening the result of a conv2D or
	maxpool2D to be used for a fully-connected (affine) layer.
	"""
	layer_shape = layer.get_shape()
	# num_features = tf.reduce_prod(tf.shape(layer)[1:])
	num_features = layer_shape[1:].num_elements()
	layer_flat = tf.reshape(layer, [-1, num_features])

	return layer_flat, num_features

def Dense(input_tensor, num_inputs, num_outputs, use_relu=True, trans=False, name=None):
	"""
	Handy wrapper function for convolutional networks.

	Performs an affine layer (fully-connected) on the input tensor.
	"""
	shape = [num_inputs, num_outputs]

	# initialize weights and biases of the affine layer
	W = init_weights(name=name+'_W' ,shape=shape)
	b = init_bias(name=name+'_b', shape=shape[-1], trans=trans)

	fc = tf.matmul(input_tensor, W, name=name) + b

	if use_relu:
		fc = tf.nn.relu(fc)

	return fc

def theta_bias(name):
	with tf.variable_scope(name):
		x = np.array([[1., 0, 0], [0, 1., 0]])
		x = x.astype('float32').flatten()
		return tf.Variable(initial_value=x)
