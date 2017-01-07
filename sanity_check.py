from __future__ import print_function

import numpy as np
import tensorflow as tf
from scipy import ndimage
import matplotlib.pyplot as plt

from keras.preprocessing.image import load_img, img_to_array, array_to_img
from spatial_transformer import spatial_transformer_network

# load 2 cat images
dims = (600, 600)
image_path = './data/cat1.jpg'
img = load_img(image_path, target_size=dims)
img1 = img_to_array(img)
img1 = np.expand_dims(img1, axis=0)
img1 = img1 / 255.0
image_path = './data/cat2.jpg'
img = load_img(image_path, target_size=dims)
img2 = img_to_array(img)
img2 = np.expand_dims(img2, axis=0)
img2 = img2 / 255.0

# concat into 1 vector
input_img = np.concatenate([img1, img2], axis=0)

# placeholders
x = tf.placeholder(tf.float32, [None, 600, 600, 3])
# x = tf.cast(input_img, 'float32')

# Create localisation network and convolutional layer
with tf.variable_scope('spatial_transformer_0'):

	# Create a fully-connected layer with 6 output nodes
	n_fc = 6
	W_fc1 = tf.Variable(tf.zeros([600 * 600 * 3, n_fc]), name='W_fc1')

	# Zoom into the image
	initial = np.array([[1., 0, 0], [0, 1., 0]])
	initial = initial.astype('float32')
	initial = initial.flatten()

	b_fc1 = tf.Variable(initial_value=initial, name='b_fc1')
	h_fc1 = tf.matmul(tf.zeros([2, 600 * 600 * 3]), W_fc1) + b_fc1
	h_trans = spatial_transformer_network(x, h_fc1)

# %% Run session
sess = tf.Session()
sess.run(tf.global_variables_initializer())
y = sess.run(h_trans, feed_dict={x: input_img})
y = np.reshape(y, (2, 600, 600, 3))
plt.imshow(y[1])
plt.show()