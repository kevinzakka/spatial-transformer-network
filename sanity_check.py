from __future__ import print_function

import numpy as np
import tensorflow as tf
from spatial_transformer import spatial_transformer_network
from keras.preprocessing.image import load_img, img_to_array, array_to_img

DIMS = (600, 600)
CAT1 = 'cat1.jpg'
CAT2 = 'cat2.jpg'
CAT3 = 'cat3.jpg'
CAT4 = 'cat4.jpg'

def load_data(dims, img_name, view=False):
	"""
	Util function for processing RGB image into 4D tensor.

	Returns tensor of shape (1, H, W, C)
	"""
	image_path = './data/' + img_name
	img = load_img(image_path, target_size=dims)
	if view:
		img.show()
	img = img_to_array(img)
	img = np.expand_dims(img, axis=0)
	img = img / 255.0
	return img

def main():
	# out dims
	out_H = 400
	out_W = 400
	out_dims = (out_H, out_W)

	# load 4 cat images
	img1 = load_data(DIMS, CAT1, view=True)
	img2 = load_data(DIMS, CAT2)
	img3 = load_data(DIMS, CAT3)
	img4 = load_data(DIMS, CAT4)

	# concat into tensor of shape (2, 600, 600, 3)
	input_img = np.concatenate([img1, img2, img3, img4], axis=0)

	# dimension sanity check
	print("Input Img Shape: {}".format(input_img.shape))

	# grab shape
	B, H, W, C = input_img.shape

	# placeholders
	x = tf.placeholder(tf.float32, [None, H, W, C])

	# Create localisation network and convolutional layer
	with tf.variable_scope('spatial_transformer_0'):

		# Create a fully-connected layer with 6 output nodes
		n_fc = 6
		W_fc1 = tf.Variable(tf.zeros([H*W*C, n_fc]), name='W_fc1')

		# identity transform
		theta = np.array([[1., 0, 0], [0, 1., 0]])
		theta = theta.astype('float32')
		theta = theta.flatten()

		b_fc1 = tf.Variable(initial_value=theta, name='b_fc1')
		h_fc1 = tf.matmul(tf.zeros([B, H*W*C]), W_fc1) + b_fc1
		h_trans = spatial_transformer_network(x, h_fc1, out_dims)

	# run session
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	y = sess.run(h_trans, feed_dict={x: input_img})
	print(y.shape)
	y = np.reshape(y, (B, out_H, out_W, C))
	y = array_to_img(y[0])
	y.show()

if __name__ == '__main__':
	main()