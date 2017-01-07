from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from vis_utils import view_images

# ensuring reproducibility
np.random.seed(42)

# global params
DIM = 60
NUM_CLASSES = 10
mnist_cluttered = "./data/mnist_cluttered_60x60_6distortions.npz"

def load_data(view_grid=False):
	data = np.load(mnist_cluttered)

	X_train, y_train = data['x_train'], np.argmax(data['y_train'], axis=-1)
	X_valid, y_valid = data['x_valid'], np.argmax(data['y_valid'], axis=-1)
	X_test, y_test = data['x_test'], np.argmax(data['y_test'], axis=-1)

	# reshape for convolutions
	X_train = X_train.reshape((X_train.shape[0], 1, DIM, DIM))
	X_valid = X_valid.reshape((X_valid.shape[0], 1, DIM, DIM))
	X_test = X_test.reshape((X_test.shape[0], 1, DIM, DIM))

	if view_grid:
		# let's view a grid of the images
		mask = np.arange(100)
		sample = X_train.squeeze()[mask]
		view_images(sample)

	y_train = np_utils.to_categorical(y_train, NUM_CLASSES)
	y_valid = np_utils.to_categorical(y_valid, NUM_CLASSES)
	y_test = np_utils.to_categorical(y_test, NUM_CLASSES)

	print("Train samples:", X_train.shape)
	print("Validation samples:", X_valid.shape)
	print("Test samples:", X_test.shape)

	return X_train, y_train, X_test, y_test, X_test, y_test

def main():

	X_train, y_train, X_test, y_test, X_test, y_test = load_data(view_grid=True)

	# initialize theta to identity transform
	theta = np.array([[1., 0., 0.], [0., 1., 0.]])

	# repeat num_batch times
	theta = np.resize(theta, (num_batch, 2, 3))

	# theta = K.variable(theta)

	height, width = img1.shape[1], img1.shape[2]

	# get grids
	batch_grids = affine_grid_generator(height, width, theta)

	x_s = batch_grids[:, :, :, 0:1].squeeze()
	y_s = batch_grids[:, :, :, 1:2].squeeze()

	out = bilinear_sampler(input_img, x_s, y_s)


if __name__ == '__main__':
	main()