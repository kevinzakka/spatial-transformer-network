# loading packages
import numpy as np
import matplotlib.pyplot as plt

def visualize_grid(Xs, ubound=255.0, padding=1):
	"""
	Reshape a 4D tensor of image data to a grid for easy visualization.

	Inputs:
	- Xs: Data of shape (N, H, W, C)
	- ubound: Output grid will have values scaled to the range [0, ubound]
	- padding: The number of blank pixels between elements of the grid

	Returns:
	- grid

	References:
	- Adapted from CS231n - http://cs231n.github.io/
	"""

	(N, H, W, C) = Xs.shape
	grid_size = int(np.ceil(np.sqrt(N)))
	grid_height = H * grid_size + padding * (grid_size - 1)
	grid_width = W * grid_size + padding * (grid_size - 1)
	grid = np.zeros((grid_height, grid_width, C))
	next_idx = 0
	y0, y1 = 0, H

	for y in range(grid_size):
		x0, x1 = 0, W
		for x in range(grid_size):
			if next_idx < N:
				img = Xs[next_idx]
				low, high = np.min(img), np.max(img)
				grid[y0:y1, x0:x1] = ubound * (img - low) / (high - low)
				next_idx += 1
			x0 += W + padding
			x1 += W + padding
		y0 += H + padding
		y1 += H + padding
	return grid

def view_images(X, ubound=1.0, save=False, name=''):
	""" Quick helper function to view rgb or gray images."""
	if X.ndim == 3:
		H, W, C = X.shape
		X = X.reshape(H, W, C, 1)
		grid = visualize_grid(X, ubound)
		H, W, C = grid.shape
		grid = grid.reshape((H, W))
		plt.imshow(grid, cmap="Greys_r")
		if save:
			plt.savefig('/Users/kevin/Desktop/' + name, format='png', dpi=1000)
		plt.show()
	elif X.ndim == 4:
		grid = visualize_grid(X, ubound)
		plt.imshow(grid)
		if save:
			plt.savefig('/Users/kevin/Desktop/' + name, format='png', dpi=1000)
		plt.show()
	else:
		raise ValueError

def plot_images(images, cls_true, cls_pred=None):

	assert len(images) == len(cls_true) == 9

	# Create figure with sub-plots.
	fig, axes = plt.subplots(3, 3)

	for i, ax in enumerate(axes.flat):
		# plot the image
		ax.imshow(images[i], cmap='gray', interpolation='spline16')
			
		if cls_pred is None:
			xlabel = "Digit: {}".format(cls_true[i])
		else:
			xlabel = "True: {0}\nPred: {1}".format(cls_true[i], cls_pred[i])
			
		ax.set_xlabel(xlabel)
		ax.set_xticks([])
		ax.set_yticks([])

	plt.show()