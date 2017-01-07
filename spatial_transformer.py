import tensorflow as tf

def spatial_transformer_network(input_fmap, theta, **kwargs):
	"""
	Spatial Transformer Network layer implementation as described in [1].

	The layer is composed of 3 elements:

	- localisation_net: takes the original image as input and outputs 
	  the parameters of the affine transformation that should be applied
	  to the input image.

	- affine_grid_generator: generates a grid of (x,y) coordinates that 
	  correspond to a set of points where the input should be sampled 
	  to produce the transformed output.

	- bilinear_sampler: takes as input the original image and the grid
	  and produces the output image using bilinear interpolation.

	Input
	-----
	- input_fmap: output of the previous layer. Can be input if spatial
	  transformer layer is at the beginning of architecture. Should be 
	  a tensor of shape (B, H, W, C). 

	- theta: affine transform tensor of shape (B, 6). Permits cropping, 
	  translation and isotropic scaling. Initialize to identity matrix. 
	  It is the output of the localization network.

	Returns
	-------
	- out_fmap: transformed input feature map. Tensor of size (B, H, W, C).

	Notes
	-----
	[1]: 'Spatial Transformer Networks', Jaderberg et. al,
		 (https://arxiv.org/abs/1506.02025)

	"""
	# grab input dimensions
	B = tf.shape(input_fmap)[0]
	H = tf.shape(input_fmap)[1]
	W = tf.shape(input_fmap)[2]
	C = tf.shape(input_fmap)[3]

	# reshape theta to (B, 2, 3)
	theta = tf.reshape(theta, [B, 2, 3])

	# generate grids
	batch_grids = affine_grid_generator(H, W, theta)

	# extract x and y coordinates
	# x_s = tf.squeeze(tf.slice(batch_grids, [0, 0, 0, 0], [-1, -1, -1, 1]))
	# y_s = tf.squeeze(tf.slice(batch_grids, [0, 0, 0, 1], [-1, -1, -1, 1]))
	x_s = tf.squeeze(batch_grids[:, :, :, 0:1])
	y_s = tf.squeeze(batch_grids[:, :, :, 1:2])

	# sample input with grid to get output
	out_fmap = bilinear_sampler(input_fmap, x_s, y_s)

	return out_fmap

def get_pixel_value(img, x, y):
	"""
	Utility function to get pixel value for 
	coordinate vectors x and y from a  4D tensor image.
	"""
	B = tf.shape(img)[0]
	H = tf.shape(img)[1]
	W = tf.shape(img)[2]
	C = tf.shape(img)[3]

	# flatten image
	img_flat = tf.reshape(img, [-1, C])

	# flatten idx
	idx_flat = (y * W) + x

	return tf.gather(img_flat, idx_flat)

def affine_grid_generator(height, width, theta):
	"""
	This function returns a sampling grid, which when
	used with the bilinear sampler on the input feature 
	map, will create an output feature map that is an 
	affine transformation [1] of the input feature map.

	Input
	-----
	- height: desired height of grid/output. Used
	  to downsample or upsample. 

	- width: desired width of grid/output. Used
	  to downsample or upsample. 

	- theta: affine transform matrices of shape (num_batch, 2, 3). 
	  For each image in the batch, we have 6 theta parameters of 
	  the form (2x3) that define the affine transformation T.

	Returns
	-------
	- normalized gird (-1, 1) of shape (num_batch, H, W, 2).
	  The 4th dimension has 2 components: (x, y) which are the 
	  sampling points of the original image for each point in the
	  target image.

	Note
	----
	[1]: the affine transformation allows cropping, translation, 
		 and isotropic scaling.
	"""
	# grab batch size
	num_batch = tf.shape(theta)[0]

	# create normalized 2D grid
	x = tf.linspace(-1.0, 1.0, width)
	y = tf.linspace(-1.0, 1.0, height)
	x_t, y_t = tf.meshgrid(x, y)

	# flatten
	x_t_flat = tf.reshape(x_t, [-1])
	y_t_flat = tf.reshape(y_t, [-1])

	# reshape to (x_t, y_t , 1)
	ones = tf.ones_like(x_t_flat)
	sampling_grid = tf.stack([x_t_flat, y_t_flat, ones])

	# repeat grid and theta num_batch times
	sampling_grid = tf.expand_dims(sampling_grid, axis=0)
	sampling_grid = tf.tile(sampling_grid, tf.pack([num_batch, 1, 1]))
	# thetas = tf.expand_dims(theta, axis=0)
	# thetas = tf.tile(thetas, tf.pack([num_batch, 1, 1]))

	# cast to float32 (required for matmul)
	theta = tf.cast(theta, 'float32')
	sampling_grid = tf.cast(sampling_grid, 'float32')
	
	# transform the sampling grid - batch multiply
	batch_grids = tf.matmul(theta, sampling_grid)
	# batch grid has shape (num_batch, 2, H*W)

	# reshape to (num_batch, H, W, 2)
	batch_grids = tf.reshape(batch_grids, [num_batch, height, width, 2])

	return batch_grids

def bilinear_sampler(img, x, y):
	"""
	Performs bilinear sampling of the input images according to the 
	normalized coordinates provided by the sampling grid. Note that 
	the sampling is done identically for each channel of the input.

	To test if the function works properly, output image should be
	identical to input image when theta is initialized to identity
	transform.

	Input
	-----
	- img: batch of images in (B, H, W, C) layout.
	- grid: x, y which is the output of affine_grid_generator.

	Returns
	-------
	- interpolated images according to grids. Same size as grid.

	"""
	# grab dimensions
	B = tf.shape(img)[0]
	H = tf.shape(img)[1]
	W = tf.shape(img)[2]
	C = tf.shape(img)[3]

	# # flatten
	# x = tf.reshape(x, [-1])
	# y = tf.reshape(y, [-1])

	# cast as float32 (required for rescaling)
	x = tf.cast(x, 'float32')
	y = tf.cast(y, 'float32')

	# and rescale x and y to [0, W/H]
	x = ((x + 1.) * tf.cast(W, 'float32')) * 0.5
	y = ((y + 1.) * tf.cast(H, 'float32')) * 0.5

	# grab 4 nearest corner points for each (x_i, y_i)
	x0 = tf.cast(tf.floor(x), 'int32')
	x1 = x0 + 1
	y0 = tf.cast(tf.floor(y), 'int32')
	y1 = y0 + 1

	# making sure we don't violate image boundaries
	zero = tf.zeros([], dtype='int32')
	max_y = tf.cast(H - 1, 'int32')
	max_x = tf.cast(W - 1, 'int32')

	# so we clip to range [0, H] or [0, W]
	x0 = tf.clip_by_value(x0, zero, max_x)
	x1 = tf.clip_by_value(x1, zero, max_x)
	y0 = tf.clip_by_value(y0, zero, max_y)
	y1 = tf.clip_by_value(y1, zero, max_y)

	# get pixel value at corner coords
	Ia = get_pixel_value(img, x0, y0)
	Ib = get_pixel_value(img, x0, y1)
	Ic = get_pixel_value(img, x1, y0)
	Id = get_pixel_value(img, x1, y1)
	
	# recast as float for delta calculation
	x0 = tf.cast(x0, 'float32')
	x1 = tf.cast(x1, 'float32')
	y0 = tf.cast(y0, 'float32')
	y1 = tf.cast(y1, 'float32')

	# calculate deltas
	wa = (x1-x) * (y1-y)
	wb = (x1-x) * (y-y0)
	wc = (x-x0) * (y1-y)
	wd = (x-x0) * (y-y0)

	# add dimension for addition
	wa = tf.expand_dims(wa, axis=3)
	wb = tf.expand_dims(wb, axis=3)
	wc = tf.expand_dims(wc, axis=3)
	wd = tf.expand_dims(wd, axis=3)

	# compute output
	out = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])

	return out
