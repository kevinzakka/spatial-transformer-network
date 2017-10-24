import os
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from utils.data_utils import *
from utils.vis_utils import *
from utils.layer_utils import *
from utils.print_utils import *
from transformer import spatial_transformer_network as stn

'''tensorboard --logdir=/home/kevin/Desktop/transformer/logs/mnist/1 --port=6006'''

# ensuring reproducibility
np.random.seed(42)

# global params
SAMPLE = True
VIEW = False
RESTORE = False
MODE = 'train'

# directory paths
root_dir = '/home/kevin/Desktop/transformer/data/'
logs_dir = '/home/kevin/Desktop/transformer/logs/'
save_dir = '/home/kevin/Desktop/transformer/checkpoints/'
vis_path = '/home/kevin/Desktop/transformer/samples/'

# network params
H, W, C = 60, 60, 1
img_shape = (H, W, C)
img_shape_flat = H * W * C
num_classes = 10

# training params
display_step = 1
learning_rate = 1e-3
batch_size = 64
num_epochs = 50
best_validation_accuracy = 0.0
last_improvement = 0
require_improvement = 1000

# define placeholder variables
X = tf.placeholder(tf.float32, [None, H, W, C], name='X')
y = tf.placeholder(tf.uint8, [None, num_classes], name='y')
phase = tf.placeholder(tf.bool, name='phase')

def load_data(root_dir, view_grid=False):
	"""
	Utility function for loading the cluttered MNIST data.

	Input
	-----
	- view_grid: (bool) choose to view a grid of 250 data samples. 

	Returns
	-------
	- X_train: (50000, 60, 60, 1)
	- X_test: (10000, 60, 60, 1)
	- X_valid: (10000, 60, 60, 1)
	- y_train: (50000, 10)
	- y_test: (10000, 10)
	- y_valid: (10000, 10)
	"""
	mnist_cluttered = "mnist_cluttered_60x60_6distortions.npz"
	data = np.load(root_dir + mnist_cluttered)

	X_train, y_train = data['x_train'], data['y_train']
	X_valid, y_valid = data['x_valid'], data['y_valid']
	X_test, y_test = data['x_test'], data['y_test']

	# let's view a grid of the images
	if view_grid:
		mask = np.arange(250)
		sample = np.reshape(X_train, [-1, 60, 60])[mask]
		view_images(sample)

	# reshape
	X_train = np.reshape(X_train, [-1, 60, 60, 1])
	X_test = np.reshape(X_test, [-1, 60, 60, 1])
	X_valid = np.reshape(X_valid, [-1, 60, 60, 1])

	return X_train, y_train, X_test, y_test, X_valid, y_valid

def random_batch(X, y, batch_size=batch_size):
	"""
	Randomly selects batch_size elements from the 4D
	tensor X and its equivalent ground truth y.
	"""
	num_train = len(X)
	batch_mask = np.random.choice(num_train, batch_size)
	X_batch = X[batch_mask]
	y_batch = y[batch_mask]

	return X_batch, y_batch

def generate_batch_indices(X, batch_size=batch_size):
	"""
	Given a certain batch size, this function computes the
	batch indices needed to index into the training tensor
	X_train during the training loop.

	For example, given num_train = 500, batch_size = 64,
	it would take 8 batches (8*64=512) to pass through 
	X_train.

	Furthermore, the batch_indices would look as follows:
	(0, 64), (64, 128), (128, 192) etc.

	Returns
	-------
	- total_batch: number of iterations or batches needed to 
	  complete 1 run of the training vector for 1 epoch.
	- batch_indices: tuple (x, y) containing the starting
	  and ending point of the each batch.
	"""
	num_train = len(X)
	total_batch = int(np.ceil(num_train / float(batch_size)))
	batch_indices = [(i * batch_size, min(num_train, (i + 1) * batch_size)) 
					for i in range(0, total_batch)]

	return total_batch, batch_indices

def validate_acc_loss(session, loss, accuracy, X_val, y_val):

	avg_loss = 0.0
	avg_acc = 0.0
	total_batch, batch_idx = generate_batch_indices(X_val)

	for i in range(total_batch):

		# create batch
		idx = batch_idx[i]
		mask = np.arange(idx[0], idx[1])
		batch_xs, batch_ys = X_val[mask], y_val[mask]

		# construct feed dict and evaluate loss and acc
		valid_feed_dict = {X: batch_xs, y: batch_ys, phase: True}
		l, a = session.run([loss, accuracy], feed_dict=valid_feed_dict)

		avg_loss += l / total_batch
		avg_acc += a / total_batch

	return avg_loss, avg_acc

def test_acc(session, accuracy, X_test, y_test):

	avg_acc = 0.0
	total_batch, batch_idx = generate_batch_indices(X_test)

	for i in range(total_batch):

		# create batch
		idx = batch_idx[i]
		mask = np.arange(idx[0], idx[1])
		batch_xs, batch_ys = X_test[mask], y_test[mask]

		# construct feed dict and evaluate loss and acc
		test_feed_dict = {X: batch_xs, y: batch_ys, phase: False}
		a = session.run(accuracy, feed_dict=test_feed_dict)

		avg_acc += a / total_batch

	return avg_acc

def build_convnet():
	# localization network
	conv1_loc = Conv2D(X, 1, 5, 32, name='conv1_loc')
	pool1_loc = MaxPooling2D(conv1_loc, use_relu=True, name='pool1_loc')
	conv2_loc = Conv2D(pool1_loc, 32, 5, 64, name='conv2_loc')
	pool2_loc = MaxPooling2D(conv2_loc, use_relu=True, name='pool2_loc')

	pool2_loc_flat, pool2_loc_size = Flatten(pool2_loc)

	fc1_loc = Dense(pool2_loc_flat, pool2_loc_size, 2048, use_relu=False, name='fc1_loc')
	fc2_loc = Dense(fc1_loc, 2048, 512, use_relu=True, name='fc2_loc')
	fc3_loc = Dense(fc2_loc, 512, 6, use_relu=False, trans=True, name='fc3_loc')

	# spatial transformer
	h_trans = stn(X, fc3_loc)

	# convnet
	conv1 = Conv2D(X, 1, 5, 32, name='conv1')
	bn1 = BatchNormalization(conv1, phase, name='bn1')
	pool1 = MaxPooling2D(bn1, use_relu=True, name='pool1')

	conv2 = Conv2D(pool1, 32, 5, 64, name='conv2')
	bn2 = BatchNormalization(conv2, phase, name='bn2')
	pool2 = MaxPooling2D(bn2, use_relu=True, name='pool2')

	conv3 = Conv2D(pool2, 64, 3, 128, name='conv3')
	bn3 = BatchNormalization(conv3, phase, name='bn3')
	pool3 = MaxPooling2D(bn3, use_relu=True, name='pool3')

	pool3_flat, pool3_size = Flatten(pool3)

	fc1 = Dense(pool3_flat, pool3_size, 2048, use_relu=False, name='fc1')
	bn4 = BatchNormalization(fc1, phase, use_relu=True, name='bn4')
	fc2 = Dense(bn4, 2048, 512, use_relu=False, name='fc2')
	bn5 = BatchNormalization(fc2, phase, use_relu=True, name='bn5')
	logits = Dense(bn5, 512, num_classes, name='fc3', use_relu=False)

	return h_trans, logits

def main():

	# load the data
	print("Loading the data...")
	X_train, y_train, X_test, y_test, X_valid, y_valid = load_data(root_dir)

	# sanity check dimensions
	# print("Train: {}".format(X_train.shape))
	# print("Test: {}".format(X_test.shape))
	# print("Valid: {}".format(X_valid.shape))

	# let's view a small sample
	if VIEW:
		mask = np.arange(9)
		gd_truth = np.argmax(y_train[mask], axis=1)
		sample = X_train.squeeze()[mask]
		plot_images(sample, gd_truth)

	if SAMPLE:
		mask = np.arange(500)
		X_train = X_train[mask]
		y_train = y_train[mask]

	num_train = X_train.shape[0]
	gd_truth = np.argmax(y_train, axis=1)

	# # let's check the frequencies of each class
	# plt.hist(gd_truth, bins=num_classes)
	# plt.title("Ground Truth Labels")
	# plt.xlabel("Class")
	# plt.ylabel("Frequency")
	# plt.show()

	print("Building ConvNet...")
	conv1_loc = Conv2D(X, 1, 5, 32, name='conv1_loc')
	pool1_loc = MaxPooling2D(conv1_loc, use_relu=True, name='pool1_loc')
	conv2_loc = Conv2D(pool1_loc, 32, 5, 64, name='conv2_loc')
	pool2_loc = MaxPooling2D(conv2_loc, use_relu=True, name='pool2_loc')

	pool2_loc_flat, pool2_loc_size = Flatten(pool2_loc)

	fc1_loc = Dense(pool2_loc_flat, pool2_loc_size, 2048, use_relu=False, name='fc1_loc')
	fc2_loc = Dense(fc1_loc, 2048, 512, use_relu=True, name='fc2_loc')
	fc3_loc = Dense(fc2_loc, 512, 6, use_relu=False, trans=True, name='fc3_loc')

	# spatial transformer
	h_trans = stn(X, fc3_loc)

	# convnet
	conv1 = Conv2D(h_trans, 1, 5, 32, name='conv1')
	bn1 = BatchNormalization(conv1, phase, name='bn1')
	pool1 = MaxPooling2D(bn1, use_relu=True, name='pool1')

	conv2 = Conv2D(pool1, 32, 5, 64, name='conv2')
	bn2 = BatchNormalization(conv2, phase, name='bn2')
	pool2 = MaxPooling2D(bn2, use_relu=True, name='pool2')

	conv3 = Conv2D(pool2, 64, 3, 128, name='conv3')
	bn3 = BatchNormalization(conv3, phase, name='bn3')
	pool3 = MaxPooling2D(bn3, use_relu=True, name='pool3')

	pool3_flat, pool3_size = Flatten(pool3)

	fc1 = Dense(pool3_flat, pool3_size, 2048, use_relu=False, name='fc1')
	bn4 = BatchNormalization(fc1, phase, use_relu=True, name='bn4')
	fc2 = Dense(bn4, 2048, 512, use_relu=False, name='fc2')
	bn5 = BatchNormalization(fc2, phase, use_relu=True, name='bn5')
	logits = Dense(bn5, 512, num_classes, name='fc3', use_relu=False)

	# define cost function
	cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)
	loss = tf.reduce_mean(cross_entropy)

	# define optimizer
	global_step = tf.Variable(initial_value=0, name='global_step', trainable=False)
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step)

	# define accuracy
	correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

	# define saver object for storing and retrieving checkpoints
	saver = tf.train.Saver()
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	save_path = os.path.join(save_dir, 'best_validation') # path for the checkpoint file

	total_batch = int(np.ceil(num_train / float(batch_size)))
	num_iterations = num_epochs * total_batch

	global best_validation_accuracy
	global last_improvement
	global require_improvement

	# create summary for loss and acc
	tf.summary.scalar('train_loss', loss)
	tf.summary.scalar('train_accuracy', accuracy)
	summary_op = tf.summary.merge_all()

	if not os.path.exists(logs_dir):
		os.makedirs(logs_dir)
	logs_path = os.path.join(logs_dir, 'cluttered_mnist/')

	if not os.path.exists(vis_path):
		os.makedirs(vis_path)

	with tf.Session() as sess:

		if RESTORE:
			# restore checkpoint if it exists
			try:
				print("Trying to restore last checkpoint ...")
				last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=save_dir)
				saver.restore(sess, save_path=last_chk_path)
				print("Restored checkpoint from:", last_chk_path)
			except:
				print("Failed to restore checkpoint. Initializing variables instead.")		
				sess.run(tf.global_variables_initializer())
		else:
			sess.run(tf.global_variables_initializer())

		# for tensorboard viewing
		writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

		# for visualization purposes
		fig = plt.figure()

		if MODE == 'train':

			tic = time.time()
			print("Training on {} samples, validating on {} samples".format(len(X_train), len(X_valid)))

			iter_per_epoch, batch_indices = generate_batch_indices(X_train)
			batch_indices = batch_indices * num_epochs
			epoch_num = 0

			for i in range(num_iterations):

				# grab the batch index from list 
				idx = batch_indices[i]
				mask = np.arange(idx[0], idx[1])

				# slice into batches
				batch_X_train, batch_y_train = X_train[mask], y_train[mask]

				# create feed dict
				train_feed_dict = {X: batch_X_train, y: batch_y_train, phase: True}

				i_global, _ = sess.run([global_step, optimizer], feed_dict=train_feed_dict)

				if (i_global % display_step == 0) or (i == num_iterations - 1):

					# calculate loss and accuracy on training batch
					train_batch_loss, train_batch_acc, train_summary = sess.run([loss, accuracy, summary_op], feed_dict=train_feed_dict)
					writer.add_summary(train_summary, i_global)

					# calculate loss and accuracy on validation batch
					valid_batch_loss, valid_batch_acc = validate_acc_loss(sess, loss, accuracy, X_valid, y_valid)

					# check to see if there's an improvement
					improved_str = ''
					if valid_batch_acc > best_validation_accuracy:
						best_validation_accuracy = valid_batch_acc
						last_improvement = i_global
						saver.save(sess=sess, save_path=save_path+str(best_validation_accuracy), global_step=i_global)
						improved_str = '*'

					print("Iter: {}/{} - loss: {:.4f} - acc: {:.4f} - val_loss: {:.4f} - val_acc: {:.4f} - {}".format(i_global, 
							num_iterations, train_batch_loss, train_batch_acc, valid_batch_loss, valid_batch_acc, improved_str))

				# if no improvement in a while, stop training
				if i_global - last_improvement > require_improvement:
					print("No improvement found in a while, stopping optimization.")
					break

				# for plotting
				if i_global == 1:
					print("Plotting input imgs...")
					input_imgs = batch_X_train[:9]
					input_imgs = np.reshape(input_imgs, [-1, 60, 60])
					plt.clf()
					for j in range(9):
						plt.subplot(3, 3, j+1)
						plt.imshow(input_imgs[j], cmap='gray')
						plt.axis('off')
					fig.canvas.draw()
					plt.savefig(vis_path + 'epoch_0.png', bbox_inches='tight')

				# plotting
				thetas = sess.run(h_trans, feed_dict={X: batch_X_train, phase: True})
				thetas = thetas[0:9].squeeze()
				plt.clf()
				for j in range(9):
					plt.subplot(3, 3, j+1)
					plt.imshow(thetas[j], cmap='gray')
					plt.axis('off')
				fig.canvas.draw()
				plt.savefig(vis_path + 'epoch_' + str(i_global) + '.png', bbox_inches='tight')

			toc = time.time()
			print("Time: {:.2f}s".format(toc-tic))
			print("Best valid acc: {}".format(best_validation_accuracy))

		else:
			test_accuracy = test_acc(sess, accuracy, X_test, y_test)
			print("Test Set Accuracy: {}".format(test_accuracy))

if __name__ == '__main__':
	main()
