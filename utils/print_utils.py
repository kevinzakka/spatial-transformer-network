import tensorflow as tf

# util function for viewing value
def view_value(x):
	# intialize the variable
	init_op = tf.global_variables_initializer()

	# run the graph
	with tf.Session() as sess:
		sess.run(init_op)
		print(sess.run(x))

# util function for converting tensor to numpy array
def run_op(x):
	# intialize the variable
	init_op = tf.global_variables_initializer()

	# run the graph
	with tf.Session() as sess:
		sess.run(init_op)
		return sess.run(x)
