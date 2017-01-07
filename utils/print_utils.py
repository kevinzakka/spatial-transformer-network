import tensorflow as tf

def view_value(x):
	# intialize the variable
	init_op = tf.global_variables_initializer()

	# run the graph
	with tf.Session() as sess:
		sess.run(init_op)
		print(sess.run(x))