import pandas as pd
import numpy as np
import argparse
import pickle
import utils.config as config
import tensorflow as tf

from tensorflow.contrib.tensor_forest.python import tensor_forest
from tensorflow.python.ops import resources
from sklearn.model_selection import train_test_split
from utils.data_loader import build_vocab, Vocabulary


def main(args):
	vocab = build_vocab(args.data_path)
	data = pd.DataFrame({
		'label':vocab.labels,
		'lprox':vocab.lprox,
		'rprox':vocab.rprox,
		'x':vocab.x,
		'y':vocab.y,
		'z':vocab.z,
	})
	y = data['label']
	lprox = pd.DataFrame(data['lprox'].values.tolist())
	rprox = pd.DataFrame(data['rprox'].values.tolist())
	xax = pd.DataFrame(data['x'].values.tolist())
	yax = pd.DataFrame(data['y'].values.tolist())
	zax = pd.DataFrame(data['z'].values.tolist())

	X = pd.concat([lprox, rprox, xax, yax, zax], axis=1)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

	num_steps = 100 # Total steps to train
	num_classes = 2 
	num_features = 585
	num_trees = 10 
	max_nodes = 1000


	X = tf.placeholder(tf.float32, shape=[None, num_features])
	Y = tf.placeholder(tf.int64, shape=[None]) 

	hparams = tensor_forest.ForestHParams(num_classes=num_classes, num_features=num_features, num_trees=num_trees, max_nodes=max_nodes).fill()
	forest_graph = tensor_forest.RandomForestGraphs(hparams)
	train_op = forest_graph.training_graph(X, Y)

	loss_op = forest_graph.training_loss(X, Y)
	infer_op, _, _ = forest_graph.inference_graph(X)
	correct_prediction = tf.equal(tf.argmax(infer_op, 1), tf.cast(Y, tf.int64))
	accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	init_vars = tf.group(tf.global_variables_initializer(), resources.initialize_resources(resources.shared_resources()))

	sess = tf.Session()
	sess.run(init_vars)

	for i in range(1, num_steps + 1):
		saver = tf.train.Saver()
		_, l = sess.run([train_op, loss_op], feed_dict={X: X_train, Y: y_train})
		if i % 50 == 0 or i == 1:
			acc = sess.run(accuracy_op, feed_dict={X: X_train, Y: y_train})
			save_path = saver.save(sess, 'models/model%i.ckpt' % (i))
			print('Step %i, Loss: %f, Acc: %f' % (i, l, acc))

	print("Test Accuracy:", sess.run(accuracy_op, feed_dict={X: X_test, Y: y_test}))


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_path', type=str,
		default='data/', help='path to data files')
	# parser.add_argument('--output', type=str, required=True,
	# 	help='filename to write model out to')
	args = parser.parse_args()
	main(args)