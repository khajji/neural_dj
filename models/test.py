#!/usr/bin/python
import os, sys, math, random, pdb
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
sys.path.insert(0, os.path.abspath("../dataio/"))
from dataio import *


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def apply_cnn(Xtr, Ytr, Xte, Ground_Truth=None, drop_pr=0.5, n_training=1000):
	Xte, Ground_Truth = loadx(Xte), loady(Ground_Truth)

	x = tf.placeholder(tf.float32, shape=[None, 784]) #placeholder for data points
	y_ = tf.placeholder(tf.float32, shape=[None, 784]) #placeholder for labels
	x_image = tf.reshape(x, [-1,28,28,1]) #transform image to 2D 28*28 pixels matrix
	#y_image = tf.reshape(y_, [-1,28,28,1])

	#Hidden Layer 1: 
	W_conv1 = weight_variable([1, 1, 1, 16]) #weights: 5*5 filter and 32 features
	b_conv1 = bias_variable([16]) #bais
	h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) #convolution with relu transfer
	

	keep_prob = tf.placeholder(tf.float32) #dropout probability holder
	

	#Hidden layer 4: softmax
	W_fc2 = weight_variable([1,1,16,1])
	b_fc2 = bias_variable([1])
	y_conv = conv2d(h_conv1, W_fc2) + b_fc2
	y_res = tf.reshape(y_conv, [-1,784])


	#cross_entropy = tf.reduce_mean(
    #	tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv)) #we use the cross entropy loss
	#pdb.set_trace()
	#cross_entropy = tf.reduce_mean(tf.square(tf.subtract(y_conv, y_image)))
	cross_entropy = tf.reduce_mean(tf.divide(tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(y_res, y_)),reduction_indices=1)),784)) #the mean euclidian distance between predictions and labels
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy) #we use adam method to minimize the loss
	#correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
	#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	
	sess = tf.InteractiveSession()
	sess.run(tf.global_variables_initializer())

	
	#train
	for i in range(n_training):
		X_batch, Y_batch = sample_batch(Xtr, Ytr, 100, i)
		#X_batch, Y_batch = sample_seqential_batch(Xtr, Ytr, 100, i)
		#X_batch, Y_batch = sample_random_batch(Xtr, Ytr, 100)

		if i%50 == 0:
			train_accuracy = cross_entropy.eval(feed_dict= {x:X_batch, y_: Y_batch, keep_prob: 1.0})
			print("step %d, training accuracy %g"%(i, train_accuracy))
			#if Ground_Truth !=None:
				#print("test accuracy %g"%cross_entropy.eval(feed_dict={x: Xte, y_: Ground_Truth, keep_prob: 1.0}))
			#print("\n")
		train_step.run(feed_dict={x: X_batch, y_: Y_batch, keep_prob: 1-drop_pr})

	#return accuracy score on test set
	#predictions = vectorize_labels(sess.run(tf.argmax(y_conv,1), feed_dict={x: Xte, keep_prob: 1.0})) 
	predictions = y_res.eval(feed_dict={x: Xte, keep_prob:1.0})
	#confidences = np.max(sess.run(tf.nn.softmax(y_conv), feed_dict={x: Xte, keep_prob: 1.0}),1) #compute the probability associated with each prediction
	acc=None
	if Ground_Truth is not None:
		acc = cross_entropy.eval(feed_dict={x: Xte, y_: Ground_Truth, keep_prob: 1.0})
	return predictions, acc
	
	'''
	#Hidden Layer 1: 
	W_conv1 = weight_variable([5, 5, 1, 32]) #weights: 5*5 filter and 32 features
	b_conv1 = bias_variable([32]) #bais
	h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) #convolution with relu transfer
	#h_pool1 = max_pool_2x2(h_conv1) #do pooling

	#Hidden Layer 2: 
	W_conv2 = weight_variable([5, 5, 32, 64]) #weights: 5*5 patch and 64 output feature for each one of the 32 features
	b_conv2 = bias_variable([64]) #bais
	h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2) #convolution with relu transfer
	#h_pool2 = max_pool_2x2(h_conv2)#do pooling

	#Hidden layer 3: (fully connected layer)
	W_fc1 = weight_variable([28 * 28 * 64, 1024]) #weights: map neurons to 1024 output
	b_fc1 = bias_variable([1024]) #bais
	h_conv2_flat = tf.reshape(h_conv2, [-1, 28*28*64])
	h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1) #fully connected layer (i.e mat mul) with relu transfer
	keep_prob = tf.placeholder(tf.float32) #dropout probability holder
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob) #apply drop out on fully connected layer

	#Hidden layer 4: softmax
	W_fc2 = weight_variable([1024, 784])
	b_fc2 = bias_variable([784])
	y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

	#cross_entropy = tf.reduce_mean(
    #	tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv)) #we use the cross entropy loss
	pdb.set_trace()
	cross_entropy = tf.reduce_mean(tf.square(tf.subtract(y_conv, y_)))
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy) #we use adam method to minimize the loss
	#correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
	#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	sess = tf.InteractiveSession()
	sess.run(tf.global_variables_initializer())
	
	#train
	for i in range(n_training):
		X_batch, Y_batch = sample_batch(Xtr, Ytr, 100, i)
		#X_batch, Y_batch = sample_seqential_batch(Xtr, Ytr, 100, i)
		#X_batch, Y_batch = sample_random_batch(Xtr, Ytr, 100)

		if i%10 == 0:
			train_accuracy = cross_entropy.eval(feed_dict= {x:X_batch, y_: Y_batch, keep_prob: 1.0})
			print("step %d, training accuracy %g"%(i, train_accuracy))
			if Ground_Truth !=None:
				print("test accuracy %g"%cross_entropy.eval(feed_dict={x: Xte, y_: Ground_Truth, keep_prob: 1.0}))
			print("\n")
		train_step.run(feed_dict={x: X_batch, y_: Y_batch, keep_prob: 1-drop_pr})

	#return accuracy score on test set
	#predictions = vectorize_labels(sess.run(tf.argmax(y_conv,1), feed_dict={x: Xte, keep_prob: 1.0})) 
	predictions = y_conv.eval(feed_dict={x: Xte, keep_prob:1.0})
	#confidences = np.max(sess.run(tf.nn.softmax(y_conv), feed_dict={x: Xte, keep_prob: 1.0}),1) #compute the probability associated with each prediction
	acc=None
	if Ground_Truth !=None:
		acc = cross_entropy.eval(feed_dict={x: Xte, y_: Ground_Truth, keep_prob: 1.0})
	return predictions, acc'''


if __name__ == '__main__':

	data_path="../data_sample/pprocessed_data2"
	prediction_path = "../data_sample/predictions"
	#pdb.set_trace()
	dataset = Dataset(data_path)
	Xtr, Ytr, Xte, Ground_Truth = dataset.split(ratio=0.8)

	#load data
	train_iterations = 200
	dropout_probability = 0.5
	#Xtr, Ytr, Xte, Ground_Truth = load_data()
	predictions, acc = apply_cnn(Xtr, Ytr, Xte, Ground_Truth=Ground_Truth, drop_pr=dropout_probability, n_training=train_iterations)
	#predict_testlabels(Xtr, Ytr, Xte, train_iterations)
	
	print("loss on validation set : "+str(acc))
	
	output_names = dataset.give_outputs()
	save(predictions, prediction_path, output_names)
	#predictions, confidences, accuracy = predict_onvalidation(Xtr, Ytr, train_iterations)
	#log_likelihood = np.sum(np.log(confidences))/np.size(confidences)
	#pdb.set_trace()
	#print("log likelihood prediction (confidence of predictions):"+str(log_likelihood))
	

	

