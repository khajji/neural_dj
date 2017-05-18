#!/usr/bin/python
import os, sys, pdb
import numpy as np
sys.path.insert(0, os.path.abspath("../data_processing/"))
from dataio import *
import matplotlib.pyplot as plt
import tensorflow as tf
class Cnn:
	def __init__(x_shape, y_shape, netwrok_specs): 
		#x_shape = [batch, height, width, channels] 
		#y_shape=[batch, height, width, channels]
		#network_specs = #[(['conv', ('height', 'width', 'depth'), 'activation'], number), (['connected', (outsize), 'activation'], number)]
		self.specs = netwrok_specs; self.x_shape = x_shape; self.y_shape = y_shape
		self.loss = None
		self.y_hat=None


	def build_network():
		
		xshape, yshape, specs = specs[0], specs[1], specs[2:]
		x = tf.placeholder(tf.float32, shape=self.x_shape)
		y = tf.placeholder(tf.float32, shape=self.y_shape)
		z=x
		for spec in self.specs:
			([layer, params, activation], depth) = spec
			for in i in range(depth):
				if layer == 'conv':
					(height, width, depth)=params
					z = convolution_layer(z, height, width, depth , activation)
				elif layer == 'connected':
					(outsize) = params
					z =  connected_layer(z, outsize , activation)

		self.y_hat=z
		self.loss = tf.reduce_mean(tf.square(tf.subtract(y_hat, y)))

	def train(xTr, yTr, batchsize=50, n_training=10000):
		#Xtr list of files, Ytr list of files
		train_step = tf.train.AdamOptimizer(1e-4).minimize(self.loss) #we use adam method to minimize the loss

		sess = tf.InteractiveSession()
		sess.run(tf.global_variables_initializer())

		for i in range(n_training):
			x_batch, y_batch = sample_batch(xTr, ytr, batchsize, i)
			train_step.run(feed_dict={x: x_batch, y_in: y_batch})

		sess.close()

		

	def test(xVl, yVl):
		sess = tf.InteractiveSession()
		sess.run(tf.global_variables_initializer())
		y_pred=self.y_hat.eval(feed_dict={x: xVl})
		score = self.loss.eval()accuracy.eval(feed_dict={x: xVe, y_: yVl})

		return y_pred, loss

	@staticmethod
	def weight_variable(shape):
		initial = tf.truncated_normal(shape, stddev=0.1)
		return tf.Variable(initial)
	@staticmethod
	def conv2d(x, W):
		return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

	@staticmethod
	def convolution_layer(x, height, width, depth , activation):
		height, width, in_channels, depth = filter_height, filter_width, x.shape()[3], out_channels
		w = weight_variable([height, width, in_channels, depth])
		b = bias_variable([depth])
		conv = conv1d(z, w, padding='SAME'
    	stride=[1,1,1,1], #stride on [filter_height, filter_width, in_channels, out_channels]
    	)
		z = activation(conv + b)
		return z

	@staticmethod
	def connected_layer(x, output_size , activation):
		[batch, in_height, in_width, in_channels]=x.shape()
		insize, outsize = in_height*in_width*in_channels, outsize
		w = weight_variable([insize, outsize])
		b = bias_variable([outsize])
		z = activation(tf.matmul(x, w) + b)
		return z





	




