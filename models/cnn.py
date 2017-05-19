#!/usr/bin/python
import os, sys, pdb
import numpy as np
sys.path.insert(0, os.path.abspath("../dataio/"))
from dataio import *
import matplotlib.pyplot as plt
import tensorflow as tf
class Cnn:
	def __init__(self, data_specs, network_specs): 
		#data_specs = (x_shape, y_shape, xnetwork_shape, ynetwork_shape)
		#y_shape=[batch, height, width, channels]
		#network_specs = #[(['conv', ('height', 'width', 'depth'), 'activation'], number), (['connected', (outsize), 'activation'], number)]
		self.specs = network_specs; 
		self.x_shape, self.y_shape, self.xcnn_shape, self.ycnn_shape = data_specs
		#tensors
		self.x, self.y, self.xnn, self.ycnn = None, None, None, None
		self.ycnn_hat, self.y_hat=None, None
		self.loss = None;
		self.dropout = None 
		self.build_network()


	def build_network(self):
		
		specs = self.specs
		self.x = tf.placeholder(tf.float32, shape=self.x_shape)
		self.y = tf.placeholder(tf.float32, shape=self.y_shape)
		self.xcnn = tf.reshape(self.x, Cnn.reshape(self.xcnn_shape))#[-1,1,self.x_shape,1]
		self.ycnn = tf.reshape(self.y, Cnn.reshape(self.ycnn_shape)) #[-1,1,self.y_shape,1]
		z=self.xcnn; self.dropout = tf.placeholder(tf.float32) #dropout probability holder
		
		
		for spec in self.specs:
			([layer, params, activation], depth) = spec
			for i in range(depth):
				if layer == 'conv':
					(height, width, depth)=params
					z = Cnn.convolution_layer(z, height, width, depth , activation)
				elif layer == 'connected':
					(outsize) = params
					z =  Cnn.connected_layer(z, outsize , activation)
				elif layer == 'dropout':
					z = Cnn.dropout_layer(z, self.dropout)

		self.ycnn_hat=tf.reshape(z, Cnn.reshape(self.ycnn_shape))
		self.loss = tf.reduce_mean(tf.square(tf.subtract(self.ycnn_hat, self.ycnn)))

		self.y_hat=tf.reshape(self.ycnn_hat, Cnn.reshape(self.y_shape))


	def train(self, xTr, yTr, dropout=0.5, batchsize=50, n_training=10000): #data are file paths
		#Xtr list of files, Ytr list of files
		train_step = tf.train.AdamOptimizer(1e-4).minimize(self.loss) #we use adam method to minimize the loss

		sess = tf.InteractiveSession()
		sess.run(tf.global_variables_initializer())

		for i in range(n_training):
			x_batch, y_batch = sample_batch(xTr, yTr, batchsize, i)

			if i%10==0:
				#pdb.set_trace()
				loss = self.loss.eval(feed_dict={self.x: x_batch, self.y: y_batch, self.dropout: dropout})
				print ("iteration "+str(i)+": loss = "+str(loss))

			train_step.run(feed_dict={self.x: x_batch, self.y: y_batch, self.dropout: dropout})

		'''
		xVl, yVl = loadx(xVl), loady(yVl)
		y_pred=self.y_hat.eval(feed_dict={self.x: xVl, self.dropout:0.0})
		score = self.loss.eval(feed_dict={self.x: xVl, self.y: yVl, self.dropout:0.0})
		return y_pred, score'''
		sess.close()
		

		

	def test(self, xVl, yVl): #data are file paths
		xVl, yVl = loadx(xVl), loady(yVl)
		sess = tf.InteractiveSession()

		sess.run(tf.global_variables_initializer())
		y_pred=self.y_hat.eval(feed_dict={self.x: xVl, self.dropout:0})
		score = self.loss.eval(feed_dict={self.x: xVl, self.y: yVl, self.dropout:0})
		sess.close()
		return y_pred, score

	@staticmethod
	def weight_variable(shape):
		initial = tf.truncated_normal(shape, stddev=0.1)
		return tf.Variable(initial)

	@staticmethod	
	def bias_variable(shape):
  		initial = tf.constant(0.1, shape=shape)
  		return tf.Variable(initial)

	@staticmethod
	def convolution_layer(x, filter_height, filter_width, filter_depth , activation):
		height, width, in_channels, depth = filter_height, filter_width, x.get_shape().as_list()[3], filter_depth
		w = Cnn.weight_variable([height, width, in_channels, depth])
		b = Cnn.bias_variable([depth])
		conv = tf.nn.conv2d(x, w, padding='SAME', strides=[1,1,1,1]) #stride on [filter_height, filter_width, in_channels, out_channels]
		z = activation(conv + b)
		return z

	@staticmethod
	def connected_layer(x, output_size , activation):
		[batch, in_height, in_width, in_channels]=x.get_shape().as_list()
		insize, outsize = in_height*in_width*in_channels, output_size
		x = tf.reshape(x, [-1, in_height*in_width*in_channels])
		w = Cnn.weight_variable([insize, outsize])
		b = Cnn.bias_variable([outsize])
		z = activation(tf.matmul(x, w) + b)
		return z

	@staticmethod
	def dropout_layer(x, dropout_tensor):
		z = tf.nn.dropout(x, 1-dropout_tensor) 
		return z

	@staticmethod
	def reshape(shape):
		return [s if s is not None else -1 for s in shape]





	




