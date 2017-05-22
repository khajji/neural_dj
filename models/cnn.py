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
		self.session = None
		self.build_network()


	def build_network(self):
		
		specs = self.specs
		self.x = tf.placeholder(tf.float32, shape=self.x_shape)
		self.y = tf.placeholder(tf.float32, shape=self.y_shape)
		#pdb.set_trace()
		self.xcnn = tf.reshape(self.x, Cnn.reshape(self.xcnn_shape))#[-1,1,self.x_shape,1]
		#self.ycnn = tf.reshape(self.y, Cnn.reshape(self.ycnn_shape)) #[-1,1,self.y_shape,1]
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
				elif layer == 'pooling':
					(height, width)=params
					z = Cnn.pooling_layer(z, height, width)

		self.y_hat = tf.reshape(z, Cnn.reshape(self.y.get_shape().as_list()))
		#self.y_hat = tf.reshape(z, [-1,784])
		#y_res = tf.reshape(y_conv, )
		self.loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(self.y_hat, self.y)),reduction_indices=1))) #the mean euclidian distance between predictions and labels
		#self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.y_hat))
		#self.loss = tf.reduce_mean(tf.square(tf.subtract(self.y_hat, self.y)))
		#self.ycnn_hat=tf.reshape(z, Cnn.reshape(self.ycnn_shape))
		
		#self.y_hat=tf.reshape(self.ycnn_hat, Cnn.reshape(self.y_shape))


	def train(self, xTr, yTr, dataset, dropout=0.5, batchsize=50, n_training=10000, xVl=None, yVl=None): #data are file paths
		#Xtr list of files, Ytr list of files
		prediction_path="../data_sample/predictions/big_test"
		validation_loss = None
		train_step = tf.train.AdamOptimizer(1e-4).minimize(self.loss) #we use adam method to minimize the loss

		self.session = tf.InteractiveSession()
		self.session.run(tf.global_variables_initializer())
		#pdb.set_trace()
		prediction_number = 0
		for i in range(n_training):
			x_batch, y_batch, _, _ = sample_batch(xTr, yTr, batchsize, i)
			#pdb.set_trace()

			if i%10==0:
				#validation_loss=None
				loss = self.loss.eval(feed_dict={self.x: x_batch, self.y: y_batch, self.dropout: dropout})
				if xVl is not None: #validation loss if validation data provided
					xVl_batch, yVl_batch, xv_batchnames, _ = sample_batch(xVl, yVl, 2*batchsize, i%(int(len(xVl)/2*batchsize))+1)
					validation_loss = self.loss.eval(feed_dict={self.x: xVl_batch, self.y: yVl_batch, self.dropout: 0.0})
					ypred_batch=self.y_hat.eval(feed_dict={self.x: xVl_batch, self.dropout:0})
					#save data

					output_names = dataset.give_outputs(xbatch=xv_batchnames)
					save(ypred_batch, prediction_path, output_names)
					prediction_number+=1
				print ("iteration "+str(i)+": training loss = "+str(loss)+" , validation_loss (prediction "+str(prediction_number)+") : "+str(validation_loss))
				
			train_step.run(feed_dict={self.x: x_batch, self.y: y_batch, self.dropout: dropout})

		if xVl is not None: #validation loss if validation data provided
			xVl_batch, yVl_batch, _, _ = sample_batch(xVl, yVl, 2*batchsize, i%(int(len(xVl)/2*batchsize))+1)
			validation_loss = self.loss.eval(feed_dict={self.x: xVl_batch, self.y: yVl_batch, self.dropout: 0.0})
		print ("iteration "+str(i)+": training loss = "+str(loss)+" , validation_loss : "+str(validation_loss))

		#sess.close()


	def test(self, xVl, yVl, dataset, batchsize=10): #data are file paths
		prediction_path="../data_sample/predictions"
		#xVl, yVl = loadx(xVl), loady(yVl)
		
		#pdb.set_trace()

		n = len(xVl); iterations = int(n/batchsize)+1
		score=0; j=0;
		for i in range(iterations):
			x_batch, y_batch, x_batchnames, _ = sample_batch(xVl, yVl, batchsize, i)
			y_pred=self.y_hat.eval(feed_dict={self.x: x_batch, self.dropout:0})
			score += self.loss.eval(feed_dict={self.x: x_batch, self.y: y_batch, self.dropout:0})
			output_names = dataset.give_outputs(xbatch=x_batchnames)
			save(y_pred, prediction_path, output_names)
			print("test batch "+str(i)+" of "+str(n)+" done.")

		return score/iterations

	def train2(self,data_train, dropout=0.5, batchsize=100, n_training=2000, data_validation=None):
		validation_loss = None
		train_step = tf.train.AdamOptimizer(1e-4).minimize(self.loss) #we use adam method to minimize the loss

		self.session = tf.InteractiveSession()
		self.session.run(tf.global_variables_initializer())
		#pdb.set_trace()
		for i in range(n_training):
			x_batch, y_batch = sample_batch2(data_train, batchsize, i)

			if i%50==0:
				#validation_loss=None
				loss = self.loss.eval(feed_dict={self.x: x_batch, self.y: y_batch, self.dropout: dropout})
				if data_validation is not None: #validation loss if validation data provided
					xVl_batch, yVl_batch = sample_batch2(data_validation, 2*batchsize, i%(int(len(data_validation)/2*batchsize))+1)
					validation_loss = self.loss.eval(feed_dict={self.x: xVl_batch, self.y: yVl_batch, self.dropout: 0.0})
					
				print ("iteration "+str(i)+": training accuracy = "+str(loss)+" , validation accuracy : "+str(validation_loss))
				
			train_step.run(feed_dict={self.x: x_batch, self.y: y_batch, self.dropout: dropout})

		if data_validation is not None: #validation loss if validation data provided
			xVl_batch, yVl_batch = sample_batch2(data_validation, 2*batchsize, i%(int(len(data_validation)/2*batchsize))+1)
			validation_loss = self.loss.eval(feed_dict={self.x: xVl_batch, self.y: yVl_batch, self.dropout: 0.0})
		print ("iteration "+str(i)+": training loss = "+str(loss)+" , validation_loss : "+str(validation_loss))

		

		

	

	def test2(self, data_validation, batchsize=10):
		prediction_path="../data_sample/predictions"
		correct_prediction = tf.equal(tf.argmax(self.y_hat,1), tf.argmax(self.y,1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		
		n = len(data_validation); iterations = int(n/batchsize)+1
		score=0; j=0;acc=0
		for i in range(iterations):
			x_batch, y_batch = sample_batch2(data_validation, batchsize, i)
			#y_pred=self.y_hat.eval(feed_dict={self.x: x_batch, self.dropout:0})
			#predictions = vectorize_labels(sess.run(tf.argmax(self.y_hat,1), feed_dict={self.x: x_batch, self.dropout:0})) 
			#score = self.loss.eval(feed_dict={self.x: x_batch, self.y: y_batch, self.dropout:0})
			acc += accuracy.eval(feed_dict={self.x: x_batch, self.y: y_batch, self.dropout:0})
			
			print("test batch "+str(i)+" of "+str(n)+" done.")
		return acc/iterations
		
		


	def close(self):
		self.session.close()

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
		if activation is not None:
			z = activation(conv + b)
		else:
			z = conv+b

		return z

	@staticmethod
	def connected_layer(x, output_size , activation):
		outsize = output_size
		shape_x = x.get_shape().as_list()
		insize = 1
		for i in range(1,len(shape_x)):
			insize*=shape_x[i] #the first dimention is the number of points so we skip it. Then we nultiply all the remaining ones.

		#in the case of 4 dimentions this means:
		#[batch, in_height, in_width, in_channels]=x.get_shape().as_list()
		#insize = in_height*in_width*in_channels
		x = tf.reshape(x, [-1, insize])
		w = Cnn.weight_variable([insize, outsize])
		b = Cnn.bias_variable([outsize])
		
		if activation is not None:
			z = activation(tf.matmul(x, w) + b)
		else:
			z=tf.matmul(x, w) + b
		return z

	@staticmethod
	def pooling_layer(x, pool_height, pool_width):
		z = tf.nn.max_pool(x, ksize=[1, pool_height, pool_width, 1],
                        strides=[1, pool_height, pool_width, 1], padding='SAME')
		return z

	@staticmethod
	def dropout_layer(x, dropout_tensor):
		z = tf.nn.dropout(x, 1-dropout_tensor) 
		return z

	@staticmethod
	def reshape(shape):
		return [s if s is not None else -1 for s in shape]





	




