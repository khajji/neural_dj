#!/usr/bin/python
import os, sys, pdb
import numpy as np
sys.path.insert(0, os.path.abspath("../dataio/"))
from dataio import *
import matplotlib.pyplot as plt
import tensorflow as tf
from cnn import *

data_path="../data_sample/pprocessed_data2"
prediction_path = "../data_sample/predictions"
#pdb.set_trace()
dataset = Dataset(data_path)
xTr, yTr, xVl, yVl = dataset.split(ratio=0.8)
dx = 784
dy = 2
#x_shape = [batch, height, width, channels] 
#y_shape=[batch, height, width, channels]
#network_specs = #[(['conv', ('height', 'width', 'depth'), 'activation'], number), (['connected', (outsize), 'activation'], number)]
network_specs = [
				 (['conv',(5, 5, 16), tf.nn.relu], 1),
				 (['pool',(2,2), None], 1), #layer 1: convolution layer with a filter of 1*25 and a depth of 32 using relu as an activation function
				 (['conv',(5, 5, 64), tf.nn.relu], 1),
				 (['pool',(2,2), None], 1),
				 (['conv', (5,5,1), tf.nn.relu], 1),
				 (['pool',(2,2), None], 1),
				 (['connected', (1024), tf.nn.relu], 1),
				 (['dropout',   None, None], 1),
				 (['connected', (dy), None], 1)] #layer 2: convolution layer with a filter of 1*25 and a depth of 64 using relu as an activation function
				 #(['connected', (dy), None], 1), #layer 3: fully connected layer with output size dy and relu activation function
				 #(['dropout',   None, None],     1)
h_pool1 = max_pool_2x2(h_conv1)
##data_specs = (x_shape, y_shape, xnetwork_shape, ynetwork_shape)
data_specs = ([None, dx], [None, dy], [None,28,28,1], [None,dy])
cnn = Cnn(data_specs, network_specs)
cnn.loss = tf.reduce_mean(
    	tf.nn.softmax_cross_entropy_with_logits(labels=cnn.yhat, logits=cnn.y)) #we use the cross entropy loss

#cnn.build_network(xTr, yTr, batchsize=50, n_training=1)
cnn.train(xTr, yTr, batchsize=100, n_training=2000, xVl=xVl, yVl=yVl)

y_pred, loss = cnn.test(xVl, yVl)

print("loss on validation set : "+str(loss))

output_names = dataset.give_outputs()
save(y_pred, prediction_path, output_names)



'''
network_specs = [
				 (['conv',(1, 25, 32), tf.nn.relu], 1), #layer 1: convolution layer with a filter of 1*25 and a depth of 32 using relu as an activation function
				 (['conv', (1,25,64), tf.nn.relu], 1), #layer 2: convolution layer with a filter of 1*25 and a depth of 64 using relu as an activation function
				 (['connected', (dy), tf.nn.relu], 1), #layer 3: fully connected layer with output size dy and relu activation function
				 (['dropout',   None, None],     1)]

data_specs = ([None, dx], [None, dy], [None,1,dx,1], [None,1,dy,1])
'''
