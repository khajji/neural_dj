#!/usr/bin/python
import os, sys, pdb
import numpy as np
sys.path.insert(0, os.path.abspath("../dataio/"))
from dataio import *
import matplotlib.pyplot as plt
import tensorflow as tf
from cnn import *

data_path="../data_sample/pprocessed_data"
prediction_path = "../data_sample/predictions"
#pdb.set_trace()
dataset = Dataset(data_path)
xTr, yTr, xVl, yVl = dataset.split(ratio=0.8)
dx =39*2586*2
dy = 39*1293
#x_shape = [batch, height, width, channels] 
#y_shape=[batch, height, width, channels]
#network_specs = #[(['conv', ('height', 'width', 'depth'), 'activation'], number), (['connected', (outsize), 'activation'], number)]
network_specs = [
				 (['conv',(13, 5, 16), tf.nn.relu], 1), #layer 1: convolution layer with a filter of 1*25 and a depth of 32 using relu as an activation function
				 (['conv',(13, 5, 16), tf.nn.relu], 1),
				 (['pooling',(1, 2), None], 1),
				 (['conv',(13, 5, 32), tf.nn.relu], 1),
				 (['conv',(13, 5, 32), tf.nn.relu], 1),
				 (['pooling',(1,2), None], 1),
				 (['conv',(13, 5, 16), tf.nn.relu], 1),
				 (['conv',(13, 5, 16), tf.nn.relu], 1),
				 (['conv',(13,5, 3), None], 1)
				 ]

				  #VG net convolution layer with a filter of 1*25 and a depth of 64 using relu as an activation function
				

data_specs = ([None, dx], [None, dy], [None,13, 2586*2,3], [None,13, 1293,3])
cnn = Cnn(data_specs, network_specs)

#cnn.build_network(xTr, yTr, batchsize=50, n_training=1)
cnn.train(xTr, yTr,dataset, batchsize=10, n_training=2000, xVl=xVl, yVl=yVl)

loss = cnn.test(xVl, yVl,dataset, batchsize=10)

print("loss on validation set : "+str(loss))

#output_names = dataset.give_outputs()
#save(y_pred, prediction_path, output_names)



'''
network_specs = [
				 (['conv',(1, 25, 32), tf.nn.relu], 1), #layer 1: convolution layer with a filter of 1*25 and a depth of 32 using relu as an activation function
				 (['conv', (1,25,64), tf.nn.relu], 1), #layer 2: convolution layer with a filter of 1*25 and a depth of 64 using relu as an activation function
				 (['connected', (dy), tf.nn.relu], 1), #layer 3: fully connected layer with output size dy and relu activation function
				 (['dropout',   None, None],     1)]

data_specs = ([None, dx], [None, dy], [None,1,dx,1], [None,1,dy,1])
'''
