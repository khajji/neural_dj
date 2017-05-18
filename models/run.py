#!/usr/bin/python
import os, sys, pdb
import numpy as np
sys.path.insert(0, os.path.abspath("../data_processing/"))
from dataio import *
import matplotlib.pyplot as plt
import tensorflow as tf

dataset = Dataset("my_path")
xTr, yTr, xVl, yVl = dataset.split(ratio=0.8)
dx = 1000
dy = 100
#x_shape = [batch, height, width, channels] 
#y_shape=[batch, height, width, channels]
#network_specs = #[(['conv', ('height', 'width', 'depth'), 'activation'], number), (['connected', (outsize), 'activation'], number)]
network_specs = [
				 (['conv',(1, 25, 32), tf.nn.relu], 1), #layer 1: convolution layer with a filter of 1*25 and a depth of 32 using relu as an activation function
				 (['conv', (1,25,64), tf.nn.relu], 1), #layer 2: convolution layer with a filter of 1*25 and a depth of 64 using relu as an activation function
				 (['connected', (dy), tf.nn.relu], 1) #layer 3: fully connected layer with output size dy and relu activation function
				 ]
cnn = Cnn([None, 1, dx, 1], [None, 1, dy, 1], network_specs)

cnn.train(xTr, yTr, batchsize=50, n_training=10000)
y_pred, loss = cnn.test(xVl, yVl)

print("loss on validation set : "+str(loss))

output_names = dataset.give_outputs()
save(y_pred, "output_path", output_names)
