#!/usr/bin/python
import os, sys, pdb
import numpy as np
sys.path.insert(0, os.path.abspath("../dataio/"))
from dataio import *
import matplotlib.pyplot as plt
import tensorflow as tf
from cnn import *

computer_transition_path="../data_sample/predictions/prediction_4/"
human_transition_path="../data_sample/pprocessed_data"
data_comp =  [os.path.join(computer_transition_path,pred) for pred  in os.listdir(computer_transition_path)]
#predictions_numbers = [path.split("_")[1] for path in data_comp]

n = len(data_comp)
dataset = Dataset(human_transition_path)
data_hum= dataset.y

data_hum=data_hum[np.random.permutation(np.size(data_hum))]
data_hum=data_hum[:n]

data = np.concatenate((data_hum, data_comp))

ratio = 0.8
n = len(data)
shuffle = np.random.permutation(n)
data = data[shuffle]

data_train, data_val = data[:int(ratio*n)], data[int(ratio*n)+1:]




dx = 39*1293
dy = 2
#x_shape = [batch, height, width, channels] 
#y_shape=[batch, height, width, channels]
#network_specs = #[(['conv', ('height', 'width', 'depth'), 'activation'], number), (['connected', (outsize), 'activation'], number)]
network_specs = [
				 (['conv',(1, 1, 1), tf.nn.relu], 1),
				 (['pooling',(8,16), None], 1),
				 (['connected', (200), tf.nn.relu], 1),
				 (['connected', (dy), None], 1)] #layer 2: convolution layer with a filter of 1*25 and a depth of 64 using relu as an activation function
				 #(['connected', (dy), None], 1), #layer 3: fully connected layer with output size dy and relu activation function
				 #(['dropout',   None, None],     1)

##data_specs = (x_shape, y_shape, xnetwork_shape, ynetwork_shape)
data_specs = ([None, dx], [None, dy], [None,39,1293,1], [None,dy])
cnn = Cnn(data_specs, network_specs)
cnn.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=cnn.y, logits=cnn.y_hat)) #we use the cross entropy loss

#cnn.build_network(xTr, yTr, batchsize=50, n_training=1)
cnn.train2(data_train, batchsize=100, n_training=100, data_validation=data_val)
#pdb.set_trace()
accuracy = cnn.test2(data_val, batchsize=10)

print("Accuracy on validation set : "+str(accuracy))

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
