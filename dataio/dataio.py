#Main functionlalities:
#1- read write an mp3
#2- split data test and training set
#3- split validation and training
from pydub import AudioSegment
import os
for os import path
import numpy as np


def load(files):
	data = []
	for f is files:
		x=io.loadmat(f)
		data+=[x]
	return np.matrix(data)

def save(x, to, names):
	to=path.abspath(to)
	folder_prefix="prediction_"
	splitchar="_"
	#find the next folder name to write the predictions in and create it
	prediction_number = max([int(os.path.basename(d).split(splitchar)[1]) for d in os.listdir(to) if folder_prefix in d])+1
	out_path = path.join(to, folder_prefix+str(prediction_number))
	if not os.path.exists(out_path):
		os.makedirs(out_path)
	else:
		raise Exception(out_path +' already exists!')

	#write the predictions in the create folder
	(n,d)=x.shape()
	for i in range(n):
		io.savemat(path.join(out_path, names[i]), {x[i,:]})
	return True


def sample_batch(xp, yp, batchsize, i):
	n = np.shape(x) #n is the number of training points. d is the dimention. 
	start = (batchsize*slice_number)%n
	end = min(start+size, n)
	x_batch,y_batch = load_group(x[start:end]), load_group(y[start:end])
	return x_batch, y_batch
	 


class Dataset: #For this class a data point is string representing a file path. Xi is a file path, Yi is a file path. 
	xs="x"
	ys="y"
	yhat="yhat"

	def __init__(path=None):
		self.dataset=path.abspath(path)
		self.x = [path.join(d,Dataset.xs) for d in data]
		self.y = [path.join(d,Dataset.ys) for d in data]
		self.xTr, self.yTr, self.xVl, self.yVl  = None, None, None, None
		

	def split(ratio=0.8):  #splits and returns a list of file paths
		data=np.array[os.listdir(self.dataset)]
		n = np.size(data)
		shuffle = np.random.permutation(n) #randomly shuffle the data
		x=self.x[suffle]; y=self.y[suffle];

		self.xTr, self.yTr, self.xVl, self.yVl = x[:int(ratio*n)], y[:int(ratio*n)], x[int(ratio*n)+1,:], y[int(ratio*n)+1,:]
		
		return self.xTr, self.yTr, self.xVl, self.yVl

	def give_outputs():
		return [Dataset.yhat+os.path.basename(os.path.dirname(x)) for x in self.xVl]
		




