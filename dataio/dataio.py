import os
import numpy as np
import pdb
from scipy import io
import matplotlib.pyplot as plt

def loadx(files):
	return load(files, 'x')

def loady(files):
	return load(files, 'x')

def load(files, key):
	data = None
	#pdb.set_trace()
	for f in files:
		x=io.loadmat(f)[key]
		(n,d)=np.shape(x)
		x=x[:,:min(d,2588)].reshape(1,-1)
		data = x if data is None else np.concatenate((data,x)) 
	return np.matrix(data)

def save(x, to, names):
	to=os.path.abspath(to)
	folder_prefix="prediction_"
	splitchar="_"
	#find the next folder name to write the predictions in and create it
	prediction_number = max([int(os.path.basename(d).split("_")[1]) for d in os.listdir(to) if "prediction"in d]+[0])+1
	out_path = os.path.join(to, folder_prefix+str(prediction_number))
	if not os.path.exists(out_path):
		os.makedirs(out_path)
	else:
		raise Exception(out_path +' already exists!')

	#write the predictions in the create folder
	(n,d)=np.shape(x)
	for i in range(n):
		io.savemat(os.path.join(out_path, names[i]), {'x':x[i,:], 'y': 0})
	return True


def sample_batch(xp, yp, batchsize, slice_number):
	#pdb.set_trace()
	n = np.size(xp) #n is the number of training points. d is the dimention. 
	start = (batchsize*slice_number)%n
	end = min(start+batchsize, n)
	x_batch,y_batch = loadx(xp[start:end]), loady(yp[start:end])
	return x_batch, y_batch, xp[start:end], yp[start:end]

def plot_images(X):
	xdim=28; ydim=28 
	n, d = X.shape
	f, axarr = plt.subplots(1, n, sharey=True)
	f.set_figwidth(10 * n)
	f.set_figheight(n)
	
	if n > 1:
		for i in range(n):
			axarr[i].imshow(X[i, :].reshape(ydim, xdim), cmap=plt.cm.binary_r)
	else:
		axarr.imshow(X[0, :].reshape(ydim, xdim), cmap=plt.cm.binary_r)
	plt.show()

def sample_batch2(data, batchsize, i):
	n = np.size(data) #n is the number of training points. d is the dimention. 
	start = (batchsize*slice_number)%n
	end = min(start+batchsize, n)
	x_batch,y_batch = load2(data[start:end])
	return x_batch,y_batch

def load2(files):
	data_x = None
	data_y = None
	#pdb.set_trace()
	for f in files:
		x=io.loadmat(f)['x']
		(n,d)=np.shape(x)
		x=x[:,:min(d,2588)].reshape(1,-1)
		data_x = x if data_x is None else np.concatenate((data_x,x))

		y=io.loadmat(f)['y']
		yarr = np.array([0,0]); yarr[y]=1
		data_y = yarr if data_y is None else np.concatenate((data_y,y))

	return np.matrix(data_x), np.matrix(data_y)
	 


class Dataset: #For this class a data point is string representing a file path. Xi is a file path, Yi is a file path. 
	xs="x"
	ys="y"
	yhat="yhat"

	def __init__(self, path=None):
		self.dataset=os.path.abspath(path)
		data=[]
		for d in os.listdir(self.dataset):
			d = os.path.join(self.dataset, d)
			if os.path.isdir(d) and ".DS" not in d:
				data += [d]
		#pdb.set_trace()
		
		self.x = np.array([os.path.join(d,Dataset.xs) for d in data])
		self.y = np.array([os.path.join(d,Dataset.ys) for d in data])
		self.xTr, self.yTr, self.xVl, self.yVl  = None, None, None, None
		

	def split(self, ratio=0.8):  #splits and returns a list of file paths
		n = np.size(self.x)
		shuffle = np.random.permutation(n) #randomly shuffle the data
		
		x=self.x[shuffle]; y=self.y[shuffle];

		self.xTr, self.yTr, self.xVl, self.yVl = x[:int(ratio*n)], y[:int(ratio*n)], x[int(ratio*n)+1:], y[int(ratio*n)+1:]
		
		return self.xTr, self.yTr, self.xVl, self.yVl

	def give_outputs(self, xbatch=None):
		if xbatch is None:
			batch_paths = self.xVl
		else:
			batch_paths = []
			for x in xbatch:
				if x in self.xVl:
					batch_paths+=[x]

		return [Dataset.yhat+"_"+os.path.basename(os.path.dirname(x)) for x in batch_paths]
		




