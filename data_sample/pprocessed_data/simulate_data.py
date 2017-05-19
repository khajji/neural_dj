import os
from os import path
import numpy as np
from scipy import io
import scipy.ndimage as scim
import scipy

def rotate(vec): #applay a rotation and a translation
	angle_window = 45 #generate uniformly an angle between [-angle_window, angle_window] (in degrees)
	d = np.size(vec)
	xdim=int(np.sqrt(d)); ydim=int(np.sqrt(d))
	angle = int(np.random.rand()*(angle_window*2)+1)-angle_window #sample uniformly an angle 
	im = vec.reshape(ydim, xdim)
	rot_im = scim.interpolation.rotate(im, angle, reshape=False)
	rot_vec = rot_im.reshape((xdim*ydim,))
	
	return rot_vec

data=io.loadmat("train.mat")
data = data['x']
data = np.matrix(data)
(n,d) = np.shape(data)


#data = data[np.random.permutation(n), :]
xd = data
yd = np.apply_along_axis(rotate, 1, xd)


for i in range(n):
	x = xd[i,:]
	y = yd[i+int(n/2), :]
	os.makedirs(str(i))
	io.savemat(path.join(str(i), 'x.mat'), {'x':x})
	io.savemat(path.join(str(i), 'y.mat'), {'y':y})


'''
for i in range(int(n/2)):
	x = data[i,:]
	y = data [i+int(n/2), :]
	os.makedirs(str(i))
	io.savemat(path.join(str(i), 'x.mat'), {'x':x})
	io.savemat(path.join(str(i), 'y.mat'), {'y':y})
'''