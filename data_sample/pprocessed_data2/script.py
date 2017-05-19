#!/usr/bin/python
import sys
import os
import numpy as np
sys.path.insert(0, os.path.abspath("../../../dataio/"))
from dataio import *
x1=io.loadmat("../../pprocessed_data/6/x.mat")['x']
x2=io.loadmat("../prediction_3/yhat6.mat")['y']

xconc=np.concatenate((x1,x2))
plot_images(xconc)
