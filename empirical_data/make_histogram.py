from random import *
from math import *
from scipy import stats
import time
import os
import numpy as np
from numpy.linalg import norm
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as img

#read in empirical data from Erik Brookman
dfile=open("abortion.txt","r")
dataset="abortion"
raw_data=dfile.read()
dfile.close()
brookman_data=[]
for line in raw_data.split('\n'):
	value=int(line.split(',')[0])
	for counts in range(int(line.split(',')[1])):
		brookman_data.append(value)

#plot the histogram
ohistplot=plt.figure()
a=ohistplot.add_subplot(111)
weights=100*np.ones_like(brookman_data)/len(brookman_data) #normalizes properly, convert to %
print weights
plt.hist(brookman_data,bins=7,weights=weights,normed=False)
plt.xlabel("Opinion")
plt.ylabel("Percent")
#plt.xlim([0,100])
# plt.ylim([0,100])
plt.title("Opinions about Abortion")
ohistplot.savefig("Empirical Abortion.png")