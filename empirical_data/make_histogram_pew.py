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

#read in empirical data from Pew Reserach Center
dfile=open("PewData2014.txt","r")
dataset="PewData2014"
raw_data=dfile.read()
dfile.close()
pew_data=[]
for line in raw_data.split('\n'):
	value=int(line.split(',')[0])
	for counts in range(int(100*float(line.split(',')[1]))):
		pew_data.append(value)
# print pew_data

#plot the histogram
ohistplot=plt.figure()
a=ohistplot.add_subplot(111)
# weights=100*np.ones_like(pew_data)/len(pew_data) #normalizes properly, convert to %
weights=[0.01 for i in range(len(pew_data))] #dunno why the above line fails
plt.hist(pew_data,weights=weights,bins=21,normed=False)
plt.xlabel("Opinion")
plt.ylabel("Percent")
#plt.xlim([0,100])
plt.ylim([0,10])
plt.title("Ideological Consistency")
ohistplot.savefig("Empirical PewData2014.png")