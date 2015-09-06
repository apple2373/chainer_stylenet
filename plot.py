#!/usr/bin/env python
# coding: UTF-8

import argparse

import numpy as np
import matplotlib.pylab as plt

#Override Settings by argument
parser = argparse.ArgumentParser(description=u"plot log by style net")
parser.add_argument("directory", type=str, help=u"the name of directory")
args = parser.parse_args()
 
log = np.loadtxt(open(args.directory+'/log.txt','rb'),delimiter=',',skiprows=0)


# 左
plt.plot(log[:, 1],label = "loss")
#plt.plot(log[:, 2],label = "|grad|")
plt.legend()

#plt.tight_layout() 
plt.show()