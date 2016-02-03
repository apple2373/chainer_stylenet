#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Worning! Be sure to use chainer 1.3.2
this code does not work in chainer 1.6
(I didn't test with from 1.4 to 1.5)

This is a quick implementation of http://arxiv.org/abs/1508.06576
There is a hard coding part. The computation of L_style in forward()
This part is coresponding to equation (4) and (5).
I did not paramatalized it because it would be too complizated. 
I think defalt is fine, but if you want, you can easily change it directly. See L_style in forward().

I recommend to make content_image a squared size.
However, you can use rectangular one, but the output will be forsed to a square.
You need to resize agian, ex,
img = imread('filename.png')
img = imresize(img,[400,300])

Basic settings is configured, in #Settings part, which is just after the import sentences.
20000 iteraton will be done. Image is saved every 500 iteration.
'''

import argparse
import os
import numpy as np
from chainer import cuda, Function, FunctionSet, gradient_check, Variable, optimizers
import chainer.functions as Fu
from chainer.functions import caffe
import chainer
import matplotlib.pyplot as plt
from scipy.misc import imread, imresize, imsave

#Settings
gpu_id=-1# GPU ID. if you want to use cpu, -1
content_image='content.png' #name of content image 
style_image='style.png' #name of style image
savedir='results'# name of log and results image saving directory
a_p_ratio = 0.001 #alpha/beta in equation (7) in the original paper

#Override Settings by argument
parser = argparse.ArgumentParser(description=u"Style Net Parser")
parser.add_argument("-g", "--gpu",default=gpu_id, type=int, help=u"GPU ID.CPU is -1")
parser.add_argument("-c", "--content",default=content_image, type=str, help=u"a content image name")
parser.add_argument("-s", "--style",default=style_image, type=str, help=u"a style image name ")
parser.add_argument("-d", "--directory",default=savedir, type=str, help=u"a directory name to save log and results images")
args = parser.parse_args()
 
gpu_id=args.gpu
content_image=args.content
style_image=args.style 
savedir=args.directory


if gpu_id >= 0:
    cuda.get_device(gpu_id).use()
    xp = cuda.cupy
else:
    xp = np

if os.path.isdir(savedir)== False:
    os.mkdir(savedir)

#Funcations
def readimage(filename):
    img = imread(filename)
    img = imresize(img,[224, 224])
    img = np.transpose(img,(2,0,1))
    img = img.reshape((1,3,224,224))
    p_data = np.ascontiguousarray(img,dtype=np.float32)
    if gpu_id >=0:
        p = Variable(cuda.to_gpu(p_data))
    else:
        p = Variable(p_data)
    return p

def reshape2(conv1_1):
    k=conv1_1.data.shape[1]
    pixels=conv1_1.data.shape[2]*conv1_1.data.shape[3]
    return chainer.functions.reshape(conv1_1,(k,pixels))

# save the original image
def save_x(img,filename="output.png"):
    img = img.reshape((3,224,224))
    img = np.transpose(img,(1,2,0))
    imsave(filename,img)

def compute_A_P(a,p):
    #compute matrix P 
    conv1_1,conv2_1, conv3_1, conv4_1,conv5_1, = func(inputs={'data': p}, outputs=['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1'])
    P = [ reshape2(x) for x in [conv1_1,conv2_1, conv3_1, conv4_1,conv5_1]]
    #compute matrix A
    conv1_1,conv2_1, conv3_1, conv4_1,conv5_1, = func(inputs={'data': a}, outputs=['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1'])
    conv1_1A0,conv2_1A0, conv3_1A0, conv4_1A0,conv5_1A0, = [ reshape2(x) for x in [conv1_1,conv2_1, conv3_1, conv4_1,conv5_1]]
    A = [ Fu.matmul(x, x, transa=False, transb=True) for x in [conv1_1A0,conv2_1A0, conv3_1A0, conv4_1A0,conv5_1A0]]

    return A,P

def forward(x, p, a, A=None,P=None):

    conv1_1, conv2_1, conv3_1, conv4_1,conv5_1, = func(inputs={'data': x}, outputs=['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1'])
    conv1_1F,conv2_1F, conv3_1F, conv4_1F,conv5_1F, = [ reshape2(x) for x in [conv1_1,conv2_1, conv3_1, conv4_1,conv5_1]]
    conv1_1G,conv2_1G, conv3_1G, conv4_1G,conv5_1G, = [ Fu.matmul(x, x, transa=False, transb=True) for x in [conv1_1F,conv2_1F, conv3_1F, conv4_1F,conv5_1F]]
    
    # Because P an A is not change over iteration, it's better to calcurate onece.
    if A is None and B is None:
        #compute matrix P 
        conv1_1,conv2_1, conv3_1, conv4_1,conv5_1, = func(inputs={'data': p}, outputs=['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1'])
        conv1_1P,conv2_1P, conv3_1P, conv4_1P,conv5_1P, = [ reshape2(x) for x in [conv1_1,conv2_1, conv3_1, conv4_1,conv5_1]]
        #compute matrix A
        conv1_1,conv2_1, conv3_1, conv4_1,conv5_1, = func(inputs={'data': a}, outputs=['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1'])
        conv1_1A0,conv2_1A0, conv3_1A0, conv4_1A0,conv5_1A0, = [ reshape2(x) for x in [conv1_1,conv2_1, conv3_1, conv4_1,conv5_1]]
        conv1_1A,conv2_1A, conv3_1A, conv4_1A,conv5_1A, = [ Fu.matmul(x, x, transa=False, transb=True) for x in [conv1_1A0,conv2_1A0, conv3_1A0, conv4_1A0,conv5_1A0]]
    else:
        conv1_1P,conv2_1P, conv3_1P, conv4_1P,conv5_1P,=P
        conv1_1A,conv2_1A, conv3_1A, conv4_1A,conv5_1A,=A

    L_content = Fu.mean_squared_error(conv4_1F,conv4_1P)/2

    #caution! the deviding number is hard coding!
    #this part is correspnding to equation (4) in the original paper
    #to check the current N and M, run the following
    #[x.data.shape  for x in [conv1_1F,conv2_1F, conv3_1F, conv4_1F,conv5_1F]]
    L_style = (Fu.mean_squared_error(conv1_1G,conv1_1A)/(4*64*64*50176*50176)
    + Fu.mean_squared_error(conv2_1G,conv2_1A)/(4*128**128*12544*12544)
    + Fu.mean_squared_error(conv3_1G,conv3_1A)/(4*256*256*3136*3136)
    + Fu.mean_squared_error(conv4_1G,conv4_1A)/(4*512*512*784*784)\
    )/4 # this is equal weighting of E_l

    loss = a_p_ratio*L_content + L_style
    return loss 

#main

p=readimage(content_image)#read content image
a=readimage(style_image)#read style image

print "Loading caffe model.It takes time...."
func = caffe.CaffeFunction('VGG_ILSVRC_19_layers.caffemodel')
if gpu_id >= 0:
    func.to_gpu()
print "....fhinish loading!"


x_data=xp.random.randn(1,3,224,224).astype(np.float32)
x = Variable(x_data)

#x = readimage('imge230.png') # if you want to start from a exsiting image

#optimize x(=image) with adam

alpha=1
beta1=0.9
beta2=0.999
eps=1e-8

v=xp.zeros_like(x.data)
m=xp.zeros_like(v)

A,P=compute_A_P(a,p)# for efficient computation!
print "Optimization stated. See the log file to see the progress"
for epoch in xrange(0,20000):
    t=0
    loss=forward(x,p,a,A,P)
    recompute=False
    loss.backward()
    grad=x.grad.copy()
    t +=1
    m =  beta1*m + (1-beta1)*grad
    v =  beta2*v + (1-beta2)*(grad*grad)
    m_hat=m/(1-np.power(beta1,t))
    v_hat=v/(1-np.power(beta2, t))
    x.data -= alpha * m_hat / (xp.sqrt(v_hat) + eps)
    
    grad_norm=xp.sqrt(xp.sum(grad*grad))
    with open(savedir+"/log.txt", "a") as f:
        f.write(str(epoch)+','+str(loss.data)+','+str(grad_norm)+'\n')
    if epoch%500==0:
        savename = savedir+'/img'+str(epoch)+'.png'
        save_x(cuda.to_cpu(x.data),savename)