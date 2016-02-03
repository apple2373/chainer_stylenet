# Stylenet

This is a quick implementation of http://arxiv.org/abs/1508.06576

Blog post: http://t-satoshi.blogspot.com/2015/09/a-neural-algorithm-of-artistic-style.html

## Requirements
chainer (1.3.2)
http://chainer.org
And pre-tarined caffe VGG model

Warning! Be suer to use chainer 1.3.2!!  
This code does not work in chainer 1.6 (I didn't test with from 1.4 to 1.5)  

## Quick Usage
install Anaconda (https://www.continuum.io/downloads) and then, 
```
pip install chainer==1.3.2
wget http://www.robots.ox.ac.uk/%7Evgg/software/very_deep/caffe/VGG_ILSVRC_19_layers.caffemodel
python style_net.py -g -1 -c kinkaku.jpg -s style.png -d kinkaku
```
if you want to use GPU (recommended). 0 is GPU ID.
```
python style_net.py -g 0 -c kinkaku.jpg -s style.png -d kinkaku
```

## Comment
There is a hard coding part. The computation of L_style in forward()
This part is coresponding to equation (4) and (5).
I did not paramatalized it because it would be too complizated. 
I think defalt is fine, but if you want, you can easily change it directly. See L_style in forward().

I recommend to make content image a squared size.
However, you can use rectangular one, but the output will be forsed to a square.
You need to resize agian, ex in python,
```
from scipy.misc import imread, imresize
img = imread('filename.png')
img = imresize(img,[400,300])
```

Basic settings is configured, in #Settings part, which is just after the import sentences.
20000 iteraton will be done. Image is saved every 500 iteration.
