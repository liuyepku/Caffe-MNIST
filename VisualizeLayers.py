import sys
import caffe
import numpy as np
import matplotlib as plt
from VisualizeWeights import *

path = '/home/mozat/git/caffe'



#load model
net = caffe.Net(path + '/examples/mnist/lenet.prototxt', path + '/examples/mnist/lenet_iter_10000.caffemodel', caffe.TRAIN)
visualize_weights(net, 'conv1', filename='conv1.png')
visualize_weights(net, 'conv2', filename='conv2.png')
# print net.params['conv1'][0].data.shape
print callable(visualize_weights)