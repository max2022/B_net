# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 15:18:43 2021

@author: hasan
"""

from branchynet.links import *
import chainer.functions as F
import chainer.links as L
from branchynet import utils, visualize
from chainer import cuda
from networks import lenet_mnist

branchyNet = lenet_mnist.get_network()
if cuda.available:
    branchyNet.to_gpu()
branchyNet.training()

from datasets import mnist
from sklearn.datasets import fetch_openml
import numpy as np

def get_data():
    mnist = fetch_openml('mnist_784')
    x_all = mnist['data'].astype(np.float32) / 255
    y_all = mnist['target'].astype(np.int32)
    x_train, x_test = np.split(x_all, [60000])
    y_train, y_test = np.split(y_all, [60000])

    x_train = x_train.reshape([-1,1,28,28])
    x_test = x_test.reshape([-1,1,28,28])
    return x_train,y_train,x_test,y_test

x_train, y_train, x_test, y_test = get_data()

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
TRAIN_BATCHSIZE = 512
TEST_BATCHSIZE = 1

#branchynet is 100
TRAIN_NUM_EPOCHS = 100
import dill
branchyNet = None
with open("_models/lenet_mnist.bn", "rb") as f:
    branchyNet = dill.load(f)

branchyNet.testing()
branchyNet.verbose = False
thresholds = [0.025]
if cuda.available:
    branchyNet.to_gpu()
g_ts, g_accs, g_diffs, g_exits = utils.screen_branchy(branchyNet, x_test, y_test, thresholds, batchsize=TEST_BATCHSIZE, verbose=True)