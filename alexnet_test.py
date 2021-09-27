

from branchynet.net import BranchyNet
from branchynet.links import *
import chainer.functions as F
import chainer.links as L
from branchynet import utils, visualize
from chainer import cuda
#from tensorflow import keras
from networks import alex_cifar10

branchyNet = alex_cifar10.get_network()
if cuda.available:
    branchyNet.to_gpu()
branchyNet.training()

from datasets import cifar10gcn
x_train,y_train,x_test,y_test = cifar10gcn.get_data()

#from datasets import pcifar10
#x_train,y_train,x_test,y_test = pcifar10.get_data()

training_size=10000
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
x_train, y_train, x_test, y_test = x_train[:training_size*5], y_train[:training_size*5], x_test[:training_size], y_test[:training_size]
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
#print(x_test)



TEST_BATCHSIZE = 1

##Cpu Utilization
import psutil
import time

def calculate(t1, t2):
    # from psutil.cpu_percent()
    # see: https://github.com/giampaolo/psutil/blob/master/psutil/__init__.py
    t1_all = sum(t1)
    #print(t1_all)
    t1_busy = t1_all - t1.idle
    t2_all = sum(t2)
    t2_busy = t2_all - t2.idle
    if t2_busy <= t1_busy:
        return 0.0
    busy_delta = t2_busy - t1_busy
    all_delta = t2_all - t1_all
    #print(busy_delta)
    busy_perc = (busy_delta / all_delta) * 100
    return round(busy_perc, 1)


cpu_time_a = (time.time(), psutil.cpu_times())

import dill
branchyNet = None
with open("_models/alexnet_cifar10.bn", "rb") as f:
    branchyNet = dill.load(f)


#set network to inference mode, this is for measuring baseline function. 
branchyNet.testing()
branchyNet.verbose = True
thresholds = [[0.0001, 0.05]]
if cuda.available:
    branchyNet.to_gpu()
g_ts, g_accs, g_diffs, g_exits = utils.screen_branchy(branchyNet, x_test, y_test, thresholds,batchsize=TEST_BATCHSIZE, enumerate_ts=False, verbose=True)

cpu_time_b = (time.time(), psutil.cpu_times())
t = cpu_time_b[0] - cpu_time_a[0]
x = calculate(cpu_time_a[1], cpu_time_b[1])
print('Cpu used in seconds ',t)
print('CPU usage in Percentage ',x)
#print(g_accs)
#print(g_diffs)
#print(g_ts)
#print(g_exits)