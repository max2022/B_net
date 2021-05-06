#!/bin/sh
mkdir -p datasets/data/pcifar10
cd datasets/data/pcifar10
# or download from here https://drive.google.com/file/d/0Byyuc5LmNmJPWUc5dVdUSms3U1E/view?usp=sharing
echo "Please download data from https://drive.google.com/file/d/0Byyuc5LmNmJPWUc5dVdUSms3U1E/view?usp=sharing and put it at datasets/data/pcifar10/data.npz"
# wget 'https://dl.dropboxusercontent.com/u/4542002/branchynet/data.npz' -O data.npz


echo "please download mnist data from https://github.com/amplab/datascience-sp14/blob/master/lab7/mldata/mnist-original.mat"

echo "use this python command to put mnist data to this directory"
echo "from sklearn.datasets import get_data_home"
echo "print(get_data_home())"
