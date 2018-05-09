import torch
from torch import Tensor,FloatTensor, LongTensor
import numpy
from math import sqrt, pi   
import matplotlib.pyplot as plt
from plot import plot_data

train_data = Tensor(10,2).uniform_()
train_target = LongTensor([int(sqrt((train_data[i][0])**2 + (train_data[i][1])**2) < 1/sqrt(2*pi)) for i in range(len(train_data))])

plot_data(train_data, train_target)
