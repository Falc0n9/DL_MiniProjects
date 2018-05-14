import torch
from torch import Tensor,FloatTensor, LongTensor
from mini_project_2 import *
import numpy
from math import sqrt, pi
import matplotlib.pyplot as plt
from plot import plot_data
from dlc_practical_prologue import convert_to_one_hot_labels

train_data = Tensor(10000,2).uniform_()
train_target = LongTensor([int(sqrt((train_data[i][0])**2 + (train_data[i][1])**2) < 1/sqrt(2*pi)) for i in range(len(train_data))])
train_target_one_hot = convert_to_one_hot_labels(train_data,train_target)

train_data_for_plot = train_data
train_data = train_data.pow(2).sum(1).unsqueeze(0).t()

#print(train_data)
#plot_data(train_data, train_target)


fc_input = Linear(1, 2)
relu1 = Tanh()
fc_hidden_1 = Linear(2, 25)
relu2 = Tanh()
fc_hidden_2 = Linear(25, 25)
relu3 = Tanh()
fc_hidden_3 = Linear(25, 25)
relu4 = Tanh()
fc_output = Linear(25, 2)

seq = Sequential(fc_input,
                 relu1,
                 fc_hidden_1,
                 relu2,
                 fc_hidden_2,
                 relu3,
                 fc_hidden_3,
                 relu4,
                 fc_output)

lr = 0.001
nb_epochs = 50
batch_size = 10

loss_list = []

for i in range(0,nb_epochs):
    for b in range(0, train_data.size(0), batch_size):

        input = train_data.narrow(0,b,batch_size)
        target = train_target_one_hot.narrow(0,b,batch_size)

        seq.zero_grad()
        loss, loss_grad = loss_mse(seq.forward(input), target)
        seq.backward(loss_grad)
        for param in seq.param():
            param.data -= lr * param.grad
    loss_list += [loss]

plt.plot(loss_list)
plt.show()

output = seq.forward(train_data)
_, predicted_classes = output.max(1)

plot_data(train_data_for_plot,predicted_classes)

#for param in seq.param():
#    print(param.data)

