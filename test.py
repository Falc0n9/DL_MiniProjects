from torch import Tensor,FloatTensor, LongTensor
from mini_project_2 import *
from math import sqrt, pi
import matplotlib.pyplot as plt
from plot import plot_data
from dlc_practical_prologue import convert_to_one_hot_labels

train_data = Tensor(10000,2).uniform_()
train_target = (train_data.pow(2).sum(1) < 1 / (2 * pi)).long()
train_target_one_hot = convert_to_one_hot_labels(train_data,train_target)

seq = Sequential(Linear(2,25),
                 ReLU(),
                 Linear(25, 25),
                 ReLU(),
                 Linear(25, 25),
                 ReLU(),
                 Linear(25, 2))

lr, nb_epochs, batch_size = 0.01, 50, 10

loss_list = []
for e in range(0,nb_epochs):
    sum_loss = 0
    for b in range(0, train_data.size(0), batch_size):

        data = train_data.narrow(0, b, batch_size)
        target = train_target_one_hot.narrow(0,b,batch_size)

        seq.zero_grad()
        loss, loss_grad = loss_mse(seq.forward(data), target)
        seq.backward(loss_grad)
        sum_loss = sum_loss + loss
        for p in seq.param():
            p.data.sub_(lr * p.grad)
    loss_list += [sum_loss]

plt.plot(loss_list)
plt.show()

output = seq.forward(train_data)
_, predicted_classes = output.max(1)
plot_data(train_data, predicted_classes)

